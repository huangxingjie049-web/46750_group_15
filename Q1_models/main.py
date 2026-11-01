"""Assignment 2, Q1 first deliverable: deterministic R/I/C LPs.

Run: python main.py --variant all
Requires gurobipy and a working Gurobi licence. No input data are modified.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parent
TOL = 2e-6


def load_data(folder):
    """Load the supplied d1 catalogue, community and contract (kWh/kW/DKK)."""
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    app = read("appliance_params.json")
    community = read("params_community_base.json")
    bus = read("bus_params.json")[0]
    contract = read("dso_contract_params.json")
    loads = {x["load_id"]: x for x in app["loads"]}
    pvs = {x["DER_id"]: x for x in app["DERs"]}
    batteries = {x["storage_id"]: x for x in app["storages"]}
    members = []
    for item in community["consumers"]:
        ids = item["list_appliances"]
        load = next(loads[k] for k in ids if k in loads)
        pv = next((pvs[k] for k in ids if k in pvs), None)
        bat = next((batteries[k] for k in ids if k in batteries), None)
        ref = [load["max_load_KWh_per_hour"] * r
               for r in load["reference_load_capacity_factor"]]
        deviation = load["max_hourly_deviation_kWh"]
        members.append(SimpleNamespace(
            id=item["consumer_id"], cL=load["linear_disutility_DKK_per_kWh"],
            ref=ref, minimum=sum(ref),
            lower=[max(load["min_load_KWh_per_hour"], r-deviation) for r in ref],
            upper=[min(load["max_load_KWh_per_hour"], r+deviation) for r in ref],
            pv=[pv["max_power_kW"] * r for r in pv["hourly_profile_ratio"]]
               if pv else [0.0] * 24,
            c=pv["marginal_cost_DKK_per_kWh"] if pv else 0.0,
            capacity=bat["storage_capacity_kWh"] if bat else 0.0,
            charge=bat["max_charging_power_kW"] if bat else 0.0,
            discharge=bat["max_discharging_power_kW"] if bat else 0.0,
            rho_ch=bat["charging_efficiency"] if bat else 1.0,
            rho_dis=bat["discharging_efficiency"] if bat else 1.0,
            initial=bat["storage_capacity_kWh"] * bat["initial_state_of_charge_ratio"]
                    if bat else 0.0))
    return SimpleNamespace(members=members, prices=bus["energy_price_DKK_per_kWh"],
        tau_imp=bus["import_tariff_DKK_per_kWh"],
        tau_exp=bus["export_tariff_DKK_per_kWh"], contract=contract)


class CommunityModel:
    """One parametrized model; switches change metering and contract costs.

    All hourly flows are kWh, battery powers kW, dt=1 h; J is in DKK.
    Default terminal condition is e[i,23] >= initial, as in the report.
    'cyclic' and 'free' are explicitly named diagnostic variants.
    Discounted tariffs apply to all metered imports/exports, plus an additive
    exceedance penalty, following the supplied contract JSON and README.
    """
    def __init__(self, data, variant="R", *, shares=None, terminal="minimum",
                 beta=None, alpha=None, cap=None):
        if variant not in ("R", "I", "C"):
            raise ValueError("variant must be R, I or C")
        self.data, self.variant, self.terminal = data, variant, terminal
        self.shares = shares
        self.beta = data.contract["tariff_discount_ratio"] if beta is None else beta
        self.alpha = data.contract["penalty_DKK_per_kWh"] if alpha is None else alpha
        self.cap = data.contract["capacity_limit_kW"] if cap is None else cap
        ids = {a.id for a in data.members}
        if not 0 <= self.beta <= 1 or self.alpha < 0:
            raise ValueError('Require beta in [0,1] and alpha >= 0')
        if len(self.cap) != 24 or min(self.cap) < 0:
            raise ValueError('Require 24 nonnegative hourly capacity values')
        if variant == "I" and (shares is None or set(shares) != ids or abs(sum(shares.values())-1) > 1e-9):
            raise ValueError("Individual caps need nonnegative shares summing to one")
        if shares is not None and min(shares.values()) < 0:
            raise ValueError("Negative net-import shares require another justified rule")
        self.variables = SimpleNamespace()
        self.constraints = SimpleNamespace()

    def build(self):
        """Build the primal LP, keeping balance/cap constraint handles for Pi."""
        d, v, rows = self.data, self.variables, self.constraints
        m = self.model = gp.Model("Assignment2_" + self.variant)
        m.Params.OutputFlag = 0
        m.Params.Method = 0  # primal simplex, as in Assignment 1
        m.Params.FeasibilityTol = 1e-9
        m.Params.OptimalityTol = 1e-9
        ids, hours = [x.id for x in d.members], range(24)
        v.L = m.addVars(ids, hours, name="L")
        v.P = m.addVars(ids, hours, name="P")
        v.d_plus = m.addVars(ids, hours, name="d_plus")
        v.d_minus = m.addVars(ids, hours, name="d_minus")
        v.net = m.addVars(ids, hours, lb=-GRB.INFINITY, name="q")
        v.ch = m.addVars(ids, hours, name="p_ch")
        v.dis = m.addVars(ids, hours, name="p_dis")
        v.e = m.addVars(ids, hours, name="e")
        rows.balance, rows.cap, rows.terminal = {}, {}, {}
        private_cost = gp.LinExpr()
        for a in d.members:
            i = a.id
            for t in hours:
                v.L[i,t].LB, v.L[i,t].UB = a.lower[t], a.upper[t]
                v.P[i,t].UB = a.pv[t]
                v.ch[i,t].UB, v.dis[i,t].UB = a.charge, a.discharge
                v.e[i,t].UB = a.capacity
                m.addConstr(v.L[i,t]-a.ref[t] == v.d_plus[i,t]-v.d_minus[i,t],
                            name=f"deviation_split[{i},{t}]")
                previous = a.initial if t == 0 else v.e[i,t-1]
                m.addConstr(v.e[i,t] == previous + a.rho_ch*v.ch[i,t]
                            - v.dis[i,t]/a.rho_dis, name=f"soc[{i},{t}]")
                m.addConstr(v.net[i,t] == v.L[i,t]+v.ch[i,t]-v.P[i,t]-v.dis[i,t],
                            name=f"net_consumption[{i},{t}]")
                private_cost += a.cL*(v.d_plus[i,t]+v.d_minus[i,t]) + a.c*v.P[i,t]
            m.addConstr(gp.quicksum(v.L[i,t] for t in hours) >= a.minimum,
                        name=f"daily_minimum[{i}]")
            if self.terminal == "minimum":
                rows.terminal[i] = m.addConstr(v.e[i,23] >= a.initial, name=f"terminal[{i}]")
            elif self.terminal == "cyclic":
                rows.terminal[i] = m.addConstr(v.e[i,23] == a.initial, name=f"terminal[{i}]")
            elif self.terminal != "free":
                raise ValueError("terminal must be minimum, cyclic or free")
        if self.variant in ("R", "I"):
            v.imp = m.addVars(ids, hours, name="I")
            v.exp = m.addVars(ids, hours, name="E")
            if self.variant == "I":
                v.excess = m.addVars(ids, hours, name="x")
            grid_cost = gp.LinExpr()
            for i in ids:
                for t in hours:
                    rows.balance[i,t] = m.addConstr(
                        v.net[i,t]-v.imp[i,t]+v.exp[i,t] == 0, name=f"balance[{i},{t}]")
                    discount = 1 if self.variant == "R" else 1-self.beta
                    grid_cost += ((d.prices[t]+discount*d.tau_imp)*v.imp[i,t]
                                  -(d.prices[t]-discount*d.tau_exp)*v.exp[i,t])
                    if self.variant == "I":
                        rows.cap[i,t] = m.addConstr(v.imp[i,t]-v.excess[i,t]
                            <= self.shares[i]*self.cap[t], name=f"cap[{i},{t}]")
                        grid_cost += self.alpha*v.excess[i,t]
        else:
            v.imp = m.addVars(hours, name="P_imp")
            v.exp = m.addVars(hours, name="P_exp")
            v.excess = m.addVars(hours, name="x")
            grid_cost = gp.LinExpr()
            for t in hours:
                rows.balance[t] = m.addConstr(gp.quicksum(v.net[i,t] for i in ids)
                    -v.imp[t]+v.exp[t] == 0, name=f"community_balance[{t}]")
                rows.cap[t] = m.addConstr(v.imp[t]-v.excess[t] <= self.cap[t], name=f"cap[{t}]")
                grid_cost += ((d.prices[t]+(1-self.beta)*d.tau_imp)*v.imp[t]
                    -(d.prices[t]-(1-self.beta)*d.tau_exp)*v.exp[t]
                    + self.alpha*v.excess[t])
        self.private_cost, self.grid_cost = private_cost, grid_cost
        m.setObjective(private_cost+grid_cost, GRB.MINIMIZE)

    def run(self):
        """Solve; independently audit physical/accounting and LP dual residuals."""
        self.build()
        m, d, v = self.model, self.data, self.variables
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"No optimal solution: status {m.Status}")
        members, hourly = [], []
        physical = 0.0
        for a in d.members:
            i = a.id
            for t in range(24):
                previous = a.initial if t == 0 else v.e[i,t-1].X
                physical = max(physical, abs(v.e[i,t].X-previous-a.rho_ch*v.ch[i,t].X
                                            +v.dis[i,t].X/a.rho_dis))
                members.append(dict(member=i, hour=t, L_kWh=v.L[i,t].X,
                    PV_kWh=v.P[i,t].X, net_kWh=v.net[i,t].X,
                    charge_kW=v.ch[i,t].X, discharge_kW=v.dis[i,t].X,
                    soc_kWh=v.e[i,t].X,
                    d_plus_kWh=v.d_plus[i,t].X, d_minus_kWh=v.d_minus[i,t].X,
                    deviation_kWh=v.d_plus[i,t].X+v.d_minus[i,t].X))
                members[-1].update(reference_kWh=a.ref[t],
                    load_change_kWh=v.L[i,t].X-a.ref[t],
                    PV_available_kWh=a.pv[t],
                    import_kWh=v.imp[i,t].X if self.variant != 'C' else None,
                    export_kWh=v.exp[i,t].X if self.variant != 'C' else None,
                    charged_exceedance_kWh=v.excess[i,t].X if self.variant == 'I' else None)
        for t in range(24):
            if self.variant == "C":
                imp, exp = v.imp[t].X, v.exp[t].X
                excess = v.excess[t].X
                price = -self.constraints.balance[t].Pi
            else:
                imp = sum(v.imp[a.id,t].X for a in d.members)
                exp = sum(v.exp[a.id,t].X for a in d.members)
                excess = sum(v.excess[a.id,t].X for a in d.members) if self.variant=="I" else 0
                price = None
            hourly.append(dict(hour=t, import_kWh=imp, export_kWh=exp,
                cap_kW=self.cap[t], feeder_exceedance_kWh=max(imp-self.cap[t],0),
                charged_exceedance_kWh=excess, internal_price_DKK_per_kWh=price))
        # LP certificate using solver Pi and reduced costs, including variable bounds.
        dual = sum(row.Pi*row.RHS for row in m.getConstrs())
        complement = max(abs(row.Pi*row.Slack) for row in m.getConstrs())
        dual_sign = max([0.0]+[max(0,r.Pi) if r.Sense=='<' else max(0,-r.Pi)
                        if r.Sense=='>' else 0 for r in m.getConstrs()])
        bound_sign = 0.0
        for var in m.getVars():
            rc = var.RC
            bound = var.LB if rc >= 0 else var.UB
            if abs(bound) >= GRB.INFINITY:
                bound_sign = max(bound_sign,abs(rc))
            else:
                dual += rc*bound
                complement = max(complement,abs(rc*(var.X-bound)))
        stationarity = 0.0
        for var in m.getVars():
            col = m.getCol(var)
            residual = var.Obj-var.RC-sum(col.getCoeff(k)*col.getConstr(k).Pi
                                        for k in range(col.size()))
            stationarity = max(stationarity,abs(residual))
        certificate = dict(primal_violation=max(m.ConstrVio,m.BoundVio,physical),
            stationarity_residual=stationarity, dual_sign_violation=max(dual_sign,bound_sign),
            complementarity_residual=complement, primal_dual_gap=abs(dual-m.ObjVal),
            reconstructed_objective_error=abs(m.ObjVal-self.private_cost.getValue()
                                               -self.grid_cost.getValue()))
        assert max(certificate.values()) < TOL, certificate
        overlap = sum(r['charge_kW']>TOL and r['discharge_kW']>TOL for r in members)
        trade_overlap = (sum(v.imp[t].X>TOL and v.exp[t].X>TOL for t in range(24))
            if self.variant=='C' else sum(v.imp[a.id,t].X>TOL and v.exp[a.id,t].X>TOL
                                         for a in d.members for t in range(24)))
        # Report overlaps rather than rejecting valid optima in degenerate
        # parameter variants (e.g. zero tariff spread). Base checks are separate.
        energy_cost = sum(d.prices[r['hour']]*(r['import_kWh']-r['export_kWh']) for r in hourly)
        discount = 1 if self.variant == 'R' else 1-self.beta
        tariff_cost = discount*sum(d.tau_imp*r['import_kWh']+d.tau_exp*r['export_kWh'] for r in hourly)
        disutility_cost = sum(a.cL*(v.d_plus[a.id,t].X+v.d_minus[a.id,t].X)
                              for a in d.members for t in range(24))
        pv_cost = sum(a.c*v.P[a.id,t].X for a in d.members for t in range(24))
        self.results = SimpleNamespace(members=members, hourly=hourly, certificate=certificate,
            metrics=dict(variant=self.variant, terminal=self.terminal,
                J_DKK=m.ObjVal, W_DKK=-m.ObjVal, total_cost_DKK=m.ObjVal,
                private_cost_DKK=self.private_cost.getValue(), grid_cost_DKK=self.grid_cost.getValue(),
                energy_cost_DKK=energy_cost, tariff_cost_DKK=tariff_cost,
                disutility_cost_DKK=disutility_cost, PV_cost_DKK=pv_cost,
                daily_import_kWh=sum(r['import_kWh'] for r in hourly),
                daily_export_kWh=sum(r['export_kWh'] for r in hourly),
                peak_import_kW=max(r['import_kWh'] for r in hourly),
                feeder_exceedance_kWh=sum(r['feeder_exceedance_kWh'] for r in hourly),
                charged_exceedance_kWh=sum(r['charged_exceedance_kWh'] for r in hourly),
                penalty_DKK=self.alpha*sum(r['charged_exceedance_kWh'] for r in hourly),
                hours_above_cap=sum(r['feeder_exceedance_kWh']>TOL for r in hourly),
                max_exceedance_kW=max(r['feeder_exceedance_kWh'] for r in hourly),
                cap_binding_hours=sum(abs(r['import_kWh']-self.cap[r['hour']])<TOL for r in hourly),
                battery_overlap_hours=overlap, trade_overlap_hours=trade_overlap,
                variables=m.NumVars, constraints=m.NumConstrs,
                gurobi_version=list(gp.gurobi.version())))
        return self.results

    def save(self, folder):
        """Save dispatch, objective components, duals and certificate for reproduction."""
        folder.mkdir(parents=True, exist_ok=True)
        for name, records in [('members',self.results.members),('hourly',self.results.hourly)]:
            with (folder/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        for name, value in [('metrics',self.results.metrics),('validation',self.results.certificate)]:
            (folder/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
        duals = [dict(name=r.ConstrName, sense=r.Sense, RHS=r.RHS, Pi=r.Pi, slack=r.Slack)
                 for r in self.model.getConstrs()]
        with (folder/'duals.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(duals[0]));writer.writeheader();writer.writerows(duals)
        self.model.write(str(folder/'model.lp'))


def net_import_shares(reference):
    """Compute fixed kappa_i from optimal R daily net imports before solving I."""
    net={}
    for row in reference.members:
        net[row['member']]=net.get(row['member'],0.0)+row['net_kWh']
    positive = {i:max(0,x) for i,x in net.items()}
    denominator = sum(positive.values())
    if denominator <= TOL:
        raise ValueError('Net-import split is undefined without any positive daily net importer')
    return {i:x/denominator for i,x in positive.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['R','I','C','all'],default='all')
    parser.add_argument('--terminal',choices=['minimum','cyclic','free'],default='minimum')
    args=parser.parse_args()
    data=load_data(ROOT/'data')
    reference_model=CommunityModel(data,'R',terminal=args.terminal)
    reference=reference_model.run()
    shares=net_import_shares(reference)
    out=ROOT/'results'/args.terminal
    out.mkdir(parents=True,exist_ok=True)
    (out/'shares.json').write_text(json.dumps(shares,indent=2),encoding='utf-8')
    results={}
    for variant in ['R','I','C'] if args.variant=='all' else [args.variant]:
        model=reference_model if variant=='R' else CommunityModel(data,variant,shares=shares,terminal=args.terminal)
        result=reference if variant=='R' else model.run()
        model.save(out/variant)
        results[variant]=result.metrics
        print(json.dumps(result.metrics,indent=2))
    if args.variant=='all':
        assert results['C']['W_DKK'] >= results['I']['W_DKK']-TOL
        # Required limit case: with separate metering, no discount and no cap,
        # centralized optimization is separable and reproduces R.
        limit=CommunityModel(data,'I',shares=shares,terminal=args.terminal,
                             beta=0,alpha=0,cap=[1e5]*24)
        limit_result=limit.run()
        assert abs(limit_result.metrics['W_DKK']-reference.metrics['W_DKK'])<TOL
        summary=dict(results=results, limit_case_objective_error_DKK=abs(
            limit_result.metrics['W_DKK']-reference.metrics['W_DKK']),
            explanation='C net meter with beta=0 and no cap need not equal R; restore separate metering.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
