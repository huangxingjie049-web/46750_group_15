"""Explore reference dispatch multiplicity while preserving minimum cost J."""
import json
from main import ROOT, CommunityModel, load_data, gp, GRB, TOL


def main():
    """Find two R optimal schedules using a secondary linear tie-breaker.

    The secondary objective sums imports over the original congested hours.
    We record actual positive-part exceedance over all 24 hours afterwards;
    no claim is made that this linear proxy globally bounds that metric.
    Continuity along convex combinations establishes the attainable interval
    between the recorded exceedances of these two optimal schedules.
    """
    data=load_data(ROOT/'data')
    model=CommunityModel(data,'R')
    result=model.run()
    m=model.model
    cost=m.getObjective()
    optimum=m.ObjVal
    m.addConstr(cost == optimum, name='preserve_optimal_cost')
    hours=[r['hour'] for r in result.hourly if r['feeder_exceedance_kWh']>TOL]
    secondary=gp.quicksum(model.variables.imp[a.id,t] for a in data.members for t in hours)
    audits=[]
    for name,sense in [('min_import_in_original_congested_hours',GRB.MINIMIZE),
                       ('max_import_in_original_congested_hours',GRB.MAXIMIZE)]:
        m.setObjective(secondary,sense)
        m.optimize()
        assert m.Status == GRB.OPTIMAL
        exceedance=sum(max(0,sum(model.variables.imp[a.id,t].X for a in data.members)
                           -model.cap[t]) for t in range(24))
        audits.append(dict(secondary_objective=name,J_DKK=cost.getValue(),
                           feeder_exceedance_kWh=exceedance))
    payload=dict(optimal_cost_DKK=optimum, terminal=model.terminal, original_congested_hours=hours,
        tie_schedules=audits, provided_exceedance_kWh=66.5,
        interpretation='Recorded schedules are equally optimal; their exceedances need not be identical.')
    out=ROOT/'results/minimum/reference_tie_audit.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
