"""Build the explanatory Q1 notebook; execution is done with Jupyter/nbclient."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))

md(r'''# Assignment 2 — Q1: base-case models R, I and C

本 Notebook 运行三个 Q1 基准模型。模型源代码在 `main.py`，实验、独立求解检查和成员结算在 `base_case.py`。

**要回答的问题**：个人合同能减少多少拥塞？社区共享容量后，拥塞与总成本如何变化？

展示 24 小时进口与容量图、关键指标表、负荷、电池和 PV 响应。

资料：A2 pp.4–5、8（Tables 8–9）、10–11；教师 A1 Solutions §§2.1、3.2；教师对应线性不适成本和电池代码；本组 A1 报告与实现。''')
code('''from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / 'base_case.py').exists():
    ROOT = ROOT / 'Q1_models'
if not (ROOT / 'base_case.py').exists():
    raise RuntimeError('Open this notebook from Q1_models or the repository root.')
sys.path.insert(0, str(ROOT.resolve()))
import pandas as pd
from IPython.display import display, Image, Markdown
from base_case import run_base_case
from main import load_data
data = load_data(ROOT / 'data')
display(pd.DataFrame([dict(member=a.id, cL=a.cL, daily_min_kWh=a.minimum,
    PV_kW=max(a.pv), battery_kWh=a.capacity, initial_SOC_kWh=a.initial)
    for a in data.members]).rename(columns={'PV_kW':'peak_available_PV_kWh'}))''')
md(r'''## 共同约束与模型选择

每个成员的净需求 $q_{i,t}=\ell_{i,t}-P_{i,t}+p^{\rm ch}_{i,t}-p^{\rm dis}_{i,t}$。

$$\max(0,\ell^{\rm ref}_{i,t}-1)\le\ell_{i,t}\le
\min(3,\ell^{\rm ref}_{i,t}+1),\quad
\sum_t\ell_{i,t}\ge E_i^{\min}.$$

偏离拆分 $\ell_{i,t}-\ell^{\rm ref}_{i,t}=d^+_{i,t}-d^-_{i,t}$、$d^+_{i,t},d^-_{i,t}\ge0$，不适成本 $c_i^L(d^+_{i,t}+d^-_{i,t})$。PV 上限为输入可用量，可弃光，实际产量支付边际成本。

$$e_{i,t}=e_{i,t-1}+\rho_i^{\rm ch}p^{\rm ch}_{i,t}
-p^{\rm dis}_{i,t}/\rho_i^{\rm dis},\qquad e_{i,23}\ge E_i^0.$$

SOC、充放电均有输入容量边界。1 h 时间步：小时流量为 kWh，电池功率与容量上限为 kW，数值上乘以 1 h。

三种情形均使用正文的日末 SOC 下限，与教师参考一致。循环 SOC 只保留为诊断选项；基准末库存取等不代表两个模型相同。A2 使用线性不适成本，因此三者都是 LP；可行有界时强对偶成立，KKT 为最优性条件。基准检查互斥，无需引入整数变量。''')
md(r'''## R、I、C：一套类，改变计量和合同参数

成员的 PV 成本与负荷偏离成本之和为
$F_i=\sum_t[c_iP_{i,t}+c_i^L(d^+_{i,t}+d^-_{i,t})]$。

**R**：$q_{i,t}-I_{i,t}+E_{i,t}=0$，最小化
$\sum_i F_i+\sum_{i,t}(a_tI_{i,t}-b_tE_{i,t})$。各成员独立计量和调度。

**I**：由 R 求出固定份额 $\kappa_i=n_i^{\mathcal R}/\sum_j n_j^{\mathcal R}$，
其中 $n_i^{\mathcal R}=[\sum_t (I_{i,t}^{\mathcal R}-E_{i,t}^{\mathcal R})]_+$。
式 (15c) 写为 $q_{i,t}-I^{\rm cap}_{i,t}-x_{i,t}+E_{i,t}=0$，
式 (15d) 为 $0\le I^{\rm cap}_{i,t}\le\kappa_i\overline P_t^{\rm DSO}$。
总进口为 $I_{i,t}=I^{\rm cap}_{i,t}+x_{i,t}$。限额内进口按
$a_t^{\rm d}$ 计费，超限进口按 $p_t+\alpha^{\rm DSO}$ 计费。

**C**：式 (17c) 写为 $\sum_i q_{i,t}-P_t^{\rm imp,cap}-x_t+P_t^{\rm exp}=0$，式 (17d) 为 $0\le P_t^{\rm imp,cap}\le\overline P_t^{\rm DSO}$。最小化
$$J^{\mathcal C}=\sum_i F_i+\sum_t[a_t^{\rm d}P_t^{\rm imp,cap}
-b_t^{\rm d}P_t^{\rm exp}+(p_t+\alpha^{\rm DSO})x_t].$$
$x_t\ge0$，公共电表总进口为 $P_t^{\rm imp}=P_t^{\rm imp,cap}+x_t$。程序输出 $W=-J$ 用于福利比较。

基准 $\beta=0.75$、$\alpha=10$ DKK/kWh、$cap_t=9.39$ kW。折扣电网费适用于限额内进口和出口。日前电价适用于全部进口和出口，超限进口另加罚金。

**先定义指标**：R/I 的馈线进口为各成员 gross import 之和；C 为公共电表进口。$E^{exc}=\sum_t[import_t-cap_t]_+$ kWh；超限小时按 $2\times10^{-6}$ 容差计数。I 的收费超额与馈线超限分别记录。收益 $J^R-J$，正值为整体节省。''')
code('''data, models, results, validation = run_base_case()
fields = ['variant','total_cost_DKK','gain_vs_R_DKK','peak_import_kW',
          'feeder_exceedance_kWh','hours_above_cap','charged_exceedance_kWh','penalty_DKK']
table = pd.DataFrame(validation['results']).set_index('variant')
display(table[[f for f in fields if f != 'variant']].round(6))''')
code("display(Image(filename=str(ROOT/'results/base_case/feeder_comparison.png')))")
md('''## 解释调度机制与成本

曲线检查负荷是否移出拥塞时段、电池净充电是否移到其他时段，以及 PV 利用量是否变化。共同合同减少拥塞的同时仍可能增加日进口或改变 PV 出力，因此只看总进口不能评价容量服务。成本必须区分电能、电网费、罚金、不适和 PV 成本。''')
code('''display(Image(filename=str(ROOT/'results/base_case/dispatch_response.png')))
display(table[['energy_cost_DKK','tariff_cost_DKK','penalty_DKK',
               'disutility_cost_DKK','PV_cost_DKK','daily_import_kWh',
               'daily_export_kWh','cap_binding_hours']].round(6))''')
md('''## 验证：数值最优不等于模型已被证实

以下同时检查 Gurobi 原始可行性/对偶/KKT 残差、相同 LP 在 HiGHS 的最优目标、全额费率且容量不绑定时还原 R、C 成本不高于 I、成本分解求和、基准无同时充放和同表进口出口。另按报告公式独立重构成本。

官方 R 的 173.4 kWh、27.6 kW、398.24 DKK 与本结果在其精度下一致。66.5 kWh 超限依赖多重最优调度；不能强制每次得到同一个调度或对偶。下面保持 R 最优目标不变寻找替代最优解，验证这一解释。另按正文逐式复算残差，并用末小时负电价反例确认程序允许末库存严格大于初始库存。''')
code('''display(pd.DataFrame({k:r.certificate for k,r in results.items()}).T)
display(pd.DataFrame(validation['independent_solver']).T)
display(pd.DataFrame(validation['report_equation_checks']).T)
display(validation['terminal_condition_check'])
print('Full-tariff, nonbinding-cap objective error (DKK):', validation['no_contract_limit_error_DKK'])
import audit_reference_ties
audit_reference_ties.main()''')
md('''## 基准结论与口试准备

- R 的峰值为 27.6 kW，9 小时超限；I 将超限显著降低但仍保留 3 小时，且整体广义成本高于 R。签约不是必然省钱。
- C 的峰值降至 9.39 kW，没有超限，成本低于 R 和 I；共享容量和电表净额计量共同发挥作用。该成本结论不代表每个成员结算后都获益。
- 日末 SOC 下限、软上限、线性不适成本、份额来源、计费解释和总成本的组成，都是需要能够解释的选择。
- 后续沿用 `CommunityModel` 的成员输入、净需求和社区平衡；Q2/Q3 还需新增价格决策、成员最优响应或情景结构，当前并未实现。

输出位于 `results/base_case/`。社区逐时对偶价格见 `C/hourly.csv`，成员电费、总成本和个体理性见 `member_costs.csv`，管理者收支见 `settlement_summary.json`。敏感性分析单独运行。''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nb.write(book,ROOT/'Assignment2_Q1_base_case.ipynb')
print(ROOT/'Assignment2_Q1_base_case.ipynb')
