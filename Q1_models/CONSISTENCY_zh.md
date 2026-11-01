# 代码与最新报告数学模型的一致性核对

本次以用户粘贴的 `Mathematical formulations and problem classes` 为准，而非沿用旧版本的建模选择。先记录修订规划，再修改统一模型，最后重新求解。

## 修订结论

旧代码有一处实质差异：日末库存约束是等式。已改为正文的 `e[i,23]>=initial`，默认名为 `minimum`。此外，将两种数学等价写法改为与正文同形：使用 d_plus/d_minus 拆分偏离，直接最小化 J。净需求 q 显式设为可正可负的变量，通过物理定义等式连接到资产调度。

| 正文定义/约束 | 当前实现 | 核对结论 |
|---|---|---|
| eq:q1-reference-load | load_data: ref=max_load*capacity_factor | 一致 |
| eq:q1-load-lower、eq:q1-load-upper | lower/upper -> L.LB/UB | 一致，小时偏离 1 kWh、用电上限 3 |
| eq:q1-daily-minimum-parameter、eq:q1-daily-energy | minimum=sum(ref)，daily_minimum >= | 一致；不是日用电强制等式 |
| eq:q1-hourly-disutility、eq:q1-deviation-split | d_plus,d_minus>=0，deviation_split 等式 | 一致；不再用 z epigraph |
| eq:q1-pv-bounds | P 下限 0，上限 a.pv[t] | 一致，允许弃光 |
| eq:q1-charge-bounds、eq:q1-discharge-bounds | ch/dis 非负，UB 为成员功率限制 | 一致，隐含 dt=1 h |
| eq:q1-initial-dynamics、eq:q1-storage-dynamics | soc 等式：充电乘效率、放电除效率 | 一致 |
| eq:q1-storage-bounds | e>=0，UB=capacity | 一致 |
| eq:q1-terminal-condition | terminal='minimum': e[i,23]>=initial | 已修正；cyclic/free 仅显式诊断选项 |
| eq:q1-net-consumption | q 为自由变量；net_consumption 定义等式 | 一致；负 q 允许净盈余 |
| eq:q1-full-prices、eq:q1-discounted-prices | p+discount*tau_imp；p-discount*tau_exp | 一致，折扣不作用于日前价格 |
| eq:q1-R-objective、eq:q1-R-balance | min J；q-I+E=0；无罚金和合同 cap 约束 | 一致；成员独立计量 |
| eq:q1-reference-net-import、eq:q1-cap-share | R 日净进口的正部归一化 | 一致；分母非正时明确报错 |
| eq:q1-I-objective、eq:q1-I-cap | min J；I-x<=kappa*cap；非负 x | 一致；个人超限罚金不混同于馈线超限 |
| eq:q1-C-objective、eq:q1-community-balance | min J；sum(q)-P_imp+P_exp=0 | 一致；只有公共电表对外计费 |
| eq:q1-community-cap | P_imp-x<=cap，x>=0 | 一致；软上限 |
| Problem classes and consequences for duality | 全部连续变量、线性目标与约束 | 一致；每个基准 LP 可行有界并检查强对偶 |

代码中的 L 对应正文 ell，rho_ch/rho_dis 对应 rho 的充放电效率，c 对应 c_i，beta/alpha 对应带 DSO 上标的合同参数。额外输出 W=-J 只为兼容福利比较，求解器实际目标是最小化 J。

## 正文中建议澄清的一处符号

正文将 `q=q_imp-q_exp=物理净需求` 全部纳入共同可行集 X_i。对 R/I 可令 q_imp=I、q_exp=E；但 C 没有独立对外计量的 I/E。建议共同部分仅定义

```latex
q_{i,t}=\ell_{i,t}-P_{i,t}
       +p^{\rm ch}_{i,t}-p^{\rm dis}_{i,t},
\qquad q_{i,t}\in\mathbb R.
```

然后在 R/I 中保留 `q-I+E=0`。C 只保留社区平衡。这样可以避免误读为 C 也对每个成员收取外部电网费。若保留 C 中成员 gross import/export 作为辅助量，它们不产生费用，消去这些辅助变量后与当前代码的可行集等价。

正文将 Lmax=3 kWh/h 后隐含乘以 1 h 的说明已经给出；实现固定 dt=1。更换时间步长时必须显式转换，不能直接复用数值。

## 对偶与内部价格

目标从最大化 -J 改为最小化 J 后，已同步修改最优性检查。Gurobi 对最小化问题的 Pi 在 <= 行非正、>= 行非负；RC>0 对应下界，RC<0 对应上界。驻点检查为 `c-A^T Pi-RC=0`，强对偶包含变量上下界的贡献。

Gurobi Pi 是 RHS 敏感度，不应直接等同于所有写法下的拉格朗日乘子。按正文平衡 `sum(q)-P_imp+P_exp=0`，内部经济价格为 `-Pi`；因此当前输出仍使用这个负号。后续 Q2 写 KKT 时必须沿用本次的最小化及平衡方向。

## 数值验证与差异解释

`consistency_checks.py` 从解和输入参数直接重算正文各类方程、日最低用电、偏离、资产界限、日末下限、计量平衡、容量约束和 J，不只依赖求解器显示 optimal。`base_case.py` 还检查 KKT、强对偶、HiGHS 独立复解、无合同极限以及 J_C<=J_I。

基准结果：J_R=398.241694、J_I=480.669430、J_C=372.162434 DKK；峰值 27.6、11.082393、9.39 kW；超限小时 9、3、0。基准所有电池日末库存恰好等于初始值，因此把旧等式改为新不等式后，本次三种模型的最优成本没有变化。

但这不证明两个日末模型等价。具名诊断取一个带电池的成员，初始库存设为 0，末小时价格设为 -5 DKK/kWh，其他小时为 0：下限模型允许末库存严格大于初始值，最优成本低于循环等式模型。该诊断在数据副本上运行，不修改基准文件。结果保存到 validation_summary.json 的 terminal_condition_check。

本轮 R 返回 66.702105 kWh 超限，旧轮为 66.302105；最优成本不变。这是最优调度不唯一的表现，不是用电或合同计费错误。`audit_reference_ties.py` 固定当前最优 J 后也能得到这两个超限值；官方记录 66.5 位于其间。不能要求不同等价变量写法或求解器返回同一调度和同一组对偶。

## 执行与范围

```powershell
python base_case.py
python audit_reference_ties.py
```

Notebook 包含重新执行后的输出，results/base_case/ 是当前正文基准结果。results/cyclic/ 若仍存在，只代表旧/显式循环诊断，不能当本次正文基准引用。交付 ZIP 不打包这批旧结果。本次仍只覆盖 Q1 steps 1–3；完整 Q1 的敏感性分析、预算平衡、成员个体理性须后续完成。
