# 放进新的 Git 分支

这份代码包按最新报告的数学公式和实验设计整理。只有根目录 `main.py`
是运行入口；不需要 `run_q2c.py`、`run_q3.py` 等平行脚本。

## 建议步骤

1. 先保存现有改动（提交或备份）。在你自己的仓库新建分支，例如：

   ```bash
   git switch -c integration/report-reproducible
   ```

2. 解压本包到单独的临时文件夹，先阅读 README 和 `REPORT_ALIGNMENT.md`。
3. **保留你原仓库的 `.git`、LaTeX 报告、个人未提交资料。** 本包不是完整 Git
   仓库，没有必要重新初始化 Git。
4. 将旧 `src/`、`tests/` 先备份到仓库以外，再用包中的同名文件夹完整替换。
   更新 main.py、README.md、依赖文件、pytest.ini 和 `.gitignore`。
5. 包中 `data/` 来源于你上传的基准数据，原始字节已保留。不要把其他实验改过的
   JSON 覆盖进来。旧根目录 `run_*.py`、`estimate_q3g_pi.py`、线性模型入口及
   根目录旧模型文件可移出工作树；新包不再引用它们。**不要误删报告文件。**
6. `docs/` 如已有其他文件，可合并；本包新增文件使用明确名称。生成的 `results/`
   用本包结果或从头生成，不要混进旧脚本的敏感性结果。
7. 按 README 建立 Python 3.12 环境后运行：

   ```bash
   python -m pytest -q
   python main.py --task all
   ```

8. 检查 Git diff，确认数据哈希测试通过，报告所用数字与新结果一致，再提交：

   ```bash
   git status
   git diff --stat
   git add -A
   git commit -m "Unify report models, experiments and validation"
   ```

   `git add -A` 会记录已确认的旧文件删除；先检查变动，避免加入无关文件。
   推送/PR由你在确认新分支内容后执行。

## 重新运行某一题

例如第三问：

```bash
python main.py --task 3e
python main.py --task 3f
python main.py --task 3g
python main.py --task 3g-sensitivity
```

运行一次对应题目会保存当次的数据、结果、dual、验证及图表。全量敏感性包含
数百次求解，是为了复现曲线、乘子和饱和点，不代表报告要放所有表格。

## 对报告的处理

- 公式/图引用错误仍需在 LaTeX 修复；代码包没有编辑你的报告。
- 旧 Q2(e) 漏记耦合惩罚的总成本、目标值应使用新结果；公式保持报告的加号。
- Q2(d) 小时8的二次基准是1.15 kWh，不能写成约1.6。
- 读者应明确得到每题实现文件及本包命令；不要继续写旧 run_q*.py 命令。
- 循环方案与期末价值方案可以使用比较CSV，不要将不同库存条件的改善当成
  同一个可重复的每日投资收益。
