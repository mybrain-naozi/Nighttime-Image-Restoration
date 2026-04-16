# 夜间图像复原项目

## 一键运行

直接运行：

```bash
python -B run_project.py
```

可选参数：

```bash
python -B run_project.py --mode real
python -B run_project.py --mode synthetic
python -B run_project.py --limit 3
```

## 数据集目录

```text
our485/
  low/
  high/
eval15/
  low/
  high/
```

## 结果怎么看

每次运行都会重新生成 `results/`，里面只保留这些内容：

- `00_先看这个.txt`
- `01_真实低照度汇总.csv`
- `02_人工退化汇总.csv`
- `03_人工退化分类汇总.csv`
- `04_真实低照度逐图结果.csv`
- `05_人工退化逐图结果.csv`
- `06_真实低照度对比图/`
- `07_人工退化对比图/`

先打开 `00_先看这个.txt`。
