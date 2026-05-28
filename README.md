# 夜间监控图像模糊复原

本项目面向夜间监控场景中的图像复原问题，使用 LLVIP 可见光夜间监控图像作为原始参考图，分别构造运动模糊和散焦模糊退化图像，再对传统复原方法与改进复原方法进行对比实验。

项目重点不是简单增强亮度，而是围绕监控图像中常见的模糊退化进行复原：运动模糊模拟车辆、行人或摄像头抖动造成的方向性拖影；散焦模糊模拟摄像机失焦或景深不足造成的整体边缘扩散。

![夜间监控复原对比示例](examples/results/comparison/190001_comparison.png)

## 项目特点

- 使用原始 LLVIP 夜间监控图像作为标准图，不再把增强后的图像当作原图。
- 支持两类退化场景：运动模糊和散焦模糊。
- 传统方法包括逆滤波、维纳滤波和约束最小二乘滤波。
- 改进方法采用反卷积去模糊、边缘加权融合、双边降噪和细节锐化的分阶段流程。
- 自动生成结果图片、对比图、best 结果图和指标表。
- 指标包括 PSNR、SSIM 和 RMSE，便于论文或答辩中进行客观分析。

## 目录结构

```text
.
├── run_surveillance_pipeline.py      # 推荐运行入口
├── run_surveillance_demo.py          # 只运行复原实验
├── prepare_surveillance_samples.py   # 只准备样本图片
├── requirements.txt                  # Python 依赖
├── src/
│   ├── path_config.py                # 路径配置，换电脑时主要看这里
│   ├── degradation.py                # 模糊退化模型
│   ├── metrics.py                    # PSNR / SSIM / RMSE
│   ├── run_surveillance_pipeline.py  # 一键流程
│   ├── run_surveillance_demo.py      # 实验主流程
│   └── methods/
│       ├── deconv_filters.py         # 传统反卷积滤波
│       ├── improved_method.py        # 改进复原方法
│       └── spatial_filters.py        # 空间域滤波辅助方法
└── examples/
    └── results/                      # 轻量示例结果，不包含完整数据集
```

## 环境安装

建议使用 Python 3.9 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 数据集放置方式

本仓库不包含 LLVIP 原始数据集。请自行下载 LLVIP 数据集，并把可见光图像放到下面的位置：

```text
项目根目录/
└── LLVIP/
    └── LLVIP/
        └── visible/
            ├── 190001.jpg
            ├── 190002.jpg
            └── ...
```

默认路径已经写在 `src/path_config.py` 中：

```python
LLVIP_VISIBLE_DIR = PROJECT_ROOT / "LLVIP" / "LLVIP" / "visible"
```

如果数据集放在其他位置，只需要修改这一行即可。

## 数据集来源、链接与致谢

本项目使用的夜间监控图像来源于 LLVIP 数据集。本仓库不是 LLVIP 官方仓库，也不重新分发完整 LLVIP 原始数据集。使用者需要按照 LLVIP 官方说明自行下载，并遵守其许可条款。

推荐优先查看官方链接：

- LLVIP 官方项目主页：<https://bupt-ai-cz.github.io/LLVIP/>
- LLVIP 官方 GitHub：<https://github.com/bupt-ai-cz/LLVIP>
- LLVIP 数据集下载说明：<https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md>
- LLVIP 论文 arXiv：<https://arxiv.org/abs/2108.10831>
- LLVIP 官方许可说明：<https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md>
- LLVIP Kaggle 相关页面：<https://www.kaggle.com/c/find-person-in-the-dark>
- Papers with Code 页面：<https://paperswithcode.com/sota/pedestrian-detection-on-llvip>

本项目的实验图片、退化图、复原图和对比图均用于课程设计、毕业设计和学术研究展示。若后续用于论文、报告、答辩或公开展示，请在文中明确说明图像数据来源于 LLVIP，并引用其原始论文。

## 数据集许可与使用说明

根据 LLVIP 官方许可说明，LLVIP Dataset 可用于学术研究、教学、科学出版和个人实验等非商业用途。使用时需要引用 LLVIP 数据集或其论文；不得将数据集或基于该数据集形成的派生数据用于商业销售、授权或其他商业获利目的。

本仓库遵循以下处理原则：

- 不上传完整 LLVIP 原始数据集。
- 不声称 LLVIP 数据集由本项目创建。
- 示例结果仅用于展示夜间监控图像模糊复原流程。
- 若权利方或数据集维护者要求删除相关示例图，本仓库将及时移除。
- 使用者若下载并使用 LLVIP，应自行确认其用途符合 LLVIP 官方许可。

## LLVIP 引用格式

如果你的论文、报告或代码中使用了 LLVIP 数据集，请优先引用 LLVIP 官方推荐文献：

```bibtex
@inproceedings{jia2021llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3496--3504},
  year={2021}
}
```

也可以引用 arXiv 版本：

```bibtex
@misc{llvip_arxiv_2021,
  title={LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Liu, Shengjie and Zhou, Wenli},
  year={2021},
  eprint={2108.10831},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2108.10831},
  url={https://arxiv.org/abs/2108.10831}
}
```

## 一键运行

推荐直接运行：

```powershell
python run_surveillance_pipeline.py
```

如果只是快速测试，可以限制处理数量：

```powershell
python run_surveillance_pipeline.py --limit 5
```

运行结束后，终端会打印主要输出路径：

```text
sample_path: surveillance_samples
result_path: surveillance_results
traditional_path: surveillance_results/traditional
improved_path: surveillance_results/improved
```

## 结果文件说明

运行后会生成 `surveillance_results` 文件夹，主要内容如下：

```text
surveillance_results/
├── traditional/                  # 传统方法结果
├── improved/                     # 改进方法结果
├── comparison/                   # 同一图片的退化图、复原图、原图对比
├── improved_best20/              # 改进最明显的结果图
├── surveillance_summary.csv      # 各方法平均指标
├── surveillance_case_summary.csv # 不同退化类型下的平均指标
└── surveillance_details.csv      # 每张图片的详细指标
```

示例结果中，5 张样本的平均指标如下：

| 方法 | PSNR | SSIM | RMSE |
|---|---:|---:|---:|
| 改进复原 | 29.0153 | 0.8443 | 0.0357 |
| 维纳滤波 | 24.2545 | 0.7345 | 0.0613 |
| 逆滤波 | 23.1702 | 0.7161 | 0.0695 |
| 约束最小二乘 | 14.9024 | 0.2531 | 0.1860 |

## 方法说明

传统方法主要用于建立基线：

- 逆滤波直接根据退化函数进行频域恢复，对噪声和核误差非常敏感。
- 维纳滤波在复原时考虑噪声抑制，稳定性通常优于逆滤波。
- 约束最小二乘滤波加入平滑约束，但参数不合适时容易造成过度平滑。

改进方法针对夜间监控图像的特点进行处理：

- 先使用反卷积削弱运动模糊或散焦模糊。
- 再通过边缘加权融合保留车辆边缘、道路纹理和行人轮廓。
- 然后使用双边滤波抑制噪声，同时尽量保留边界。
- 最后进行细节锐化，使复原图在视觉上更清晰。

## 适合汇报的结论

从示例结果可以看出，改进复原方法在 PSNR、SSIM 和 RMSE 三项指标上均优于传统方法。传统方法虽然能够进行一定程度的去模糊，但容易出现噪声放大、边缘振铃和细节丢失；改进方法通过分阶段处理，在抑制模糊扩散的同时保留了夜间监控图像中的关键结构信息，因此更适合车辆、道路标线、行人边界和交通信号区域的复原分析。

## 注意事项

- 本项目不上传完整 LLVIP 数据集，避免仓库体积过大。
- `examples/results` 只用于展示运行效果，不代表完整实验规模。
- 如果换电脑运行失败，优先检查 `src/path_config.py` 中的数据集路径。
- 如果图片很多，完整运行时间会明显增加，建议先用 `--limit 5` 测试。
