from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
DESKTOP_DIR = Path.home() / "Desktop" / "nighttime_flowchart_options"


COLORS = {
    "bg": "#F7F4EC",
    "ink": "#1F2933",
    "muted": "#61707D",
    "line": "#B9C0C8",
    "blue": "#2F6F9F",
    "blue2": "#DDEBF4",
    "green": "#2F7D5A",
    "green2": "#DCEFE5",
    "orange": "#B86B2A",
    "orange2": "#F3E3D1",
    "red": "#A3483D",
    "red2": "#F2DAD6",
    "gray2": "#EEF0F2",
    "purple": "#6D5A8D",
    "purple2": "#E7E1F0",
    "yellow": "#FFF3D8",
}


def configure_fonts() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese font found. Install Noto Sans SC or Microsoft YaHei.")

    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["font.sans-serif"] = [font_name, "Microsoft YaHei", "SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["svg.fonttype"] = "path"
    mpl.rcParams["pdf.fonttype"] = 42
    return (
        font_manager.FontProperties(fname=str(font_path)),
        font_manager.FontProperties(fname=str(font_path), weight="bold"),
    )


FONT_REGULAR, FONT_BOLD = configure_fonts()


def put_text(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 10,
    color: str | None = None,
    ha: str = "left",
    va: str = "center",
    bold: bool = False,
    **kwargs,
) -> None:
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        fontproperties=FONT_BOLD if bold else FONT_REGULAR,
        **kwargs,
    )


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    fill: str,
    edge: str,
    index: int,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(box)

    circle = Circle(
        (x + 0.052, y + height - 0.058),
        0.03,
        facecolor=edge,
        edgecolor="white",
        lw=1.2,
        zorder=5,
    )
    ax.add_patch(circle)
    put_text(ax, x + 0.052, y + height - 0.059, str(index), size=8.5, color="white", ha="center", bold=True, zorder=6)
    put_text(ax, x + 0.096, y + height * 0.64, title, size=11.2, bold=True, zorder=5)
    put_text(ax, x + 0.096, y + height * 0.32, subtitle, size=8.4, color=COLORS["muted"], zorder=5, linespacing=1.35)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str, *, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )


def setup_canvas(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=220)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    put_text(ax, 0.05, 0.94, title, size=18, bold=True)
    ax.plot([0.05, 0.95], [0.895, 0.895], color=COLORS["line"], lw=1.15)
    put_text(ax, 0.05, 0.865, subtitle, size=9.5, color=COLORS["muted"])
    return fig, ax


def save_figure(fig, stem: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (ASSET_DIR, DESKTOP_DIR):
        fig.savefig(folder / f"{stem}.png", dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(folder / f"{stem}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_reproduce_workflow() -> None:
    fig, ax = setup_canvas(
        "复刻项目流程图",
        "从拿到项目到生成实验结果的最短路径，适合放在 README 或汇报材料中。",
    )

    boxes = [
        ((0.06, 0.67), 0.22, 0.13, "获取项目", "下载仓库或解压项目文件", COLORS["blue2"], COLORS["blue"]),
        ((0.39, 0.67), 0.22, 0.13, "安装环境", "Python 3.9+ 与 requirements", COLORS["green2"], COLORS["green"]),
        ((0.72, 0.67), 0.22, 0.13, "下载 LLVIP", "只需 visible 可见光图像", COLORS["orange2"], COLORS["orange"]),
        ((0.72, 0.43), 0.22, 0.13, "放置数据集", "LLVIP/LLVIP/visible", COLORS["orange2"], COLORS["orange"]),
        ((0.39, 0.43), 0.22, 0.13, "检查路径", "必要时修改 path_config.py", COLORS["purple2"], COLORS["purple"]),
        ((0.06, 0.43), 0.22, 0.13, "一键运行", "python\nrun_surveillance_pipeline.py", COLORS["green2"], COLORS["green"]),
        ((0.06, 0.19), 0.22, 0.13, "查看结果", "traditional / improved\ncomparison", COLORS["blue2"], COLORS["blue"]),
        ((0.39, 0.19), 0.22, 0.13, "读取指标", "PSNR / SSIM / RMSE", COLORS["red2"], COLORS["red"]),
        ((0.72, 0.19), 0.22, 0.13, "用于论文答辩", "对比图 + 指标表 + 结论", COLORS["gray2"], COLORS["ink"]),
    ]
    for index, box in enumerate(boxes, 1):
        add_box(ax, *box, index)

    arrows = [
        ((0.28, 0.735), (0.39, 0.735), COLORS["blue"]),
        ((0.61, 0.735), (0.72, 0.735), COLORS["green"]),
        ((0.83, 0.67), (0.83, 0.56), COLORS["orange"]),
        ((0.72, 0.495), (0.61, 0.495), COLORS["orange"]),
        ((0.39, 0.495), (0.28, 0.495), COLORS["purple"]),
        ((0.17, 0.43), (0.17, 0.32), COLORS["green"]),
        ((0.28, 0.255), (0.39, 0.255), COLORS["blue"]),
        ((0.61, 0.255), (0.72, 0.255), COLORS["red"]),
    ]
    for start, end, color in arrows:
        add_arrow(ax, start, end, color)

    callout = FancyBboxPatch(
        (0.38, 0.05),
        0.56,
        0.075,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        lw=1,
        edgecolor="#D0A14A",
        facecolor=COLORS["yellow"],
        zorder=3,
    )
    ax.add_patch(callout)
    put_text(ax, 0.405, 0.087, "提示：如果别人运行失败，优先检查数据集路径是否与 src/path_config.py 一致。", size=9.2, color="#6F4E11")
    save_figure(fig, "workflow_reproduce_cn")


def draw_experiment_workflow() -> None:
    fig, ax = setup_canvas(
        "夜间监控图像复原实验流程图",
        "展示本项目从原始监控图、退化建模、传统/改进复原到指标评价的完整实验链路。",
    )

    add_box(ax, (0.055, 0.58), 0.21, 0.16, "LLVIP 原始图", "夜间可见光监控图\n作为标准参考图", COLORS["blue2"], COLORS["blue"], 1)
    add_box(ax, (0.345, 0.69), 0.22, 0.13, "运动模糊退化", "方向性拖影\n模拟目标运动或抖动", COLORS["orange2"], COLORS["orange"], 2)
    add_box(ax, (0.345, 0.48), 0.22, 0.13, "散焦模糊退化", "边缘均匀扩散\n模拟镜头失焦", COLORS["orange2"], COLORS["orange"], 3)
    add_box(ax, (0.64, 0.69), 0.22, 0.13, "传统复原", "逆滤波 / 维纳滤波\n约束最小二乘滤波", COLORS["gray2"], COLORS["ink"], 4)
    add_box(ax, (0.64, 0.48), 0.22, 0.13, "改进复原", "反卷积 + 边缘融合\n降噪 + 细节锐化", COLORS["green2"], COLORS["green"], 5)
    add_box(ax, (0.64, 0.22), 0.22, 0.13, "质量评价", "PSNR / SSIM / RMSE\n与原图进行对比", COLORS["red2"], COLORS["red"], 6)
    add_box(ax, (0.345, 0.22), 0.22, 0.13, "结果输出", "对比图 / best20\nCSV 指标表", COLORS["purple2"], COLORS["purple"], 7)
    add_box(ax, (0.055, 0.22), 0.21, 0.13, "实验结论", "改进方法整体优于\n传统复原方法", COLORS["blue2"], COLORS["blue"], 8)

    add_arrow(ax, (0.265, 0.66), (0.345, 0.755), COLORS["blue"], rad=0.12)
    add_arrow(ax, (0.265, 0.66), (0.345, 0.545), COLORS["blue"], rad=-0.12)
    add_arrow(ax, (0.565, 0.755), (0.64, 0.755), COLORS["orange"])
    add_arrow(ax, (0.565, 0.545), (0.64, 0.545), COLORS["orange"])
    add_arrow(ax, (0.75, 0.69), (0.75, 0.61), COLORS["ink"])
    add_arrow(ax, (0.75, 0.48), (0.75, 0.35), COLORS["green"])
    add_arrow(ax, (0.64, 0.285), (0.565, 0.285), COLORS["red"])
    add_arrow(ax, (0.345, 0.285), (0.265, 0.285), COLORS["purple"])

    strip = FancyBboxPatch(
        (0.055, 0.055),
        0.805,
        0.085,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        lw=1.0,
        edgecolor="#CBD2D9",
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(strip)
    put_text(ax, 0.085, 0.097, "核心判断：退化图应能看出模糊；复原图应在边缘、纹理和目标轮廓上更接近原图。", size=9.3)
    save_figure(fig, "workflow_experiment_cn")


def main() -> None:
    draw_reproduce_workflow()
    draw_experiment_workflow()
    print(f"Saved flowcharts to: {ASSET_DIR}")
    print(f"Saved copies to: {DESKTOP_DIR}")


if __name__ == "__main__":
    main()
