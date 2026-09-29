"""AICG —— AI 实时构图指导 Agent（AI Composition Guide）。

一个"能看画面 → 判断构图 → 说人话给建议 → 拍后精修"的实时多模态 Agent 系统。

模块分层（依赖只允许自上而下，见《目录结构.md》§4）::

    api / pipeline        编排层
        ↓
    language / retrieval  语言与检索层
        ↓
    stabilization         防抖层
        ↓
    composition           构图决策层
        ↓
    perception            感知层
        ↓
    camera                帧源层
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
