# L2W1 Project Directory Structure

## Root

- **configs/**: 实验配置文件 (YAML)
  - `agent_a_config.yaml`: PaddleOCR 参数
  - `agent_b_config.yaml`: MiniCPM-V/Qwen 参数
  - `router_config.yaml`: 阈值与 MLP 参数
- **docs/**: 开发文档与 Specs
  - `L2W1-DE-001 (Data Engine - Alignment Core).md`
- **l2w1/**: 核心源码包 (Source Code)
  - **core/**: 核心组件
    - `agent_a.py`: PaddleOCR-VL 封装类 (The Scout)
    - `agent_b.py`: VLM 推理封装类 (The Judge)
    - `router.py`: 视觉熵与 PPL 计算 (The Gatekeeper)
  - **data/**: 数据工程流水线
    - `preprocess.py`: 图像切片与标准化
    - **alignment/**: [Sprint 1 重点]
      - `dtw_core.py`: DTW 强制对齐与 Gap 插值算法
      - `matcher.py`: 字符级匹配逻辑
    - `dataset.py`: PyTorch Dataset 封装
  - **utils/**: 通用工具
    - `metrics.py`: CER, 幻觉率计算
    - `visualizer.py`: 纠错结果可视化
- **scripts/**: 执行脚本
  - `01_build_dataset.py`: 运行数据清洗流水线
  - `02_train_router.py`: 训练路由模块
  - `03_inference_demo.py`: 单页推理演示
- **tests/**: 单元测试
- **experiments/**: 存放实验日志和 Checkpoints
- `requirements.txt`: 依赖清单
- `.cursorrules`: Cursor AI 行为规范
