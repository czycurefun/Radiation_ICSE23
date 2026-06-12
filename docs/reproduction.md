# RADIATION 复现说明

这个仓库复现的是 Cross-Domain Requirement Linking via Adversarial Domain Adaptation。任务是判断两个需求/测试用例文本之间是否存在 trace/link 关系；方法先在源域训练需求链接分类器，再通过距离约束和对抗域适配把目标域 encoder 对齐到源域表示空间，最后在目标域上评估链接预测效果。

复现脚本为：

```text
reproduce_radiation.py
```

它跑通的流程包括：

1. 读取 `data/processed/{domain}/newtrain.txt` 和 `newtest.txt` 中的需求链接样本。
2. 使用 BERT tokenizer 编码两个文本片段。
3. 训练 source encoder + link classifier，并加入 MMD distance loss。
4. 复制 source encoder 初始化 target encoder。
5. 训练 discriminator 区分 source/target 表示，同时训练 target encoder 欺骗 discriminator，并继续加入 MMD distance loss。
6. 输出 source-only target 评估、adapted target 评估、预测 TSV 和模型权重。

## 1. 环境安装

在仓库根目录执行：

```bash
cd /home/public/minzhi/Requirement-Linking-Adversial-Adaptation
/home/public/.local/bin/python3.10 -m venv .venv
.venv/bin/pip install -U pip setuptools wheel
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
.venv/bin/pip install transformers==4.46.3 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 tqdm==4.67.1
```

也可以参考 `requirements_reproduction.txt`；其中 `torch==2.5.1+cu121` 需要配合 PyTorch CUDA wheel 源安装。

## 2. BERT 模型下载

默认使用：

```text
prajjwal1/bert-tiny
```

这是一个轻量 BERT checkpoint，用于快速真实跑通 RADIATION 流程。首次运行 `reproduce_radiation.py` 时会自动下载到仓库内 `.hf_cache/`。也可以提前下载：

```bash
cd /home/public/minzhi/Requirement-Linking-Adversial-Adaptation
HF_HOME=/home/public/minzhi/Requirement-Linking-Adversial-Adaptation/.hf_cache \
HF_HUB_DISABLE_XET=1 \
.venv/bin/python - <<'PY'
from transformers import AutoModel, AutoTokenizer
model_name = "prajjwal1/bert-tiny"
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
PY
```

如果要更接近原仓库默认设置，可以换成：

```bash
--model-name bert-base-cased
```

对应训练会更慢、显存占用更高。

## 3. 一键复现命令

下面命令使用 GPU 0，跑 `easy -> EBT` 跨域迁移。为了快速验证链路，默认 source/target train 各取 64 条样本，训练 1 个 source epoch 和 1 个 adaptation epoch。

```bash
cd /home/public/minzhi/Requirement-Linking-Adversial-Adaptation
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/public/minzhi/Requirement-Linking-Adversial-Adaptation/.hf_cache \
HF_HUB_DISABLE_XET=1 \
.venv/bin/python reproduce_radiation.py \
  --output-dir outputs/reproduction_easy_to_ebt \
  --source easy \
  --target EBT \
  --model-name prajjwal1/bert-tiny \
  --source-epochs 1 \
  --adapt-epochs 1 \
  --source-train-limit 64 \
  --target-train-limit 64 \
  --batch-size 8 \
  --max-length 128 \
  --seed 2026
```

如需扩大训练规模，可以调大：

```bash
--source-train-limit 0
--target-train-limit 0
--source-epochs 5
--adapt-epochs 5
--max-length 256
```

其中 limit 设为 `0` 表示使用对应 split 的全部样本。

## 4. 输出文件

本次复现输出目录：

```text
/home/public/minzhi/Requirement-Linking-Adversial-Adaptation/outputs/reproduction_easy_to_ebt
```

主要文件：

- `result.json`：运行配置、数据规模和三组评估指标。
- `source_only_target_predictions.tsv`：source encoder 直接在 target test 上的预测。
- `adapted_target_predictions.tsv`：target encoder 适配后在 target test 上的预测。
- `models/source_encoder.pt`：源域 encoder 权重。
- `models/classifier.pt`：链接分类器权重。
- `models/target_encoder.pt`：适配后的目标域 encoder 权重。
- `models/discriminator.pt`：域判别器权重。

## 5. 本次真实运行结果

已在当前机器真实跑通一次：

- GPU：`cuda:0`
- Source domain：`easy`
- Target domain：`EBT`
- Encoder：`prajjwal1/bert-tiny`
- Source train：64 条
- Target train：64 条
- Source test：12 条
- Target test：10 条

结果摘要：

```text
source_domain_eval accuracy: 0.5833
source_only_target_eval accuracy: 0.6000
adapted_target_eval accuracy: 0.6000
```

这是小样本 smoke reproduction，目标是证明代码、数据、模型下载、训练、对抗适配和评估链路真实可运行；不用于对齐论文原始指标。
