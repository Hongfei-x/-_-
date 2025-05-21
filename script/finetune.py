# -*- coding: utf-8 -*-
"""
Sentiment140 三分类微调脚本
假设文件：training.1600000.processed.noemoticon.csv
列名顺序：label, tweet_id, date, query, user, text
原标签：0,2,4
"""
import re
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 读取并清洗
def clean_text(s: str) -> str:
    s = re.sub(r'http\S+', '', s)           # 去URL
    s = re.sub(r'@\w+', '', s)              # 去@mention
    s = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5 ]+', '', s)  # 非中英文数字移除
    return s.lower().strip()

# 读取
df = pd.read_csv(
    '/home/hfxia/数据挖掘/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=['label','tweet_id','date','query','user','text']
)

# 仅保留 0,2,4 标签，映射为连续 0/1/2
df = df[df['label'].isin([0,2,4])].copy()
label_map = {0:0, 2:1, 4:2}
df['sentiment'] = df['label'].map(label_map)

# 清洗文本
df['clean_text'] = df['text'].apply(clean_text)

# 2. 构建 Hugging Face Dataset 并划分
dataset = Dataset.from_pandas(df[['clean_text','sentiment']])
dataset = dataset.rename_column('clean_text', 'text')
dataset = dataset.class_encode_column('sentiment')

# 80/10/10 划分
split1 = dataset.train_test_split(test_size=0.2, seed=42)
split2 = split1['test'].train_test_split(test_size=0.5, seed=42)
full_ds = DatasetDict({
    'train': split1['train'],
    'validation': split2['train'],
    'test': split2['test']
})

# 3. Tokenization 准备
MODEL_NAMES = ['/home/hfxia/25acl/model/bert-base-uncased', '/home/hfxia/25acl/model/bert-large-uncased']
MAX_LEN = 128

def preprocess(batch, tokenizer):
    toks = tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN
    )
    toks['labels'] = batch['sentiment']
    return toks

# 4. 定义评价指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

# 5. 微调并评估
results = {}
for model_name in MODEL_NAMES:
    print(f"\n>>> 微调模型：{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize 全量数据集
    tokenized = full_ds.map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True
    )
    tokenized.set_format(
        type='torch',
        columns=['input_ids','attention_mask','labels']
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    args = TrainingArguments(
        output_dir=f'./outputs/{model_name}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1000
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_res = trainer.evaluate(tokenized['test'])
    print(f"测试集结果：{eval_res}\n")
    results[model_name] = eval_res

# 6. 对比结果输出
print("=== 最终对比 ===")
for name, res in results.items():
    print(f"{name}: acc={res['eval_accuracy']:.4f}, "
          f"precision={res['eval_precision']:.4f}, "
          f"recall={res['eval_recall']:.4f}, "
          f"f1={res['eval_f1']:.4f}")
