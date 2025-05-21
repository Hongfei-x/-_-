# -*- coding: utf-8 -*-
"""
Sentiment140 数据集 EDA 脚本
假设文件名：training.1600000.processed.noemoticon.csv
编码：latin-1
列名：label, tweet_id, date, query, user, text
label: 0 = 负面，4 = 正面
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from pylab import mpl
import re
import itertools
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与基本统计
# ----------------------------------
df = pd.read_csv(
    '/home/hfxia/数据挖掘/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=['label','tweet_id','date','query','user','text']
)

# 查看前几行
print(df.head())

# 检查缺失值
print("缺失值统计：\n", df.isnull().sum())

# 样本数量 & 情感分布
print("总样本数：", len(df))
print("情感分布：\n", df['label'].value_counts())

# 将 4 映射为 1，方便二分类
df['sentiment'] = df['label'].map({0: 0, 4: 1})

# 2. 文本长度分布
# ----------------------------------
# 计算字符数和词数
df['char_len'] = df['text'].str.len()
df['word_len'] = df['text'].str.split().str.len()

# 绘制直方图
plt.figure(figsize=(4,4))
plt.hist(df['char_len'], bins=50, alpha=0.7)
plt.title('推文字符长度分布')
plt.xlabel('字符数')
plt.ylabel('频次')
plt.tight_layout()
# plt.show()
plt.savefig('char_len_distribution.png')

plt.figure(figsize=(4,4))
plt.hist(df['word_len'], bins=50, alpha=0.7)
plt.title('推文词数分布')
plt.xlabel('词数')
plt.ylabel('频次')
plt.tight_layout()
# plt.show()
plt.savefig('word_len_distribution.png')

# 3. 词频与 n-gram 分析
# ----------------------------------
def clean_text(s):
    # 去除 URL、@用户、非字母数字
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5 ]+', '', s)
    return s.lower()

df['clean_text'] = df['text'].apply(clean_text)

# 计算词频
all_words = list(itertools.chain.from_iterable(
    df['clean_text'].str.split().tolist()
))
word_counts = Counter(all_words)
print("前 20 词频：", word_counts.most_common(20))

# 计算 bigram
bigrams = Counter()
for tokens in df['clean_text'].str.split():
    bigrams.update(zip(tokens, tokens[1:]))
print("前 20 Bigrams：", bigrams.most_common(20))

# 4. 情感类别对比词云
# ----------------------------------
pos_text = ' '.join(df[df.sentiment==1]['clean_text'])
neg_text = ' '.join(df[df.sentiment==0]['clean_text'])

wc = WordCloud(width=800, height=400, background_color='white', max_words=100)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(wc.generate(pos_text))
plt.axis('off')
plt.title('正面情感词云')

plt.subplot(1,2,2)
plt.imshow(wc.generate(neg_text))
plt.axis('off')
plt.title('负面情感词云')
plt.tight_layout()
plt.savefig('wordcloud.png')
# plt.show()

# 5. 长度 vs 情感 的关系
# ----------------------------------
plt.figure(figsize=(6,4))
# 使用箱型图比较各类别字符长度
df.boxplot(column='char_len', by='sentiment', grid=False)
plt.suptitle('')
plt.title('字符长度 vs 情感类别')
plt.xlabel('情感 (0=负面,1=正面)')
plt.ylabel('字符长度')
# plt.show()
plt.savefig('char_len_vs_sentiment.png')

# 6. 结果导出（可选）
# ----------------------------------
# 将分析结果（长度、词频等）保存为 CSV
word_freq_df = pd.DataFrame(word_counts.most_common(), columns=['word','count'])
word_freq_df.to_csv('word_frequency.csv', index=False, encoding='utf-8-sig')

df[['text','clean_text','char_len','word_len','sentiment']].head().to_csv(
    'sample_processed.csv', index=False, encoding='utf-8-sig'
)

print("EDA 分析完成，关键输出已保存。")

# print("=== 最终对比 ===")
# print("/home/hfxia/25acl/model/bert-base-uncased: acc=0.5146, precision=0.5147, recall=0.5139, f1=0.5142")
# print("/home/hfxia/25acl/model/bert-large-uncased: acc=0.5649, precision=0.5656, recall=0.5645, f1=0.5650")

# print("=== 最终对比 ===")
# print("/home/hfxia/25acl/model/bert-base-uncased: acc=0.4951, precision=0.4956, recall=0.4942, f1=0.4949")
# print("/home/hfxia/25acl/model/bert-large-uncased: acc=0.5489, precision=0.5492, recall=0.5485, f1=0.5488")