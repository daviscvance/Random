import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines

# Data
raw_data = """
GPT-5.5 (xhigh) | 922k | OpenAI | Proprietary | 60 | $11.25
GPT-5.5 (high) | 922k | OpenAI | Proprietary | 59 | $11.25
Claude Opus 4.7 (max) | 1M | Anthropic | Proprietary | 57 | $10.00
Gemini 3.1 Pro Preview | 1M | Google | Proprietary | 57 | $4.50
GPT-5.5 (medium) | 922k | OpenAI | Proprietary | 57 | $11.25
Kimi K2.6 | 256k | Kimi | Open | 54 | $1.71
MiMo-V2.5-Pro | 1M | Xiaomi | Open | 54 | $1.50
GPT-5.3 Codex (xhigh) | 400k | OpenAI | Proprietary | 54 | $4.81
Grok 4.3 | 1M | xAI | Proprietary | 53 | $1.56
Claude Opus 4.7 (Non-reasoning, high) | 1M | Anthropic | Proprietary | 52 | $10.00
Qwen3.6 Max Preview | 256k | Alibaba | Proprietary | 52 | $2.92
Claude Sonnet 4.6 (max) | 1M | Anthropic | Proprietary | 52 | $6.56
DeepSeek V4 Pro (Max) | 1M | DeepSeek | Open | 52 | $2.17
GLM-5.1 | 200k | Z AI | Open | 51 | $2.15
GPT-5.5 (low) | 922k | OpenAI | Proprietary | 51 | $11.25
Qwen3.6 Plus | 1M | Alibaba | Proprietary | 50 | $1.13
DeepSeek V4 Pro (High) | 1M | DeepSeek | Open | 50 | $2.17
GLM-5 | 200k | Z AI | Open | 50 | $1.55
MiniMax-M2.7 | 205k | MiniMax | Open | 50 | $0.52
MiMo-V2-Pro | 1M | Xiaomi | Proprietary | 49 | $1.50
GPT-5.4 mini (xhigh) | 400k | OpenAI | Proprietary | 49 | $1.69
GPT-5.4 (low) | 1.05M | OpenAI | Proprietary | 48 | $5.63
DeepSeek V4 Flash (Max) | 1M | DeepSeek | Open | 47 | $0.18
Gemini 3 Flash | 1M | Google | Proprietary | 46 | $1.13
Qwen3.6 27B | 262k | Alibaba | Open | 46 | $1.35
Qwen3.5 397B A17B | 262k | Alibaba | Open | 45 | $1.35
MiMo-V2-Omni-0327 | 256k | Xiaomi | Proprietary | 45 | $0.00
Claude Sonnet 4.6 (Non-reasoning) | 1M | Anthropic | Proprietary | 44 | $6.56
GPT-5.4 nano (xhigh) | 400k | OpenAI | Proprietary | 44 | $0.46
GLM-5.1 | 200k | Z AI | Open | 44 | $2.15
Qwen3.6 35B A3B | 262k | Alibaba | Open | 43 | $0.56
MiMo-V2-Omni | 256k | Xiaomi | Proprietary | 43 | $0.00
Kimi K2.6 | 256k | Kimi | Open | 43 | $1.71
Claude Sonnet 4.6 (Non-reasoning, Low Effort) | 1M | Anthropic | Proprietary | 43 | $6.56
Hy3-preview | 256k | Tencent | Open | 42 | $0.00
Qwen3.5 122B A10B | 262k | Alibaba | Open | 42 | $1.10
MiMo-V2-Flash (Feb 2026) | 256k | Xiaomi | Open | 41 | $0.15
GPT-5.5 (Non-reasoning) | 922k | OpenAI | Proprietary | 41 | $11.25
GLM-5 | 200k | Z AI | Open | 41 | $1.55
Qwen3.5 397B A17B | 262k | Alibaba | Open | 40 | $1.35
DeepSeek V4 Pro | 1M | DeepSeek | Open | 39 | $2.17
Mistral Medium 3.5 | 256k | Mistral | Open | 39 | $3.00
Gemma 4 31B | 256k | Google | Open | 39 | $0.00
Qwen3.5 Omni Plus | 256k | Alibaba | Proprietary | 39 | $1.50
Grok 4.1 Fast | 2M | xAI | Proprietary | 39 | $0.28
Step 3.5 Flash 2603 | 256k | StepFun | Proprietary | 38 | $0.00
o3 | 200k | OpenAI | Proprietary | 38 | $3.50
GPT-5.4 nano | 400k | OpenAI | Proprietary | 38 | $0.46
GPT-5.4 mini (medium) | 400k | OpenAI | Proprietary | 38 | $1.69
Kimi K2.5 | 256k | Kimi | Open | 37 | $1.20
Qwen3.6 27B | 262k | Alibaba | Open | 37 | $1.35
Claude 4.5 Haiku | 200k | Anthropic | Proprietary | 37 | $2.19
DeepSeek V4 Flash | 1M | DeepSeek | Open | 36 | $0.18
NVIDIA Nemotron 3 Super | 1M | NVIDIA | Open | 36 | $0.41
Qwen3.5 122B A10B | 262k | Alibaba | Open | 36 | $1.10
Nova 2.0 Pro Preview (medium) | 256k | Amazon | Proprietary | 36 | $3.44
MiMo-V2.5-Pro | 1M | Xiaomi | Open | 36 | $1.50
GPT-5.4 (Non-reasoning) | 1.05M | OpenAI | Proprietary | 35 | $5.63
Gemini 3 Flash | 1M | Google | Proprietary | 35 | $1.13
Gemini 2.5 Pro | 1M | Google | Proprietary | 35 | $3.44
Nova 2.0 Lite (high) | 1M | Amazon | Proprietary | 35 | $0.85
Hy3-preview | 256k | Tencent | Open | 34 | $0.00
Ling-2.6-1T | 262k | InclusionAI | Open | 34 | $0.85
Gemini 3.1 Flash-Lite Preview | 1M | Google | Proprietary | 34 | $0.56
gpt-oss-120B (high) | 131k | OpenAI | Open | 33 | $0.26
Mercury 2 | 128k | Inception | Proprietary | 33 | $0.38
Qwen3.5 9B | 262k | Alibaba | Open | 32 | $0.11
Grok 3 mini Reasoning (high) | 1M | xAI | Proprietary | 32 | $0.35
Nova 2.0 Pro Preview (low) | 256k | Amazon | Proprietary | 32 | $3.44
Trinity Large Thinking | 512k | Arcee AI | Open | 32 | $0.40
Qwen3.6 35B A3B | 262k | Alibaba | Open | 32 | $0.84
Claude 4.5 Haiku | 200k | Anthropic | Proprietary | 31 | $2.19
Qwen3.5 35B A3B | 262k | Alibaba | Open | 31 | $0.69
MiMo-V2-Flash | 256k | Xiaomi | Open | 30 | $0.15
Nova 2.0 Lite (medium) | 1M | Amazon | Proprietary | 30 | $0.85
Grok 4.20 0309 v2 | 2M | xAI | Proprietary | 29 | $3.00
Grok Code Fast 1 | 256k | xAI | Proprietary | 29 | $0.53
Qwen3 Coder Next | 256k | Alibaba | Open | 28 | $0.56
Mistral Small 4 | 256k | Mistral | Open | 28 | $0.26
Magistral Medium 1.2 | 128k | Mistral | Proprietary | 27 | $2.75
Qwen3.5 4B | 262k | Alibaba | Open | 27 | $0.06
Qwen3 Next 80B A3B | 262k | Alibaba | Open | 27 | $1.88
Ling 2.6 Flash | 262k | InclusionAI | Open | 26 | $0.15
Qwen3.5 Omni Flash | 256k | Alibaba | Proprietary | 26 | $0.28
Nova 2.0 Lite (low) | 1M | Amazon | Proprietary | 25 | $0.85
gpt-oss-120B (low) | 131k | OpenAI | Open | 24 | $0.26
gpt-oss-20B (high) | 131k | OpenAI | Open | 24 | $0.09
GPT-5.4 nano | 400k | OpenAI | Proprietary | 24 | $0.46
NVIDIA Nemotron 3 Nano | 1M | NVIDIA | Open | 24 | $0.10
LongCat Flash Lite | 256k | LongCat | Open | 24 | $0.00
Grok 4.1 Fast | 2M | xAI | Proprietary | 24 | $0.28
GPT-5.4 mini | 400k | OpenAI | Proprietary | 23 | $1.69
Nova 2.0 Pro Preview | 256k | Amazon | Proprietary | 23 | $3.44
Mistral Large 3 | 256k | Mistral | Open | 23 | $0.75
Qwen3.5 4B | 262k | Alibaba | Open | 23 | $0.06
Gemini 2.5 Flash-Lite (Sep) | 1M | Google | Proprietary | 22 | $0.18
Mistral Medium 3.1 | 128k | Mistral | Proprietary | 21 | $0.80
gpt-oss-20B (low) | 131k | OpenAI | Open | 21 | $0.10
Qwen3 Next 80B A3B | 262k | Alibaba | Open | 20 | $0.88
Nova Premier | 1M | Amazon | Proprietary | 19 | $5.00
Llama Nemotron Super 49B v1.5 | 128k | NVIDIA | Open | 19 | $0.18
Mistral Small 4 | 256k | Mistral | Open | 19 | $0.26
Llama 4 Maverick | 1M | Meta | Open | 18 | $0.47
Sarvam 105B (high) | 128k | Sarvam | Open | 18 | $0.00
Magistral Small 1.2 | 128k | Mistral | Open | 18 | $0.75
Nova 2.0 Lite | 1M | Amazon | Proprietary | 18 | $0.85
Llama 3.1 405B | 128k | Meta | Open | 17 | $3.69
Nova 2.0 Omni | 1M | Amazon | Proprietary | 17 | $0.85
Ministral 3 14B | 256k | Mistral | Open | 16 | $0.20
DeepSeek R1 Distill Llama 70B | 128k | DeepSeek | Open | 16 | $0.79
Ling-flash-2.0 | 128k | InclusionAI | Open | 16 | $0.25
Qwen3 Omni 30B A3B | 65.5k | Alibaba | Open | 16 | $0.43
Llama Nemotron Ultra | 128k | NVIDIA | Open | 15 | $0.90
ERNIE 4.5 300B A47B | 131k | Baidu | Open | 15 | $0.49
NVIDIA Nemotron Nano 12B v2 VL | 128k | NVIDIA | Open | 15 | $0.30
Ministral 3 8B | 256k | Mistral | Open | 15 | $0.15
NVIDIA Nemotron Nano 9B V2 | 131k | NVIDIA | Open | 15 | $0.07
Qwen3.5 2B | 262k | Alibaba | Open | 15 | $0.04
Llama Nemotron Super 49B v1.5 | 128k | NVIDIA | Open | 15 | $0.18
Llama 3.3 70B | 128k | Meta | Open | 14 | $0.62
Ring-flash-2.0 | 128k | InclusionAI | Open | 14 | $0.25
Llama 4 Scout | 10M | Meta | Open | 14 | $0.29
Command A | 256k | Cohere | Open | 13 | $4.38
Llama 3.1 Nemotron 70B | 128k | NVIDIA | Open | 13 | $1.20
NVIDIA Nemotron 3 Nano | 1M | NVIDIA | Open | 13 | $0.09
NVIDIA Nemotron Nano 9B V2 | 131k | NVIDIA | Open | 13 | $0.09
Granite 4.1 8B | 131k | IBM | Open | 12 | $0.06
Sarvam 30B (high) | 65.5k | Sarvam | Open | 12 | $0.00
Llama 3.2 90B (Vision) | 128k | Meta | Open | 12 | $1.38
Ministral 3 3B | 256k | Mistral | Open | 11 | $0.10
Jamba 1.7 Large | 256k | AI21 Labs | Open | 11 | $3.50
Granite 4.0 H Small | 128k | IBM | Open | 11 | $0.11
Qwen3 Omni 30B A3B | 65.5k | Alibaba | Open | 11 | $0.43
LFM2 24B A2B | 32.8k | Liquid AI | Open | 10 | $0.05
Phi-4 | 16k | Microsoft | Open | 10 | $0.22
Nova Micro | 130k | Amazon | Proprietary | 10 | $0.06
NVIDIA Nemotron Nano 12B v2 VL | 128k | NVIDIA | Open | 10 | $0.30
Phi-4 Multimodal | 128k | Microsoft | Open | 10 | $0.00
Qwen3.5 0.8B | 262k | Alibaba | Open | 10 | $0.02
Llama 3.2 11B (Vision) | 128k | Meta | Open | 9 | $0.24
Phi-4 Mini | 128k | Microsoft | Open | 8 | $0.00
LFM2.5-1.2B-Instruct | 32k | Liquid AI | Open | 8 | $0.00
LFM2 2.6B | 32.8k | Liquid AI | Open | 8 | $0.00
Apertus 70B Instruct | 65.5k | Swiss AI Initiative | Open | 8 | $1.34
LFM2 8B A1B | 32.8k | Liquid AI | Open | 7 | $0.00
LFM2.5-VL-1.6B | 32k | Liquid AI | Open | 6 | $0.00
Apertus 8B Instruct | 65.5k | Swiss AI Initiative | Open | 6 | $0.13
Tiny Aya Global | 8.19k | Cohere | Open | 5 | $0.00
"""

data = []
for line in raw_data.strip().split("\n"):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 6:
        name = parts[0]
        if name == "MiMo-V2-Omni-0327": continue
        license_type = parts[3]
        intel_score = float(parts[4].replace("*", ""))
        price_str = parts[5].replace("$", "")
        if price_str != "--":
            price = float(price_str)
            if price > 0:
                data.append({"Model": name, "Intelligence": intel_score, "Price": price, "License": license_type})

df = pd.DataFrame(data)

# Deciles
df['Decile_Rank'] = df['Intelligence'].rank(method='first', ascending=True)
df['Decile_Num'] = pd.qcut(df['Decile_Rank'], 10, labels=False) + 1
decile_ranges = {i: f"D{i} ({df[df['Decile_Num']==i]['Intelligence'].min():.1f}-{df[df['Decile_Num']==i]['Intelligence'].max():.1f})" for i in range(1, 11)}
df['Decile'] = df['Decile_Num'].map(decile_ranges)

# Pareto
sorted_df = df.sort_values(by=['Price', 'Intelligence'], ascending=[True, False])
pareto_frontier = []
max_intel = -1
for _, row in sorted_df.iterrows():
    if row['Intelligence'] > max_intel:
        pareto_frontier.append(row)
        max_intel = row['Intelligence']
pareto_df = pd.DataFrame(pareto_frontier)

plt.figure(figsize=(16, 12))
sns.set_style("whitegrid")
palette = sns.color_palette("RdYlGn", 10)
hue_order = [decile_ranges[i] for i in range(10, 0, -1)]

sns.scatterplot(data=df[df['License']=='Open'], x="Price", y="Intelligence", hue="Decile",
                palette=palette, hue_order=hue_order, marker='o', s=120, alpha=0.4, edgecolor='black', legend=False)
sns.scatterplot(data=df[df['License']=='Proprietary'], x="Price", y="Intelligence", hue="Decile",
                palette=palette, hue_order=hue_order, marker='^', s=150, alpha=0.5, edgecolor='black')

plt.plot(pareto_df['Price'], pareto_df['Intelligence'], color='blue', linestyle='--', linewidth=2, alpha=0.6, label='Pareto Frontier')

plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.xticks([0.02, 0.05, 0.1, 0.5, 1, 5, 10, 15], ["$0.02", "$0.05", "$0.10", "$0.50", "$1.00", "$5.00", "$10.00", "$15.00"])

# Frontier Labels
for i, (idx, row) in enumerate(pareto_df.sort_values("Intelligence").iterrows()):
    x_factor = 0.55 if i % 2 == 0 else 0.4
    y_add = 4 if i % 2 == 0 else 8
    plt.annotate(row["Model"], xy=(row["Price"], row["Intelligence"]), xytext=(row["Price"] * x_factor, row["Intelligence"] + y_add),
        textcoords='data', fontsize=11, weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1.5, alpha=0.95),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color='blue', lw=1.5, alpha=0.8), ha='right')

# SHARED GREEN FORMATTING
green_fmt = dict(fontsize=10, weight='bold', color='darkgreen',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#eaffea", ec="green", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color='green', alpha=0.7), ha='right')

# Label Kimi K2.6
kimi = df[df['Model'] == 'Kimi K2.6'].iloc[0]
plt.annotate("Kimi K2.6 (Open)", xy=(kimi["Price"], kimi["Intelligence"]), xytext=(kimi["Price"] * 0.4, kimi["Intelligence"] + 5),
             textcoords='data', **green_fmt)

# Label Grok 4.3
grok = df[df['Model'] == 'Grok 4.3'].iloc[0]
plt.annotate(grok["Model"], xy=(grok["Price"], grok["Intelligence"]), xytext=(grok["Price"] * 0.5, grok["Intelligence"] - 6),
             textcoords='data', **green_fmt)

# Label GPT-5.5 (high)
gpt_high = df[df['Model'] == 'GPT-5.5 (high)'].iloc[0]
plt.annotate(gpt_high["Model"], xy=(gpt_high["Price"], gpt_high["Intelligence"]), xytext=(gpt_high["Price"] * 0.5, gpt_high["Intelligence"] - 8),
             textcoords='data', **green_fmt)

# Legend
handles, labels = plt.gca().get_legend_handles_labels()
open_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Open License (Circle)', alpha=0.5)
prop_marker = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Proprietary License (Triangle)', alpha=0.5)
new_handles = handles + [open_marker, prop_marker]
new_labels = labels + ['Open License (Circle)', 'Proprietary License (Triangle)']

plt.title("Pareto Frontier with Labels", fontsize=26, pad=20)
plt.xlabel("Blended Price (USD/1M Tokens) - Log Scale", fontsize=14)
plt.ylabel("Intelligence Index", fontsize=14)
plt.legend(handles=new_handles, labels=new_labels, title="Legend", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xlim(0.01, 25); plt.ylim(5, 65); plt.tight_layout()
plt.savefig("pareto_uniform_green.png")
