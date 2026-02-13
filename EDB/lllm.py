import pandas as pd
import time
from tqdm import tqdm
import os
from openai import AzureOpenAI

# ==================== 配置区 ====================
ORIGINAL_EXCEL = '/root/lsz/CGM/code/食物词_BERTopic.xlsx'
EXISTING_CSV = '/root/lsz/CGM/code/食物类别分类结果.csv'
API_ENDPOINT = 'https://ai-eval-found.openai.azure.com/'
API_KEY = 'YOUR_API_KEY_HERE'  # ❗替换为真实API Key
DEPLOYMENT_NAME = 'gpt-4.1'
API_VERSION = '2024-08-01-preview'
GROUP_SIZE = 50
REQUIRED_COLUMNS = ['index', '食物词', '频次', '类别']  # 严格四列顺序
# ===============================================

# ========== 1. 初始化客户端 ==========
client = AzureOpenAI(
    azure_endpoint=API_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

# ========== 2. 重建原始数据索引（严格四列结构）==========
print("🔍 步骤1/4: 重建原始数据索引...")
original_df = pd.read_excel(ORIGINAL_EXCEL)
if not {'食物词', '全局频次'}.issubset(original_df.columns):
    raise ValueError(f"原始Excel缺失必要列！可用列: {list(original_df.columns)}")

original_df = original_df[['食物词', '全局频次']].copy()
original_df['index'] = range(len(original_df))
original_df = original_df[['index', '食物词', '全局频次']]
original_df.columns = ['index', '食物词', '频次']
print(f"✅ 原始数据重建完成 | 总词数: {len(original_df)}")

# ========== 3. 验证现有CSV结构 ==========
if not os.path.exists(EXISTING_CSV):
    raise FileNotFoundError(f"❌ 未找到分类结果文件: {EXISTING_CSV}")

existing_df = pd.read_csv(EXISTING_CSV, encoding='utf-8')
missing_cols = [col for col in REQUIRED_COLUMNS if col not in existing_df.columns]
if missing_cols:
    raise ValueError(f"❌ CSV缺失必要列: {missing_cols} | 当前列: {list(existing_df.columns)}")

existing_indices = set(existing_df['index'].unique())
all_indices = set(original_df['index'])
missing_indices = sorted(all_indices - existing_indices)

print(f"📊 索引检查:")
print(f"   • 原始总词数: {len(all_indices)}")
print(f"   • 已分类: {len(existing_indices)} | 缺失: {len(missing_indices)}")
if missing_indices:
    print(f"   • 缺失索引示例: {missing_indices[:10]}...")

if not missing_indices:
    print("\n🎉 所有数据已完整！执行最终整理...")
    final_df = existing_df[REQUIRED_COLUMNS].copy()
    final_df = final_df.drop_duplicates(subset=['index'], keep='first').sort_values('index').reset_index(drop=True)
    final_df.to_csv(EXISTING_CSV, index=False, encoding='utf-8')
    
    # 生成统计
    stats = final_df['类别'].value_counts().reset_index()
    stats.columns = ['类别', '词数']
    stats.to_csv(EXISTING_CSV.replace('.csv', '_类别统计.csv'), index=False, encoding='utf-8')
    print(f"✅ 文件已标准化 | 统计报表已更新")
    exit(0)

# ========== 4. 提取缺失数据 ==========
missing_df = original_df[original_df['index'].isin(missing_indices)].copy()
word_tuples = [(row['index'], row['食物词'], row['频次']) for _, row in missing_df.iterrows()]
word_groups = [word_tuples[i:i+GROUP_SIZE] for i in range(0, len(word_tuples), GROUP_SIZE)]
print(f"\n🧠 步骤2/4: 补全 {len(missing_indices)} 个缺失项 | 分 {len(word_groups)} 组")

# ========== 5. 精准补全（严格四列追加）==========
print("\n🤖 步骤3/4: 开始补全流程...")
appended_count = 0

for group_idx, group in enumerate(tqdm(word_groups, desc="补全流程")):
    words_text = "\n".join([f"{idx}. {word}" for idx, word, _ in group])
    prompt = (
        f"分类要求：8类别（主食或者杂粮,豆类及制品,果汁与饮料,乳类及制品,蔬菜类,肉类,水果类,其它）\n"
        f"返回格式：索引,词,类别（严格三字段，逗号分隔）\n"
        f"示例：1,米饭,主食或者杂粮\n\n"
        f"待分类词：\n{words_text}"
    )
    
    retry = 0
    while retry < 3:
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "专业食物分类助手，返回格式：索引,词,类别"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.0
            )
            
            # 严格按四列顺序构建新行
            for line in response.choices[0].message.content.strip().split('\n'):
                parts = [p.strip() for p in line.split(',', 2)]
                if len(parts) < 3: 
                    continue
                
                try:
                    idx = int(parts[0])
                    word = parts[1]
                    category = parts[2]
                    
                    # 验证类别有效性
                    valid_cats = ["主食或者杂粮","豆类及制品","果汁与饮料","乳类及制品","蔬菜类","肉类","水果类","其它"]
                    if not any(cat in category for cat in valid_cats):
                        continue
                    
                    # 获取原始频次
                    orig = next((r for r in group if r[0] == idx), None)
                    if not orig:
                        continue
                    
                    # ✅ 关键修正：严格按四列顺序构建DataFrame
                    new_row = pd.DataFrame([{
                        'index': idx,
                        '食物词': word,
                        '频次': orig[2],
                        '类别': category
                    }], columns=REQUIRED_COLUMNS)  # 强制列顺序
                    
                    # 实时追加（无表头）
                    new_row.to_csv(EXISTING_CSV, mode='a', header=False, index=False, encoding='utf-8')
                    appended_count += 1
                    
                except Exception:
                    continue
            
            time.sleep(1.0)
            break
        except Exception as e:
            retry += 1
            if retry == 3:
                print(f"\n⚠️ 组 {group_idx+1} 失败: {str(e)[:80]}")
            else:
                time.sleep(3)

print(f"\n✅ 补全流程完成 | 新增 {appended_count} 条记录")

# ========== 6. 最终标准化（强制四列顺序）==========
print("\n💾 步骤4/4: 标准化最终文件...")
final_df = pd.read_csv(EXISTING_CSV, encoding='utf-8')

# 严格校验并重排四列
if not all(col in final_df.columns for col in REQUIRED_COLUMNS):
    raise ValueError(f"CSV列结构异常！当前列: {list(final_df.columns)}")
final_df = final_df[REQUIRED_COLUMNS].copy()

# 去重 + 按index排序
final_df = final_df.drop_duplicates(subset=['index'], keep='first').sort_values('index').reset_index(drop=True)
final_df.to_csv(EXISTING_CSV, index=False, encoding='utf-8')

# 生成统计
stats = final_df['类别'].value_counts().reset_index()
stats.columns = ['类别', '词数']
stats_file = EXISTING_CSV.replace('.csv', '_类别统计.csv')
stats.to_csv(stats_file, index=False, encoding='utf-8')

# 验证完整性
final_indices = set(final_df['index'])
missing_after = sorted(all_indices - final_indices)
print(f"\n🔍 完整性验证:")
print(f"   • 最终总词数: {len(final_df)}")
print(f"   • 仍缺失: {len(missing_after)}")
if missing_after:
    print(f"   • 未补全索引: {missing_after[:15]}...")
    print("💡 建议：检查API返回或手动补充剩余项")
else:
    print("🎉 所有数据已完整分类！")

print(f"\n✅ 标准化文件: {EXISTING_CSV} (列顺序: index → 食物词 → 频次 → 类别)")
print(f"✅ 统计报表: {stats_file}")
print(f"\n📊 类别分布:\n{stats.to_string(index=False)}")