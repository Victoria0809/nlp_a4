import streamlit as st
import streamlit.components.v1 as components
import nltk
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# ====== 关键：设置环境变量强制离线模式 ======
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '.cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(__file__), '.cache')
#os.environ['TRANSFORMERS_OFFLINE'] = '1'  # ⚠️ 强制离线模式
#os.environ['HF_DATASETS_OFFLINE'] = '1'

from transformers import BertTokenizer, BertModel


nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)


# ====== 1. 缓存加载 spaCy 模型 ======
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("⚠️ 正在下载 spaCy 模型...这可能需要几分钟时间")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

# ====== 添加 NLTK 资源自动下载 ======
def download_nltk_resources():
    """自动下载所需的 NLTK 资源"""
    import ssl

    # 处理 SSL 证书问题
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # 下载所需资源
    required_resources = ['punkt_tab', 'punkt', 'wordnet', 'omw-1.4']

    for resource in required_resources:
        try:
            nltk.data.find(resource)
            print(f"✅ {resource} 已存在")
        except LookupError:
            try:
                print(f"📥 正在下载 {resource}...")
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                print(f"✅ {resource} 下载成功")
            except Exception as e:
                print(f"❌ {resource} 下载失败: {e}")
                # 如果 punkt_tab 下载失败，尝试 punkt
                if resource == 'punkt_tab':
                    try:
                        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
                        print("✅ punkt 下载成功（作为备用）")
                    except:
                        pass

# 在 initialize_app() 函数中调用
def initialize_app():
    # 先下载 NLTK 资源
    download_nltk_resources()

    nlp = load_spacy_model()
    tokenizer, model = load_bert_model()
    return nlp, tokenizer, model

# ====== 2. 缓存加载 BERT 模型和分词器 ======
@st.cache_resource
def load_bert_model():
    try:
        st.info("🔄 正在加载 BERT 模型...")

        # 从 Hugging Face Hub 自动下载
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")

        st.success("✅ BERT 模型加载成功！")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ BERT 模型加载失败: {str(e)}")
        raise

# 初始化应用
def initialize_app():
    nlp = load_spacy_model()
    tokenizer, model = load_bert_model() 
    return nlp, tokenizer, model

# 词义消歧对比测试
def wsd_tab(nlp, tokenizer, model):
    st.header("🧠 词义消歧 (WSD) 对比测试")

    # 用户输入
    text1 = st.text_area("请输入包含多义词的句子", "I went to the bank to deposit my money.")
    target_word = st.text_input("请输入目标多义词", "bank")
    text2 = st.text_area("请输入第二个对比句子", "I sat by the river bank.")

    if st.button("分析", key="wsd_analyze"):
        if not tokenizer or not model:
            st.error("BERT模型未加载，请检查网络连接")
            return

        # 在 wsd_tab 函数中，修改 Lesk 部分：

        # 传统方法 (Lesk)
        st.subheader("传统方法 (Lesk)")
        try:
            from nltk.wsd import lesk
            from nltk.tokenize import word_tokenize

            # 分词
            tokens1 = word_tokenize(text1)
            tokens2 = word_tokenize(text2)

            # 应用Lesk算法
            synset1 = lesk(tokens1, target_word)
            synset2 = lesk(tokens2, target_word)

            if synset1:
                st.write(f"句子1 - Lesk 算法预测的 Synset: {synset1}")
                st.write(f"定义: {synset1.definition()}")
            else:
                st.error("句子1 - Lesk 算法未能找到匹配的词义")

            if synset2:
                st.write(f"句子2 - Lesk 算法预测的 Synset: {synset2}")
                st.write(f"定义: {synset2.definition()}")
            else:
                st.error("句子2 - Lesk 算法未能找到匹配的词义")

        except Exception as e:
            st.error(f"Lesk 算法执行失败: {e}")
            
        # 深度学习方法 (BERT)
        st.subheader("深度学习方法 (BERT)")
        try:
            # 提取第一个句子中目标词的BERT向量
            vector1 = get_bert_embedding(text1, target_word, tokenizer, model)
            # 提取第二个句子中目标词的BERT向量
            vector2 = get_bert_embedding(text2, target_word, tokenizer, model)

            if vector1 is not None and vector2 is not None:
                # 计算余弦相似度
                similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

                st.write("第一个句子的 BERT 向量:")
                st.write(vector1[:10])  # 只显示前10个元素
                st.write("第二个句子的 BERT 向量:")
                st.write(vector2[:10])  # 只显示前10个元素
                st.write(f"余弦相似度: {similarity:.4f}")

                # 添加解释
                if similarity > 0.7:
                    st.success("✅ 两个句子中的目标词含义相似")
                elif similarity > 0.4:
                    st.warning("⚠️ 两个句子中的目标词含义有一定差异")
                else:
                    st.error("❌ 两个句子中的目标词含义完全不同")
            else:
                st.error("无法提取BERT向量，请检查目标词是否在句子中")
        except Exception as e:
            st.error(f"BERT 执行失败: {e}")

# 获取BERT上下文词向量
def get_bert_embedding(text, target_word, tokenizer, model):
    # 分词
    tokens = tokenizer.tokenize(text)
    
    # 找到目标词的子词位置
    target_subword_positions = []
    for i, token in enumerate(tokens):
        # 检查token是否是目标词的一部分（考虑子词切分，如bank可能被切分为bank或保持不变）
        if token.lower().replace('##', '') in target_word.lower() or target_word.lower() in token.lower():
            target_subword_positions.append(i)
    
    if not target_subword_positions:
        return None
    
    # 添加特殊标记并编码
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding['input_ids']
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**encoding)
        embeddings = outputs.last_hidden_state[0]
    
    # 找到目标词对应的token位置（考虑[CLS]等特殊标记的偏移）
    target_token_positions = []
    # 从1开始，跳过[CLS]标记
    for i in range(1, len(input_ids[0])-1):  # 也跳过最后的[SEP]标记
        token = tokenizer.decode([input_ids[0][i]])
        # 检查token是否是目标词的一部分
        if token.lower().replace('##', '') in target_word.lower() or target_word.lower() in token.lower():
            target_token_positions.append(i)
    
    if not target_token_positions:
        return None
    
    # 计算目标词所有子词的平均向量
    target_embedding = torch.mean(embeddings[target_token_positions], dim=0).numpy()
    return target_embedding

# 在文件末尾添加这些函数
def perform_simple_wsd(text, nlp, tokenizer, model):
    """简化版词义消歧"""
    # 这里可以实现简单的基于上下文的方法
    doc = nlp(text)
    results = []
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            # 获取词向量
            embedding = get_bert_embedding(text, token.text, tokenizer, model)
            if embedding is not None:
                results.append({
                    'word': token.text,
                    'pos': token.pos_,
                    'embedding': embedding[:5]  # 只保存前5维用于显示
                })
    return results

def perform_wsd(text, nlp, tokenizer, model):
    """完整版词义消歧（需要更多依赖）"""
    # 这里可以集成更复杂的WSD算法
    # 目前先使用简化版本
    return perform_simple_wsd(text, nlp, tokenizer, model)

# 语义角色标注提取与可视化
def srl_tab(nlp, model):
    st.header("🏷️ 语义角色标注 (SRL) 提取与可视化")
    
    # 用户输入
    text = st.text_area("请输入一个英文句子", "Apple is manufacturing new smartphones in China this year.")
    
    if st.button("分析", key="srl_analyze"):
        if not nlp:
            st.error("spaCy模型未加载，请先安装")
            return
        
        # 使用spaCy进行处理
        doc = nlp(text)
        
        # 谓词识别：找到依存树的根节点（通常是主要动词）
        predicate = None
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                predicate = token
                break
        
        # 论元映射
        roles = []
        if predicate:
            # 遍历所有token，寻找论元
            for token in doc:
                if token.dep_ == "nsubj" and token.head == predicate:
                    roles.append({"语义角色": "A0 (施事者/Agent)", "原文": token.text})
                elif token.dep_ == "nsubjpass" and token.head == predicate:
                    # 处理被动语态的主语
                    roles.append({"语义角色": "A1 (受事者/Patient)", "原文": token.text})
                elif token.dep_ == "dobj" and token.head == predicate:
                    roles.append({"语义角色": "A1 (受事者/Patient)", "原文": token.text})
                elif token.dep_ == "iobj" and token.head == predicate:
                    roles.append({"语义角色": "A2 (受益者/Recipient)", "原文": token.text})
                elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                    # 处理介词短语
                    prep = token.head
                    if prep.text.lower() in ["in", "at", "on", "from", "to", "by"]:
                        if prep.text.lower() == "by" and token.head.head == predicate:
                            # 处理被动语态的施事者
                            roles.append({"语义角色": "A0 (施事者/Agent)", "原文": f"{prep.text} {token.text}"})
                        else:
                            # 处理地点
                            roles.append({"语义角色": "AM-LOC (地点)", "原文": f"{prep.text} {token.text}"})
                elif (token.dep_ == "npadvmod" or token.ent_type_ == "DATE") and token.head == predicate:
                    # 处理时间实体
                    roles.append({"语义角色": "AM-TMP (时间)", "原文": token.text})
                elif token.dep_ == "advmod" and token.head == predicate:
                    # 处理方式副词
                    roles.append({"语义角色": "AM-MNR (方式)", "原文": token.text})
                elif token.dep_ == "auxpass" and token.head == predicate:
                    # 处理被动语态助动词
                    roles.append({"语义角色": "AM-PASS (被动)", "原文": token.text})
        
        # 显示结果
        if roles:
            st.subheader("语义角色标注结果")
            st.table(roles)
        else:
            st.info("未找到符合规则的谓词-论元结构。")

        # 显示依存关系图（修复版 - 解决显示不全问题）
        st.subheader("依存关系图")
        try:
            # 使用更大的高度和更好的布局选项
            dep_html = spacy.displacy.render(
                doc,
                style="dep",
                options={
                    "compact": False,  # 关闭紧凑模式，避免标签重叠
                    "distance": 150,  # 增加节点间距
                    "word_spacing": 60,  # 增加词间距
                    "arrow_stroke": 2,  # 箭头粗细
                    "arrow_width": 8  # 箭头宽度
                },
                jupyter=False
            )
            # 增加容器高度以适应更大的图
            components.html(dep_html, height=500, scrolling=True)
        except Exception as e:
            st.error(f"依存图渲染失败: {e}")

# 主应用
def main():
    st.title("Deep Semantic Analysis Platform")
    
    # 初始化
    nlp, tokenizer, model = initialize_app()
    
    # 创建标签页
    tab1, tab2 = st.tabs(["词义消歧 (WSD) 对比测试", "语义角色标注 (SRL) 提取与可视化"])
    
    with tab1:
        wsd_tab(nlp, tokenizer, model)
    
    with tab2:
        srl_tab(nlp, model)

if __name__ == "__main__":
    main()