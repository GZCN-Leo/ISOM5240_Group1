import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 设置页面标题和布局
st.set_page_config(page_title="Yelp Review Classifier", layout="centered")

@st.cache_resource
def load_model():
    # 直接从 Hugging Face Hub 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(
        "isom5240/2026Spring5240L1", 
        num_labels=5
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

st.title("📝 Yelp 评论星级预测")
st.markdown("输入一段 Yelp 评论，模型将预测其星级（1 星到 5 星）。")

# 加载模型（显示加载状态）
with st.spinner("正在加载模型，请稍候..."):
    model, tokenizer = load_model()
st.success("模型加载成功！")

# 用户输入文本框
user_input = st.text_area("输入评论内容：", height=200, placeholder="将评论粘贴在这里...")

# 预测按钮
if st.button("预测星级"):
    if not user_input.strip():
        st.warning("请输入文本内容。")
    else:
        # 对输入文本进行分词
        inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
        
        # 推理（不计算梯度）
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_class = np.argmax(probabilities)
        
        # 假设类别 0 对应 1 星，类别 4 对应 5 星
        star_rating = pred_class + 1
        
        # 显示预测结果
        st.subheader("预测结果")
        st.metric(label="预测星级", value=f"{star_rating} ⭐")
        
        # 显示各类概率
        st.write("**各星级置信度：**")
        prob_df = pd.DataFrame({
            "星级": [f"{i+1} 星" for i in range(5)],
            "概率": probabilities
        })
        
        # 柱状图展示概率分布
        st.bar_chart(prob_df.set_index("星级")["概率"])
        
        # 表格显示详细概率
        st.dataframe(prob_df.style.format({"概率": "{:.4f}"}), use_container_width=True)
