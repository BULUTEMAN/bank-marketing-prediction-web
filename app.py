import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 1. 页面基础配置
# ============================================================
st.set_page_config(
    page_title="银行定期存款订阅预测系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. 自定义CSS美化样式
# ============================================================
custom_css = """
<style>
.main > div {padding-top: 1.5rem;}
.block-container {
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}
.card{
    background-color:#f7f9fc;
    padding:24px;
    border-radius:14px;
    border:1px solid #e2e8f0;
    margin:12px 0px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.title-card{
    background: linear-gradient(135deg, #1a365d 0%, #2b4870 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 20px;
}
.title-card h1{color:white; margin:0;}
.title-card p{color:#cbd5e0; margin-top:8px; font-size:16px;}
h2{color:#2b4870;}
.stAlert{border-radius:10px;}
.stButton > button{
    border-radius: 8px;
    font-weight: 600;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# 3. 页面标题区
# ============================================================
st.markdown("""
<div class="title-card">
<h1>🏦 银行客户定期存款订阅预测系统</h1>
<p>基于机器学习模型，输入客户特征，预测客户订阅定期存款产品的概率，辅助电话营销资源分配决策</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4. 侧边栏：客户特征录入区
# ============================================================
with st.sidebar:
    st.header("📋 客户信息录入")
    st.caption("请填写客户全部特征后点击预测")

    st.subheader("基本信息")
    age = st.number_input("年龄", min_value=18, max_value=100, value=35)
    job = st.selectbox("职业", [
        "admin", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
    marital = st.selectbox("婚姻状态", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("学历", ["primary", "secondary", "tertiary", "unknown"])

    st.subheader("资产与负债")
    default = st.selectbox("是否信用卡违约", ["no", "yes"])
    housing = st.selectbox("是否有住房贷款", ["no", "yes"])
    loan = st.selectbox("是否有个人贷款", ["no", "yes"])

    st.subheader("营销接触信息")
    contact = st.selectbox("联系方式", ["cellular", "telephone", "nonexistent"])
    month = st.selectbox("最后联系月份", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    day_of_week = st.selectbox("联系星期", ["mon", "tue", "wed", "thu", "fri"])
    duration = st.number_input("通话时长(秒)", min_value=0, value=120)
    campaign = st.number_input("本次活动联系次数", min_value=1, value=2)
    pdays = st.number_input("距离上次联系间隔天数", min_value=-1, value=99)
    previous = st.number_input("之前营销联系次数", min_value=0, value=0)
    poutcome = st.selectbox("上一次营销结果", ["success", "failure", "nonexistent"])

    st.subheader("宏观经济指标")
    emp_var_rate = st.number_input("就业变动率", value=1.1)
    cons_price_index = st.number_input("消费价格指数", value=93.994)
    cons_conf_index = st.number_input("消费者信心指数", value=-36.4)
    lending_rate3m = st.number_input("欧元同业拆借利率", value=4.857)
    nr_employed = st.number_input("雇员数量", value=5191.0)

    st.divider()
    predict_btn = st.button("🔮 执行预测", type="primary", use_container_width=True)

# ============================================================
# 5. 主页面：预测结果展示
# ============================================================
st.divider()

if predict_btn:
    # ✅ 修复：严格和训练集列名完全保持一致
    input_df = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month],
        "day_of_week": [day_of_week],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome],
        "emp_var_rate": [emp_var_rate],
        "cons_price_index": [cons_price_index],
        "cons_conf_index": [cons_conf_index],
        "lending_rate3m": [lending_rate3m],
        "nr_employed": [nr_employed]
    })

    loaded_data = joblib.load("bank_marketing_model.pkl")
    preprocessor = loaded_data["preprocessor"]
    model = loaded_data["model"]

    X_transformed = preprocessor.transform(input_df)
    pred_proba = model.predict_proba(X_transformed)
    prob_yes = round(float(pred_proba[0][1]), 4)

    col1, col2 = st.columns([1, 1])

    # 左侧：预测结果卡片
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 预测结果")
        st.metric(
            label="客户订阅定期存款概率",
            value=f"{prob_yes * 100:.2f} %"
        )
        if prob_yes >= 0.5:
            st.success("✅ 高意向客户，大概率订阅定期存款产品")
        elif prob_yes >= 0.2:
            st.warning("⚠️ 中等意向客户，存在订阅可能性，需持续跟进")
        else:
            st.error("❌ 低意向客户，订阅概率较低，建议减少营销投入")
        st.markdown('</div>', unsafe_allow_html=True)

    # 右侧：业务建议卡片
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 营销业务建议")
        if prob_yes >= 0.5:
            st.write("**优先级：高**")
            st.write("- 该客户订阅意向较高，建议优先安排营销人员跟进通话；")
            st.write("- 通话中重点介绍定期存款产品的利率优势与安全性；")
            st.write("- 可适当提供限时优惠活动，促进客户尽快决策。")
        elif prob_yes >= 0.2:
            st.write("**优先级：中**")
            st.write("- 客户存在潜在订阅意向，建议适度跟进，控制拨打频次；")
            st.write("- 可先通过短信、邮件等方式传递产品信息，培育客户意向；")
            st.write("- 避免过度骚扰，防止客户产生反感。")
        else:
            st.write("**优先级：低**")
            st.write("- 客户订阅可能性很低，建议减少主动营销呼叫次数；")
            st.write("- 将营销资源优先分配给高意向客户，提升整体转化率；")
            st.write("- 可保留在客户池中，待宏观经济环境变化后再行评估。")
        st.markdown('</div>', unsafe_allow_html=True)

    # 底部：输入特征回显
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 本次预测输入特征")
    st.dataframe(input_df.T, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    col_guide1, col_guide2 = st.columns([1, 1])
    with col_guide1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚀 使用步骤")
        st.write("1. 在左侧侧边栏填写客户的全部特征信息；")
        st.write("2. 点击侧边栏底部的【执行预测】按钮；")
        st.write("3. 页面将展示订阅概率、客户意向等级与营销建议。")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_guide2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 模型说明")
        st.write("- 模型类型：分类预测模型，输出客户订阅概率；")
        st.write("- 输入维度：20项客户特征，涵盖基本信息、资产负债、营销接触、宏观经济；")
        st.write("- 概率阈值：≥50% 判定为高意向，20%~50% 为中等意向，<20% 为低意向。")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("银行营销数据分析 | 基于 Streamlit 构建的预测演示原型")
