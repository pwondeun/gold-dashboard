
import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ดึงข้อมูลราคาทองคำ (XAU/USD) รายชั่วโมง
data = yf.download("XAUUSD=X", interval="60m", period="30d")
data.dropna(inplace=True)

# สร้างฟีเจอร์สำหรับ Machine Learning
data['Return'] = data['Close'].pct_change()
data['High_Low'] = data['High'] - data['Low']
data['Open_Close'] = data['Close'] - data['Open']
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

# เตรียมข้อมูลสำหรับฝึกโมเดล
X = data[['Return', 'High_Low', 'Open_Close']]
y = data['Target']

# ตรวจสอบก่อนว่าเพียงพอสำหรับแบ่ง train/test หรือไม่
if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    latest = X.tail(1)
    prediction = model.predict(latest)[0]
else:
    accuracy = 0.0
    prediction = 0  # หรือ None

# === Streamlit Dashboard ===
st.set_page_config(page_title="Gold Price ML Dashboard", layout="centered")
st.title("📊 วิเคราะห์ราคาทองคำด้วย Machine Learning")

st.subheader("🔮 คำทำนายแท่งถัดไป:")
st.success("ขึ้น ⬆️" if prediction == 1 else "ลง ⬇️")

st.metric(label="📊 ความแม่นยำของโมเดล", value=f"{round(accuracy * 100, 2)} %")

# แสดงกราฟราคาทองคำย้อนหลัง
st.subheader("📈 ราคาทองคำย้อนหลัง (100 แท่ง)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Close'][-100:], label='ราคาทองคำ (XAU/USD)', color='gold')
ax.set_title('ราคาทองคำย้อนหลัง')
ax.set_xlabel('เวลา')
ax.set_ylabel('ราคา')
ax.grid(True)
st.pyplot(fig)
