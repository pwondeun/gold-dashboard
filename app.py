
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="XAUUSD Forecast (3 candles)", layout="centered")

st.title("🔮 XAU/USD H1 Forecast (Next 3 Candles)")
st.markdown("พยากรณ์ราคาทองคำ (XAU/USD) ใน Time Frame H1 ล่วงหน้า 3 แท่งด้วย Machine Learning (LSTM Model)")

# ตัวอย่างข้อมูลสุ่ม (จำลองการพยากรณ์)
np.random.seed(42)
directions = np.random.choice(["📈 ขึ้น", "📉 ลง"], size=3)
highs = np.round(np.random.uniform(2150, 2250, size=3), 2)
lows = highs - np.round(np.random.uniform(5, 15, size=3), 2)

data = {
    "แท่งที่": ["ถัดไป", "ถัดไป +1", "ถัดไป +2"],
    "ทิศทาง": directions,
    "High (คาดการณ์)": highs,
    "Low (คาดการณ์)": lows,
}

df = pd.DataFrame(data)

# ปุ่มรีเฟรช
if st.button("🔄 รีเฟรชการพยากรณ์"):
    st.experimental_rerun()

st.dataframe(df, use_container_width=True)
st.markdown("---")
st.caption("🚀 จัดทำโดย AI พยากรณ์ทองคำ | Powered by LSTM Model (Demo Version)")
