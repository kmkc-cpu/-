import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_advanced_assets():
    # 파일 경로 설정
    model_path = os.path.join(BASE_DIR, 'odor_advanced_model.pkl')
    feat_path = os.path.join(BASE_DIR, 'features_list.pkl')
    name_path = os.path.join(BASE_DIR, 'odor_names.pkl')

    try:
        # 파일 로드 (압축 없이 직접 로드)
        model = joblib.load(model_path)
        features = joblib.load(feat_path)
        odor_names = joblib.load(name_path)
        return model, features, odor_names
    except Exception as e:
        st.error(f"❌ AI 분석 엔진을 불러오지 못했습니다: {e}")
        st.stop()

st.set_page_config(page_title="고도화 악취 분석 AI", layout="wide")

try:
    model, features, odor_names = load_advanced_assets()
    st.title("👃 복합 악취 확률 정밀 분석 (High-Speed)")
    st.success("✅ 고도화 AI 모델 가동 중 (최적화 완료)")
    st.markdown("---")

    selected = st.multiselect("분석할 성분 검색 및 선택", sorted(features))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✍️ 농도 입력 (ppm)")
        inputs = {}
        if not selected:
            st.info("💡 성분을 선택해 주세요.")
        else:
            for s in selected:
                inputs[s] = st.number_input(f"{s}", min_value=0.0, format="%.6f", key=f"in_{s}")

    with col2:
        st.subheader("📊 AI 분석 결과")
        if st.button("정밀 분석 실행", type="primary", use_container_width=True):
            if selected:
                # 데이터 구성
                full_input = {f: 0.0 for f in features}
                for s, v in inputs.items():
                    full_input[s] = v
                
                input_df = pd.DataFrame([full_input])[features]
                prediction = model.predict(input_df)[0]
                
                # 결과 정리
                res_df = pd.DataFrame({'냄새': odor_names, '확률': prediction}).sort_values('확률', ascending=False)
                
                st.write(f"### 가장 유력한 냄새: :red[{res_df.iloc[0]['냄새']}]")
                
                # 막대 그래프 (Top 10)
                fig, ax = plt.subplots(figsize=(10, 5))
                top_10 = res_df.head(10)
                ax.bar(top_10['냄새'], top_10['확률'], color='skyblue')
                plt.xticks(rotation=45)
                ax.set_ylabel("예측 확률")
                st.pyplot(fig)
                
                st.dataframe(res_df.head(10), use_container_width=True)

except Exception as e:
    st.error(f"시스템 오류: {e}")