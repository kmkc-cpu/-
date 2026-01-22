import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px  # 한글 깨짐 해결을 위한 라이브러리

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_advanced_assets():
    # 파일 경로 설정
    model_path = os.path.join(BASE_DIR, 'odor_advanced_model.pkl')
    feat_path = os.path.join(BASE_DIR, 'features_list.pkl')
    name_path = os.path.join(BASE_DIR, 'odor_names.pkl')

    try:
        # 파일 로드 (최적화된 pkl 파일 직접 로드)
        model = joblib.load(model_path)
        features = joblib.load(feat_path)
        odor_names = joblib.load(name_path)
        return model, features, odor_names
    except Exception as e:
        st.error(f"❌ AI 분석 엔진을 불러오지 못했습니다: {e}")
        st.stop()

# --- 페이지 설정 ---
st.set_page_config(page_title="고도화 악취 분석 AI", layout="wide")

try:
    model, features, odor_names = load_advanced_assets()
    st.title("👃 복합 악취 확률 정밀 분석 시스템")
    st.success("✅ 고도화 AI 모델 가동 중 (한글 폰트 최적화 완료)")
    st.markdown("---")

    # 성분 선택 (Multi-select)
    selected = st.multiselect(
        "분석할 성분들을 검색하여 선택하세요.", 
        options=sorted(features),
        help="성분 이름을 입력하면 리스트에서 필터링됩니다."
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✍️ 농도 입력 (ppm)")
        user_inputs = {}
        if not selected:
            st.info("💡 위 검색창에서 분석할 성분을 먼저 선택해 주세요.")
        else:
            for s in selected:
                user_inputs[s] = st.number_input(
                    f"{s}", 
                    min_value=0.0, 
                    format="%.6f", 
                    key=f"input_{s}"
                )

    with col2:
        st.subheader("📊 AI 분석 결과")
        if st.button("정밀 분석 실행", type="primary", use_container_width=True):
            if not selected:
                st.warning("⚠️ 입력된 성분이 없습니다.")
            else:
                # 1. AI 입력 데이터 구성
                full_input_data = {f: 0.0 for f in features}
                for s, v in user_inputs.items():
                    full_input_data[s] = v
                
                input_df = pd.DataFrame([full_input_data])[features]
                
                # 2. AI 예측 실행
                prediction = model.predict(input_df)[0]
                
                # 3. 결과 데이터 정리
                res_df = pd.DataFrame({
                    '냄새 종류': odor_names, 
                    '확률(%)': [round(p * 100, 2) for p in prediction]
                }).sort_values('확률(%)', ascending=False)
                
                # 상위 결과 출력
                top_odor = res_df.iloc[0]
                st.write(f"### 가장 유력한 냄새는 **:red[{top_odor['냄새 종류']}]** 입니다.")
                
                # 4. Plotly 그래프 출력 (한글 깨짐 해결 및 인터랙티브 기능)
                top_10 = res_df.head(10)
                fig = px.bar(
                    top_10, 
                    x='냄새 종류', 
                    y='확률(%)', 
                    title='AI 분석 복합 악취 확률 Top 10',
                    color='확률(%)',
                    color_continuous_scale='Viridis',
                    text='확률(%)'
                )
                
                # 그래프 레이아웃 세부 설정
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 데이터 표
                st.write("📋 **상세 확률 리스트 (Top 10)**")
                st.table(top_10)

except Exception as e:
    st.error(f"시스템 오류 발생: {e}")