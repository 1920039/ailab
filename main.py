import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="전공적합성 검사", page_icon="Compass", layout="wide")

# 버튼 색상 지정
st.markdown(
    """
    <style>
    /* 모든 Streamlit 버튼 공통 스타일 */
    div.stButton > button, div.stFormSubmitButton > button {
        width: 100%;
        height: 3em;
        border-radius: 8px;
        background-color: #4B8BF5; /* 올바른 색상 코드 */
        color: white;
        font-weight: 600;
        font-size: 1em;
        border: none;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #3A6CD8; /* hover 시 살짝 어두운 파랑 */
        transition: 0.2s;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------
# 상수 정의
# -------------------------
TYPE_ORDER = ["A", "B", "C", "D", "E", "F"]

QUESTIONS = [
    {"id": 1, "text": "탐구적인", "type": "B"}, {"id": 2, "text": "개성적인", "type": "D"},
    {"id": 3, "text": "시사적인", "type": "E"}, {"id": 4, "text": "관찰적인", "type": "B"},
    {"id": 5, "text": "기계적인", "type": "A"}, {"id": 6, "text": "감성적인", "type": "D"},
    {"id": 7, "text": "공감적인", "type": "C"}, {"id": 8, "text": "성과적인", "type": "F"},
    {"id": 9, "text": "실험적인", "type": "B"}, {"id": 10, "text": "표현적인", "type": "D"},
    {"id": 11, "text": "설득적인", "type": "C"}, {"id": 12, "text": "질서적인", "type": "A"},
    {"id": 13, "text": "분석적인", "type": "B"}, {"id": 14, "text": "현실적인", "type": "A"},
    {"id": 15, "text": "비판적인", "type": "E"}, {"id": 16, "text": "열성적인", "type": "C"},
    {"id": 17, "text": "호기심적인", "type": "D"}, {"id": 18, "text": "선도적인", "type": "F"},
    {"id": 19, "text": "생산적인", "type": "A"}, {"id": 20, "text": "호소력있는", "type": "E"},
    {"id": 21, "text": "방법적인", "type": "B"}, {"id": 22, "text": "경제적인", "type": "F"},
    {"id": 23, "text": "통솔적인", "type": "E"}, {"id": 24, "text": "과학적인", "type": "B"},
    {"id": 25, "text": "선택적인", "type": "F"}, {"id": 26, "text": "독창적인", "type": "D"},
    {"id": 27, "text": "미적인", "type": "D"}, {"id": 28, "text": "효율적인", "type": "A"},
    {"id": 29, "text": "유연적인", "type": "F"}, {"id": 30, "text": "외향적인", "type": "E"},
    {"id": 31, "text": "질서정연한", "type": "B"}, {"id": 32, "text": "영향력있는", "type": "E"},
    {"id": 33, "text": "교육적인", "type": "C"}, {"id": 34, "text": "실용적인", "type": "A"},
    {"id": 35, "text": "환상적인", "type": "D"}, {"id": 36, "text": "계산적인", "type": "B"},
    {"id": 37, "text": "전략적인", "type": "F"}, {"id": 38, "text": "이타적인", "type": "C"},
    {"id": 39, "text": "리더십있는", "type": "E"}, {"id": 40, "text": "활동적인", "type": "D"},
    {"id": 41, "text": "수학적인", "type": "A"}, {"id": 42, "text": "이윤추구적인", "type": "F"},
    {"id": 43, "text": "헌신적인", "type": "C"}, {"id": 44, "text": "도전적인", "type": "E"},
    {"id": 45, "text": "혁신적인", "type": "A"}, {"id": 46, "text": "미래예측적인", "type": "F"},
    {"id": 47, "text": "우호적인", "type": "C"}, {"id": 48, "text": "자기주장적인", "type": "D"},
    {"id": 49, "text": "해석적인", "type": "B"}, {"id": 50, "text": "성취지향적인", "type": "E"},
    {"id": 51, "text": "논리적인", "type": "C"}, {"id": 52, "text": "계획적인", "type": "A"},
    {"id": 53, "text": "배려적인", "type": "C"}, {"id": 54, "text": "합리적인", "type": "F"},
    {"id": 55, "text": "사명감있는", "type": "B"}, {"id": 56, "text": "조직적인", "type": "E"},
    {"id": 57, "text": "실제적인", "type": "A"}, {"id": 58, "text": "분배적인", "type": "F"},
    {"id": 59, "text": "창조적인", "type": "D"}, {"id": 60, "text": "설명적인", "type": "C"},
]
assert len(QUESTIONS) == 60

TYPE_DESCRIPTIONS = {
    "A": {"title": "A타입: 공학 기술적 성향", "items": ["**[공과대학]**", "건축학부", "건축공학부", "건설환경공학과", "도시공학과", "자연환경공학과", "융합전자공학부", "컴퓨터전공", "소프트웨어전공", "정보시스템학과", "전기공학전공", "생체공학전공", "신소재공학부", "화학공학과", "생명공학과", "유기나노공학과", "에너지공학과", "기계공학부", "원자력공학과", "산업공학과", "미래자동차공학과"]},
    "B": {"title": "B타입: 자연과학적 성향", "items": ["**[의과대학]**", "의예과", "의학과", "**[사범대학]**", "수학교육과", "**[생활과학대학]**", "식품영양학과", "**[자연과학대학]**", "수학과", "물리학과", "화확과", "생명과학화", "**[간호학부]**"]},
    "C": {"title": "C타입: 인문 어문 교육적 성향", "items": ["**[인문과학대학]**", "국어국문학과", "중어중문학과", "영어영문학과", "독어독문학과", "사학과", "철학과", "**[사회과학대학]**", "관광학부"]},
    "D": {"title": "D타입: 예술 창의적 성향", "items": ["**[사범대학]**", "응용미술교육과", "**[음악대학]**", "성악과", "작곡과", "피아노과", "관현악과", "국악과", "**[예술,체육대학]**", "체육학과", "연극영화과", "무용학과", "**[생활과학대학]**", "의류학과", "실내건축디자인과"]},
    "E": {"title": "E타입: 사회과학 글로벌", "items": ["**[사회과학대학]**", "정치외교학과", "사회학과", "미디어커뮤니케이션학과", "**[정책과학대학]**", "정책학과", "행정학과", "**[사범대학]**", "교육공학과", "**[국제학부]**"]},
    "F": {"title": "F타입: 경제 효율지향적 성향", "items": ["**[예술, 체육대학]**", "스포츠산업학과", "**[경영대학]**", "경영학부", "파이낸스경영학과", "**[경제금융학부]**"]},
}

# -------------------------
# 함수 정의
# -------------------------
def score_types(questions, responses):
    scores = {t: 0 for t in TYPE_ORDER}
    for q in questions:
        if responses.get(q["id"], False):
            scores[q["type"]] += 1
    return scores

def top_types(scores):
    max_val = max(scores.values()) if scores else 0
    if max_val == 0:
        return [], 0
    ties = [t for t, v in scores.items() if v == max_val]
    return ties, max_val

def render_type_description(t):
    data = TYPE_DESCRIPTIONS.get(t)
    if not data:
        st.info("해당 타입 설명 데이터가 없습니다.")
        return
    st.markdown(f"### {data.get('title','')}")
    items = data.get("items", [])
    if items:
        st.markdown("<br>".join(items), unsafe_allow_html=True)
        #st.markdown("- " + "\n- ".join(items))

# -------------------------
# UI 시작
# -------------------------
st.title("전공적합성 검사")
st.caption("나에게 맞는 전공은 무엇일까요? 그 전공에는 어떤 진로가 있을까요?")

# 세션 초기화
if "responses" not in st.session_state:
    st.session_state.responses = {}

st.subheader("해당 문항에 체크하세요.")

# 폼 + 4열 체크박스
cols = st.columns(4)
with st.form("question_form", clear_on_submit=False):
    for q in QUESTIONS:
        idx = (q["id"] - 1) % 4
        with cols[idx]:
            st.checkbox(
                f"{q['id']:02d}. {q['text']}",
                value=st.session_state.responses.get(q["id"], False),
                key=f"q_{q['id']}",
                help=f"타입: {q['type']}",
            )
    submitted = st.form_submit_button("결과 보기", width='stretch')

# 제출 후 처리
if submitted:
    st.session_state.responses = {
        q["id"]: bool(st.session_state.get(f"q_{q['id']}", False))
        for q in QUESTIONS
    }

# 결과 표시 (응답이 있을 때만)
if st.session_state.responses:
    # 점수 계산 (항상 최신 응답 기준)
    scores = score_types(QUESTIONS, st.session_state.responses)

    # 표 + 그래프 (동시에 계산)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 타입별 점수")

        df_scores = pd.DataFrame([{"타입": t, "점수": scores[t]} for t in TYPE_ORDER])
        st.dataframe(df_scores, width='stretch', height=246)

    with col2:
        st.markdown("### 타입별 점수 방사형 그래프")

        labels = TYPE_ORDER
        values = [scores[t] for t in labels] + [scores[labels[0]]]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

        fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=120, subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, color="#1f77b4")
        ax.fill(angles, values, alpha=0.25, color="#1f77b4")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_title("")
        plt.tight_layout()
        st.pyplot(fig, width='content')

    # 최종 결과
    ties, max_val = top_types(scores)
    if not ties:
        st.info("체크된 문항이 없습니다.")
    else:
        if len(ties) == 1:
            chosen = ties[0]
            st.success(f"최종 타입: **{chosen}** (점수: {max_val}/10)")
        else:
            st.warning(f"여러 타입이 동점입니다 (최고점 {max_val}점): {', '.join(ties)}")
            chosen = st.radio("동점 타입 중 하나를 선택하세요", ties, index=0, horizontal=True, key="tie_radio")

        st.markdown("---")
        render_type_description(chosen)

else:
    st.info("문항을 체크하고 **결과 보기** 버튼을 눌러주세요.")




if st.button("AI에게 분석 요청하기(비밀번호 필요)", width='stretch'):

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # 비번 입력해야 LLM 사용 가능하도록
    PASSWORD = st.secrets["APP_PASSWORD"]  # secrets.toml에 저장

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        st.subheader("🔒 인증이 필요합니다")
        pw = st.text_input("비밀번호를 입력하세요", type="password")
        if st.button("로그인"):
            if pw == PASSWORD:
                st.session_state.auth = True
                st.success("인증 성공! AI 기능을 사용할 수 있습니다.")
            else:
                st.error("비밀번호가 틀렸습니다.")
        st.stop()  # 아래 코드 실행 방지



    checked_items = [q["text"] for q in QUESTIONS if st.session_state.responses.get(q["id"], False)]
    chosen_type = chosen
    prompt = f"""
    사용자가 전공적합성 검사에서 {len(checked_items)}개 문항을 선택했습니다.
    선택한 항목: {checked_items}
    타입은 다음의 6개가 있습니다: "공학 기술적 성향", "자연과학적 성향", "인문 어문 교육적 성향", "예술 창의적 성향", "사회과학 글로벌 성향", "경제 효율지향적 성향".
    검사 결과 타입: {chosen_type}
    이 사람의 성향과 적합한 진로, 학과 선택 조언을 해 주세요.

    """

    with st.spinner("AI가 분석 중입니다..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 진로·적성 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
        )

    st.markdown("### 💬 AI의 분석 결과")
    st.write(response.choices[0].message.content)

    if st.button("AI에게 추가 질문하기", width='stretch'):
        followup = st.text_input("무엇이 궁금한가요?", key="followup_input")
        if followup:
            followup_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 진로·적성 분석 전문가입니다."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": followup}
                ]
            )
            st.write(followup_response.choices[0].message.content)
