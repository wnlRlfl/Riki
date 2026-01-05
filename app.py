import streamlit as st
import trafilatura
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다. (로컬 개발 환경용)
load_dotenv()

# Streamlit 페이지의 기본 설정을 구성합니다.
# page_title: 브라우저 탭에 표시될 제목
# page_icon: 브라우저 탭 아이콘
# layout: 페이지 레이아웃 ('wide'는 전체 너비 사용)
st.set_page_config(
    page_title="AI 독해력 트레이너",
    page_icon="📚",
    layout="wide"
)

# OpenAI 클라이언트 초기화 (API Key 확인)
# 1순위: Streamlit Secrets (클라우드 환경)
# 2순위: 로컬 환경 변수 (.env)
api_key = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    # 로컬 실행 시 .streamlit/secrets.toml 파일이 없으면 에러가 발생하므로 무시하고 진행
    pass

# Secrets에서 못 찾았으면 환경 변수 확인
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

# API 키가 아직도 없으면 Session State 확인
if not api_key:
    # Streamlit의 session_state(세션 상태)에 키가 저장되어 있는지 확인합니다.
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = ""

# --- Helper Functions (보조 함수 정의) ---

def extract_text_from_url(url):
    """
    URL에서 본문 텍스트를 추출하는 함수입니다.
    trafilatura 라이브러리를 사용하여 웹페이지의 주요 콘텐츠만 가져옵니다.
    """
    try:
        # 디버깅을 위해 URL을 콘솔에 출력
        print(f'url: {url}')
        
        # URL의 HTML 콘텐츠를 다운로드합니다.
        downloaded = trafilatura.fetch_url(url)
        print(f'downloaded: {downloaded}')
        
        # 다운로드에 실패한 경우 None 반환
        if downloaded is None:
            return None
        
        # HTML에서 본문 텍스트만 추출합니다.
        text = trafilatura.extract(downloaded)
        print(f'text: {text}')
        
        return text
    except Exception as e:
        # 예외 발생 시 에러 메시지를 화면에 표시하고 None 반환
        st.error(f"URL 추출 중 오류 발생: {e}")
        return None

def generate_quiz(text, level, client):
    """
    OpenAI의 GPT-4o-mini 모델을 사용하여 퀴즈를 생성하는 함수입니다.
    JSON 형식으로 결과를 반환받습니다.
    """
    
    # 텍스트 길이 제한 (Input Token Limit 방지 및 비용 절약)
    # 너무 긴 텍스트는 앞부분 15000자만 사용합니다.
    truncated_text = text[:15000] 

    # 수준별 프롬프트 설정 (난이도 차별화)
    level_prompts = {
        "초등생": "초등학교 5~6학년 수준의 쉬운 어휘와 짧고 간결한 문장을 사용하세요. 이해하기 쉬운 구어체 느낌을 살짝 섞어도 좋습니다. 전문 용어는 반드시 풀어서 설명하거나 쉬운 말로 바꾸세요.",
        "중등생": "중학교 교과서 수준의 표준 어휘와 문장을 사용하세요. 논리적인 흐름을 유지하되 지나치게 추상적인 표현은 피하세요.",
        "고등생": "고등학교 비문학 독해 지문 수준으로 작성하세요. 고급 어휘와 복합 문장을 사용하여 논리적 추론 능력을 요하도록 구성하세요.",
        "성인": "대학교재나 전문 아티클 수준의 깊이 있는 문체와 전문적인 어휘를 사용하세요. 복잡한 논리 구조와 함축적 의미를 포함하여 고차원적인 독해력을 요구하세요."
    }
    
    selected_level_guide = level_prompts.get(level, level_prompts["고등생"])

    # 시스템 프롬프트: AI의 역할과 퀴즈 생성 규칙을 정의합니다.
    system_prompt = f"""
    당신은 한국어 독해 교육 전문가입니다. 
    사용자가 제공한 원문 텍스트를 바탕으로 '{level}' 독자를 대상으로 한 맞춤형 독해 퀴즈를 출제합니다.
    
    [난이도 지침]
    {selected_level_guide}
    
    [작업 절차]
    1. 먼저 원문의 내용을 대상 독자 수준('{level}')에 맞게 순화하거나 재구성하여 요약(summary)을 작성하세요.
    2. 생성된 요약문을 바탕으로, 글의 내용을 다각도로 평가할 수 있는 문제 5개를 출제하세요.
    
    [문제 유형 가이드]
    고정된 유형 없이, 지문의 특성에 맞춰 가장 적절한 문제 유형 5가지를 **동적으로 선정**하여 출제하세요.
    예시 유형 (참고용일 뿐, 이에 국한되지 않음):
    - 주제 파악, 세부 내용 일치, 추론하기, 글의 구조 파악, 비판적 읽기, 어휘의 문맥적 의미, 논지 전개 방식 등.
    - 상황에 따라 <보기>를 활용한 비교/분석 문제도 적극 활용하세요.

    [필수 규칙]
    - 질문(question) 안에 "<보기> ... </보기>" 태그를 사용하여 비교 지문이나 추가 자료를 명확히 구분하세요.
    - 결과는 반드시 JSON 형식으로만 출력하세요. 마크다운 태그(```json)를 포함하지 마세요.
    """

    # 사용자 프롬프트: 실제 분석할 텍스트와 원하는 출력 형식을 전달합니다.
    user_prompt = f"""
    다음 원문을 읽고 독해 퀴즈를 생성하세요:
    
    {truncated_text}
    
    [JSON 출력 형식]
    {{
      "summary": "난이도가 조절된 요약문",
      "questions": [
        {{
          "id": 1,
          "type": "유형 (예: 주제 파악)",
          "question": "문제 지문",
          "options": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
          "answer": 1 (정답 번호 1~5),
          "explanation": "해설"
        }}
      ]
    }}
    """

    try:
        # OpenAI API에 채팅 완료 요청을 보냅니다.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # JSON 모드를 활성화하여 항상 유효한 JSON 출력을 보장받습니다.
            response_format={"type": "json_object"},
            temperature=0.3  # 사실 기반이므로 창의성(온도)을 낮춥니다.
        )
        
        # 응답 내용(content)을 가져옵니다.
        content = response.choices[0].message.content
        
        # 혹시 모를 마크다운 코드 블록 제거 (JSON 모드에서도 가끔 포함될 수 있음)
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
            
        # JSON 문자열을 파이썬 딕셔너리로 변환하여 반환합니다.
        return json.loads(content)
    except Exception as e:
        st.error(f"문제 생성 중 오류 발생: {e}")
        return None

# --- UI Layout (화면 구성) ---

# 앱의 메인 제목 표시
st.title("📚 AI 독해력 트레이너")
st.markdown("나만의 맞춤형 독해 퀴즈로 문해력을 키워보세요!")

# 사이드바 설정 (왼쪽 메뉴)
with st.sidebar:
    st.header("설정")
    
    # API Key가 환경 변수에 없는 경우, 사용자가 직접 입력할 수 있는 필드 제공
    if not api_key:
        user_api_input = st.text_input("OpenAI API Key 입력", type="password")
        if user_api_input:
            st.session_state["OPENAI_API_KEY"] = user_api_input
            api_key = user_api_input
        else:
            st.warning("API Key가 필요합니다.")

    # 독자 수준 선택 박스 (초/중/고/성인)
    target_level = st.selectbox(
        "독자 수준 선택",
        ["초등생", "중등생", "고등생", "성인"],
        index=2 # 기본값은 '고등생'
    )
    
    st.info("💡 팁: 수준을 변경하면 문제의 난이도와 어휘가 달라집니다.")

# 메인 콘텐츠 (API Key가 있을 때만 활성화)
if api_key:
    # OpenAI 클라이언트 인스턴스 생성
    client = OpenAI(api_key=api_key)
    
    # 두 개의 탭 생성 (URL 입력용, 텍스트 직접 입력용)
    tab1, tab2 = st.tabs(["🔗 URL 입력", "📝 텍스트 직접 입력"])
    
    input_text = None
    
    # 첫 번째 탭: URL 입력
    with tab1:
        url_input = st.text_input("기사를 읽을 URL을 입력하세요")
        if url_input:
             # URL 입력 시 바로 추출을 시도하지 않고 버튼 클릭 시 수행 (UX 최적화)
             pass

    # 두 번째 탭: 텍스트 직접 입력
    with tab2:
        text_input = st.text_area("텍스트를 붙여넣으세요", height=200)

    # 문제 생성 버튼
    if st.button("🚀 문제 생성하기", type="primary"):
        # 로딩 스피너 표시
        with st.spinner("지문을 분석하고 문제를 생성 중입니다..."):
            extracted_text = ""
            
            # 입력 소스 디버깅 출력
            print(f'url_input: {url_input}')
            print(f'if: {True if url_input else False}')
            
            # URL 입력이 있고 텍스트 입력이 비어있으면 URL 우선 처리
            if url_input and text_input == '': 
                 with st.spinner("URL에서 본문 추출 중..."):
                    extracted_text = extract_text_from_url(url_input)
            # 텍스트 입력이 있으면 그 내용 사용
            elif text_input:
                extracted_text = text_input
            # 둘 다 없으면 경고 표시 후 중단
            else:
                st.warning("URL이나 텍스트 중 하나를 입력해주세요.")
                st.stop()
            
            print(f'extracted_text: {extracted_text}')

            # 텍스트가 성공적으로 준비되었는지 확인
            if extracted_text:
                if len(extracted_text) < 50:
                    st.error("텍스트가 너무 짧습니다. 더 긴 내용을 입력해주세요.")
                else:
                    # AI를 통한 퀴즈 데이터 생성
                    quiz_data = generate_quiz(extracted_text, target_level, client)
                    
                    if quiz_data:
                        # 생성된 퀴즈 데이터를 세션 상태에 저장 (새로고침 시 유지)
                        st.session_state['quiz_data'] = quiz_data
                        # 이전 답안 및 제출 상태 초기화
                        st.session_state['user_answers'] = {} 
                        st.session_state['submitted'] = False
                        st.success("문제가 생성되었습니다! 아래에서 풀어보세요.")
            else:
                st.error("텍스트를 가져오지 못했습니다. URL을 확인하거나 직접 입력해주세요.")

    # 퀴즈 풀이 화면 (퀴즈 데이터가 있을 때만 표시)
    if 'quiz_data' in st.session_state and st.session_state['quiz_data']:
        st.divider() # 구분선
        
        # 요약문 표시 (문제 풀이의 핵심 지문)
        st.subheader("📖 지문 읽기")
        st.info(st.session_state['quiz_data'].get('summary', '요약문이 없습니다.'))
        
        st.divider()
        st.subheader("📝 실전 독해 퀴즈")
        
        # 퀴즈 입력을 위한 폼 생성
        with st.form("quiz_form"):
            questions = st.session_state['quiz_data'].get('questions', [])
            
            for idx, q in enumerate(questions):
                # 문제 번호와 유형 표시
                st.markdown(f"**Q{idx+1}. [{q['type']}]**")
                
                # 문제 지문 표시 (<보기> 태그 처리)
                # replace를 통해 보기 섹션을 시각적으로 구분되게 처리
                st.write(q['question'].replace("<보기>", "\n\n> **<보기>**\n> ").replace("</보기>", "\n\n")) 
                
                # 라디오 버튼 형식이지만 선택 초기화(index=None) 상태로 시작
                # key는 각 위젯을 구분하는 고유 ID여야 함
                choice = st.radio(
                    "정답을 선택하세요:",
                    q['options'],
                    key=f"q_{idx}",
                    index=None 
                )
                
                # 사용자가 선택을 변경할 때마다 세션 상태에 답안 저장
                if choice:
                    # 선택된 문장의 인덱스를 찾아서 번호(1~5)로 변환
                    selected_index = q['options'].index(choice) + 1
                    st.session_state['user_answers'][q['id']] = selected_index
                
                st.markdown("---")
            
            # 제출 버튼 (폼 내부의 유일한 제출 트리거)
            submit_btn = st.form_submit_button("제출 및 채점")
            
            if submit_btn:
                st.session_state['submitted'] = True

        # 채점 결과 화면 (제출 되었을 때만 표시)
        if st.session_state.get('submitted', False):
            st.divider()
            st.header("📊 분석 결과")
            
            correct_count = 0
            questions = st.session_state['quiz_data']['questions']
            
            # 각 문제별 채점 진행
            for q in questions:
                user_ans = st.session_state['user_answers'].get(q['id'])
                correct_ans = q['answer']
                
                is_correct = (user_ans == correct_ans)
                if is_correct:
                    correct_count += 1
                    st.success(f"Q{q['id']} 정답! 🎉")
                else:
                    st.error(f"Q{q['id']} 오답 (선택: {user_ans}번 / 정답: {correct_ans}번)")
                
                # 해설 보기 (접이식 UI)
                with st.expander(f"Q{q['id']} 해설 보기"):
                    st.write(q['explanation'])
            
            # 오답 유형 분석
            incorrect_types = [q['type'] for q in questions if st.session_state['user_answers'].get(q['id']) != q['answer']]
            
            # 틀린 유형이 하나라도 있다면 맞춤형 조언 제공
            if incorrect_types:
                st.subheader("💡 맞춤형 학습 전략")
                strategies = {
                    "주제 찾기": "글의 첫 문단과 마지막 문단을 다시 읽으며 핵심 키워드를 찾아보세요. 반복되는 단어가 주제일 가능성이 높습니다.",
                    "어휘 선택": "단어의 사전적 의미보다 문맥 속에서의 의미를 파악하는 연습이 필요합니다. 앞뒤 문장의 흐름을 단서로 사용하세요.",
                    "빈칸 삽입": "빈칸 앞뒤의 접속사(그러나, 따라서 등)에 주목하세요. 문장의 논리적 연결(인과, 대조, 역접)을 파악해야 합니다.",
                    "내용 일치": "본문의 서술어(있다/없다, 증가했다/감소했다)를 꼼꼼히 확인하세요. 사용자의 배경지식이 아닌 '지문에 적힌 사실'만 믿어야 합니다.",
                    "비교 지문 분석": "두 지문의 공통점보다는 '차이점'에 집중하세요. 관점의 차이나 태도의 차이를 묻는 경우가 많습니다."
                }
                
                unique_incorrect_types = set(incorrect_types)
                for q_type in unique_incorrect_types:
                    # 정확히 일치하는 키가 있으면 출력, 없으면 일반적인 조언 출력
                    if q_type in strategies:
                        st.info(f"**[{q_type}]** 유형이 약하시군요. \nNOTE: {strategies[q_type]}")
                    else:
                        st.info(f"**[{q_type}]** 유형을 틀리셨네요. 해당 유형은 지문을 꼼꼼히 다시 읽고 근거를 찾는 연습이 필요합니다.")
            
            # 종합 점수 계산 및 피드백 표시
            score = (correct_count / len(questions)) * 100
            st.metric("나의 점수", f"{int(score)}점")
            
            if score == 100:
                st.balloons() # 축하 효과
                st.markdown("### 🏆 완벽합니다! 독해력이 매우 뛰어나시네요.")
            elif score >= 60:
                st.markdown("### 👍 잘하셨습니다! 틀린 문제의 해설을 꼭 확인해보세요.")
            else:
                st.markdown("### 🔥 조금 더 연습이 필요해 보입니다. 지문을 천천히 다시 읽어보세요.")

# API Key가 없는 경우의 안내 메시지 (메인 화면)
else:
    st.warning("서비스를 이용하려면 OpenAI API Key가 필요합니다.")
