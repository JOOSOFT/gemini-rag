import os
import uuid
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset

# ==========================================
# 1. Gemini API 설정 (작성해주신 코드 반영)
# ==========================================
# 주의: 실제 실행 시 환경변수에 API 키를 넣거나 아래 문자열을 직접 변경하세요.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "여기에_API_키를_입력하세요")
genai.configure(api_key=GEMINI_API_KEY)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# 모델 설정 (요청하신 gemini-3-flash-preview 유지)
model = genai.GenerativeModel('gemini-3-flash-preview', safety_settings=safety_settings)

# ==========================================
# 2. RAG 파이프라인 및 데이터베이스 설정
# ==========================================
DB_DIR = "./gemini-rag-korean"

def setup_rag_db():
    print("🔄 한국어 특화 임베딩 모델을 로드하는 중...")
    # 한국어 문장 임베딩에 성능이 좋은 모델 사용
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # DB가 비어있는지 확인 후, 비어있다면 Hugging Face에서 데이터를 가져와 초기 구축
    existing_docs = db.get()
    if len(existing_docs['ids']) == 0:
        print("📥 DB가 비어있습니다. Hugging Face에서 데이터셋을 다운로드하여 구축합니다...")
        try:
            # 데이터셋 로드 (train split의 일부만 가져와서 테스트 속도 향상, 필요시 조절)
            dataset = load_dataset("meal-bbang/Korean_message", split="train")
            
            # 테스트를 위해 우선 500개만 임베딩 (전체 임베딩은 시간이 오래 걸릴 수 있음)
            sample_data = dataset.select(range(500)) 
            
            texts = []
            metadatas = []
            ids = []
            
            for i, item in enumerate(sample_data):
                # 데이터셋의 컬럼명에 맞게 조정 (보통 text, label 등)
                text = item.get('text', '')
                label = item.get('label', -1)
                
                if text:
                    texts.append(text)
                    metadatas.append({"label": label, "source": "huggingface"})
                    ids.append(str(uuid.uuid4()))
            
            print(f"⏳ {len(texts)}개의 메시지를 벡터 DB에 임베딩 중입니다. (잠시만 기다려주세요...)")
            db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print("✅ 초기 데이터베이스 구축 완료!")
            
        except Exception as e:
            print(f"❌ 데이터셋 로드/임베딩 중 오류 발생: {e}")
    else:
        print(f"✅ 기존 DB를 로드했습니다. (현재 저장된 문서 수: {len(existing_docs['ids'])}개)")
        
    return db

# ==========================================
# 3. 메인 실행 루프 (검색 + Gemini 답변)
# ==========================================
def run_smishing_detector():
    db = setup_rag_db()
    
    print("\n🛡️ AI 스미싱 탐지기 시작 (종료: 'exit')")
    
    while True:
        user_query = input("\n의심되는 문자 메시지를 입력하세요: ").strip()
        
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query:
            continue
            
        print("\n🔎 1. 유사한 과거 스미싱/정상 메시지 검색 중...")
        docs = db.similarity_search(user_query, k=3)
        
        context_str = ""
        if docs:
            for i, doc in enumerate(docs):
                context_str += f"[유사 사례 {i+1}] 내용: {doc.page_content} / 라벨: {doc.metadata.get('label')}\n"
        else:
            context_str = "유사한 과거 사례를 찾을 수 없습니다."
            
        print("🧠 2. Gemini 모델을 통해 분석 중...")
        
        # 프롬프트 구성 (시스템 프롬프트 + RAG 검색 결과 + 사용자 입력)
        prompt = f"""[System Prompt]
당신은 사이버 보안 및 스미싱(문자 사기) 판별 전문가입니다.
사용자가 입력한 [Message]가 스미싱인지 아닌지 판별해야 합니다.
판별할 때 반드시 아래에 제공된 [과거 유사 데이터베이스 사례]를 참고하십시오.
답변은 1) 스미싱 위험도(안전/주의/위험), 2) 판단 이유, 3) 대처 방법 순으로 간결하게 작성해주세요.

[과거 유사 데이터베이스 사례]
{context_str}

[Message]
{user_query}
"""
        try:
            response = model.generate_content(prompt)
            text_data = response.text.strip()
            print(f"\n🤖 [Gemini 분석 결과]\n{text_data}") 
            
            # 입력받은 새 메시지를 DB에 학습(추가)시키기
            doc_id = str(uuid.uuid4())
            db.add_texts(
                texts=[user_query],
                metadatas=[{"label": "user_input", "source": "realtime_input"}],
                ids=[doc_id]
            )
            print("\n💾 (이 메시지는 향후 분석을 위해 벡터 DB에 실시간 업데이트 되었습니다.)")
            
        except Exception as e:
            print(f"\n❌ 에러: {e}")
            print("잠시 후 다시 시도해주세요.")

if __name__ == "__main__":
    run_smishing_detector()
