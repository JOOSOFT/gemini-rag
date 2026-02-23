import os
import uuid
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset

# ==========================================
# 1. Gemini API 설정
# ==========================================
# 터미널에서 export GEMINI_API_KEY="실제키" 를 하거나, 아래 "YOUR_API_KEY_HERE"를 실제 키로 변경하세요.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel('gemini-3-flash-preview', safety_settings=safety_settings)

# ==========================================
# 2. RAG 파이프라인 및 데이터베이스 설정
# ==========================================
DB_DIR = "./gemini-rag-korean"

def setup_rag_db():
    print("🔄 한국어 특화 임베딩 모델을 로드하는 중...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    existing_docs = db.get()
    if len(existing_docs['ids']) == 0:
        print("📥 DB가 비어있습니다. Hugging Face에서 데이터셋을 가져옵니다...")
        try:
            dataset = load_dataset("meal-bbang/Korean_message", split="train")
            sample_data = dataset.select(range(500)) # 우선 500개만 테스트 (필요시 조절)
            
            texts = []
            metadatas = []
            ids = []
            
            # 알려주신 컬럼명(content, class) 구조를 정확히 반영
            for item in sample_data:
                text = item.get('content', '').strip()
                label_class = item.get('class', -1)
                
                if text:
                    texts.append(text)
                    
                    # class가 2인 경우에만 스미싱으로 명시적 라벨링
                    label_str = "🚨스미싱" if label_class == 2 else "✅정상"
                    
                    metadatas.append({
                        "label": label_str, 
                        "class_code": label_class,
                        "source": "huggingface"
                    })
                    ids.append(str(uuid.uuid4()))
            
            print(f"⏳ {len(texts)}개의 메시지를 벡터 DB에 임베딩 중입니다. (잠시만 기다려주세요...)")
            db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print("✅ 초기 데이터베이스 구축 완료!")
            
        except Exception as e:
            print(f"❌ 데이터 로드/임베딩 오류: {e}")
            return None
    else:
        print(f"✅ 기존 DB를 로드했습니다. (현재 저장된 문서 수: {len(existing_docs['ids'])}개)")
        
    return db

# ==========================================
# 3. 메인 실행 루프
# ==========================================
def run_smishing_detector():
    db = setup_rag_db()
    if db is None:
        print("프로그램을 종료합니다. DB 구축 에러를 확인해주세요.")
        return

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
                context_str += f"[유사 사례 {i+1}] 내용: {doc.page_content} / 판정: {doc.metadata.get('label')}\n"
        else:
            context_str = "유사한 과거 사례를 찾을 수 없습니다."
            
        print("🧠 2. Gemini 모델을 통해 분석 중...")
        
        prompt = f"""[System Prompt]
당신은 사이버 보안 및 스미싱(문자 사기) 판별 전문가입니다.
사용자가 입력한 [Message]가 스미싱인지 아닌지 판별해야 합니다.
판별할 때 반드시 아래에 제공된 [과거 유사 데이터베이스 사례]를 참고하십시오.
(참고로 유사 사례의 판정이 '🚨스미싱'인 문구와 형태가 비슷할수록 스미싱일 확률이 높습니다.)

답변은 다음 순서로 간결하게 작성해주세요:
1) 스미싱 위험도 (안전 / 주의 / 위험)
2) 판단 이유 (유사 사례를 어떻게 참고했는지 포함)
3) 대처 방법

[과거 유사 데이터베이스 사례]
{context_str}

[Message]
{user_query}
"""
        try:
            response = model.generate_content(prompt)
            print(f"\n🤖 [Gemini 분석 결과]\n{response.text.strip()}") 
            
            # 입력받은 새 메시지를 DB에 추가
            db.add_texts(
                texts=[user_query],
                metadatas=[{"label": "❓사용자입력(판별대기)", "class_code": -1, "source": "realtime"}],
                ids=[str(uuid.uuid4())]
            )
            print("\n💾 (이 메시지는 향후 분석을 위해 벡터 DB에 실시간 업데이트 되었습니다.)")
            
        except Exception as e:
            print(f"\n❌ API 에러: {e}")

if __name__ == "__main__":
    run_smishing_detector()
