from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

def interactive_rag_loop():
    print("🔍 실시간 RAG DB 임베딩 및 검색 루프 시작...")
    print("💡 팁: 프로그램을 종료하려면 'exit' 또는 'quit'을 입력하세요.\n")
    
    # 1. 임베딩 모델 로드 (루프 밖에서 한 번만 로드하여 속도 최적화)
    print("🔄 임베딩 모델을 로드하는 중...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Chroma DB 연결
    persist_dir = "/home/user/ai/gemini-rag"
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # 3. 무한 루프 시작
    while True:
        # 사용자로부터 새 문장 입력 받기
        query = input("\n📝 새 문장을 입력하세요: ").strip()
        
        # 종료 조건
        if query.lower() in ['exit', 'quit']:
            print("🛑 프로그램을 종료합니다.")
            break
        
        if not query:
            print("⚠️ 빈 문장입니다. 다시 입력해 주세요.")
            continue
            
        # 4. 가장 가까운 k개의 문장 검색 (새 문장을 DB에 넣기 전 검색)
        # 참고: DB가 완전히 비어있는 초기 상태일 경우를 대비해 예외 처리를 합니다.
        print("\n🔎 가장 유사한 문장을 찾고 있습니다...")
        try:
            docs = db.similarity_search(query, k=3)
            
            if docs:
                print(f"✅ 찾은 관련 문서: {len(docs)}개\n")
                for i, doc in enumerate(docs):
                    print(f"--- [문서 {i+1}] ---")
                    print(f"내용: {doc.page_content}")
                    
                    # 새 문장은 'user_input' 라벨을 가짐 (기존 스미싱 라벨과 구분)
                    label = doc.metadata.get('label', '알 수 없음')
                    print(f"라벨: {label}")
                    print("-------------------\n")
            else:
                print("⚠️ 아직 DB에 비교할 문서가 없습니다.\n")
                
        except Exception as e:
            print(f"⚠️ 검색 중 오류 발생 (DB가 비어있을 수 있습니다): {e}\n")

        # 5. 새 문장을 DB에 자동으로 임베딩 및 추가
        print("💾 방금 입력한 문장을 DB에 추가하고 있습니다...")
        doc_id = str(uuid.uuid4()) # 고유 ID 생성
        
        db.add_texts(
            texts=[query],
            metadatas=[{"label": "user_input"}], # 새로 추가된 문장임을 표시하는 라벨
            ids=[doc_id]
        )
        print("✨ DB 업데이트 완료! (이제 다음 검색부터 이 문장도 비교 대상이 됩니다.)")

if __name__ == "__main__":
    interactive_rag_loop()
