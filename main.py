import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
genai.configure(api_key="GEMINI_API_KEY")
# 모델 설정 (사용자가 지정한 gemini-3 유지)

# --- 안전 설정 ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel('gemini-3-flash-preview', safety_settings=safety_settings)

prompt = f"""[System Prompt]

        [Message]"""
try:
    response = model.generate_content(prompt)
    text_data = response.text.strip()
    print(f"🤖 Gemini 응답: {text_data}") 

except Exception as e:
    print(f"❌ 에러: {e}")
    score = 50
    answer_str = "잠시 후 다시 시도해주세요."
