import os
from google import genai
from google.genai import types

class IntentionClassifier:
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash-exp", 
        api_key: str = None
    ):
        print(f'>> Establishing Intention Classifier ...')
        self.model_id = model_id
        self.client = genai.Client(api_key=api_key)

        print(f'>> Intention Classifier has been established successfully.')
        print('--- Model Details ---')
        print(f'Model ID: {self.model_id}')
        print()
    
    def classify(self, user_input: str) -> str:
        prompt = f"""You are an Intent Classifier for a Public Health Support Bot in Vietnam.

Classification Rules:
- general: Greetings, small talk, gratitude, identity questions. (NO database search).
- specific: Public health questions, procedures, laws, medical practice, documents. (REQUIRES database search).

Few-Shot Examples:
Input: "Xin chào, bạn có khỏe không?"
Class: general

Input: "Làm thế nào để xin giấy phép mở nhà thuốc?"
Class: specific

Input: "Nghị định 109 có còn hiệu lực không?"
Class: specific

Input: "Alo admin ơi."
Class: general

Input: "Thủ tục hành chính."
Class: specific

Current Task:
Input: "{user_input}"
Class:"""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10
                )
            )
            result_text = response.text.strip().lower()
            if "general" in result_text:
                return "general"
            elif "specific" in result_text:
                return "specific"
            else:
                # print(f">> Warning: Unclear intent '{result_text}'. Defaulting to specific.")
                return "specific"

        except Exception as e:
            # print(f">> Error in classification: {e}")
            return "specific"