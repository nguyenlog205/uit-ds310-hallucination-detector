from google import genai


class IntentionClassifier():
    def __init__(
        self,
        model_id = "gemini-2.5-flash"
    ):
        print(f'>> Establishing Intention Classifier ...')
        self.model_id = model_id
        self.client = genai.Client()

        print(f'>> Intention Classifier has been established successfully.')
        print('--- Model Details ---')
        print(f'Model ID: {self.model_id}')
        print()
    
    def classify(
        self,
        user_input: str
    ):
        content = f'''
        You are an Intent Classifier for a Public Health Support Bot in Vietnam.
        Your goal is to determine if the user requires information from the external database or is simply engaging in social conversation.

        **Classification Rules:**
        - **general**: Inputs that are greetings, farewells, expressions of gratitude, compliments, or questions about the bot's identity. (NO database search needed).
        - **specific**: ANY question related to public health, administrative procedures, laws, documents, or specific medical practice inquiries, regardless of how short or simple the question is. (REQUIRES database search).

        **Few-Shot Examples:**

        Input: "Xin chào, bạn có khỏe không?"
        Class: general

        Input: "Làm thế nào để xin giấy phép mở nhà thuốc?"
        Class: specific

        Input: "Cảm ơn bạn nhiều nha."
        Class: general

        Input: "Nghị định 109 có còn hiệu lực không?"
        Class: specific

        Input: "Bạn là AI hay người thật?"
        Class: general

        Input: "Thủ tục hành chính."
        Class: specific

        Input: "Alo admin ơi."
        Class: general

        Input: "Cho tôi hỏi về quy trình cấp chứng chỉ hành nghề."
        Class: specific

        **Current Task:**
        Input: "{user_input}"
        Class:
        '''

        intention_result = self.client.models.generate_content(
            model=self.model_id,
            contents=content
        )

        if intention_result in ["general", "specific"]:
            return intention_result
        else:
            return "specific"