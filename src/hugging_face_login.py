from huggingface_hub import login
from dotenv import load_dotenv
import os
# Load environment
load_dotenv()

def hugging_face_login():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    login(token=hf_token)


# def openai_login()
#     openai_token = os.getenv("OPENAI_API_KEY")
#     login(token=openai_token)
if __name__ == '__main__':
    pass
