import os
from dotenv import load_dotenv
import gradio as gr
from chat import Chat

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
def inference(message, *args):
    return main(message)

def main(text):
    os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
    chat = Chat()
    response_chat = chat.ask(text)

    #저장된 메세지 출력하는 부분
    all_messages = chat.memory_buffer.get_all_messages()
    print(all_messages)

    return response_chat

demo = gr.ChatInterface(fn=inference,
                        textbox=gr.Textbox(placeholder="Message ChatBot..", container=False, scale=5),
                        chatbot=gr.Chatbot(height=700),
                        title="PDF Chat Bot",
                        description="PDF 챗봇입니다.",
                        theme="soft",
                        retry_btn="다시보내기",
                        undo_btn="이전챗 삭제",
                        clear_btn="전챗 삭제")

if __name__ == "__main__":
    demo.launch(debug=True, share=True)

