import os
from flask import Flask, request, make_response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import whisper
import openai
import moviepy.editor as mp


from datetime import datetime

import pickle

load_dotenv()

UPLOAD_FOLDER = "./data"

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
openai.api_key = os.getenv("OPENAI_API_KEy")


def chatgpt(query: str):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a chatbot that has fun conversations with users. It should be fun and humorous.",
            },
            {"role": "user", "content": query},
        ],
    )
    ans_docqa = completion.choices[0].message.content
    return ans_docqa

def allowed_file(filename, ALLOWED_EXTENSIONS):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/chat", methods=["POST"])
def chat():
    if "file" not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files["file"]
    
    if file.filename == "":
        return {"state": "error", "message": "No selected file"}
    if file and allowed_file(file.filename, {"mp3", "wav"}):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        audio_file = open(os.path.join(app.config["UPLOAD_FOLDER"], filename), "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        texts = transcript.text

        result = chatgpt(texts)

        return {"state": "success", "answer": result}
    return {"state": "error", "message": "Invalid file format"}

if __name__ == "__main__":
    app.run(debug=True)
