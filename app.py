import os
from flask import Flask, request, make_response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import openai



load_dotenv()

UPLOAD_FOLDER = "./data"

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
openai.api_key = os.getenv("OPENAI_API_KEy")


def chatgpt(query: str, user, industry, c_size, c_title, type):
    if type == "0":
        print("isCalling")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """
                        In this exercise, you will be playing the role of a prospective customer interested in our SaaS products. Your name is Alex and your details are as follows:
                        User:""" + user +
                        "Industry:" + industry +
                        "Company Size:" + c_size +
                        "Title:" + c_title +
                        """You, Alex, are encouraged to generate additional details about your company, its product lines, target customers, and pain points as needed to make the conversation more realistic and dynamic.
                        The conversation will be in a typical sales call format. You have shown initial interest in our product, but you're not fully convinced yet. You will have questions and concerns about its functionality, pricing, implementation, and support. The goal of this exercise is to simulate a realistic customer conversation, allowing the Account Executive, [User's Name], to practice handling objections and articulating value propositions.
                        At the end of this exercise, you will be asked to provide feedback and coaching on the Account Executive's performance during the call. Until then, please focus on responding in a manner consistent with your given role.
                        """,
                },
                {"role": "user", "content": query},
            ],
        )
        ans_docqa = completion.choices[0].message.content
        return ans_docqa
    else:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """
                        Alex, thank you for participating in the sales call simulation. As part of your role, we're now asking you to provide feedback and coaching opportunities based on the call. Please consider the following areas:
Understanding Customer Needs and Building Rapport: Did the Account Executive, """ + user + """, ask relevant questions to fully understand your company's needs and challenges? Did they effectively build a rapport and establish a relationship with you during the conversation?
Product Knowledge: How effectively did """ + user + """ demonstrate knowledge about the product's features and benefits?
Handling Objections: How well did """ + user + """ address your concerns and objections throughout the call?
Communication Skills: How clear, concise, and persuasive was """ + user + """ in their communication? Did they exhibit skills conducive to building rapport, such as active listening and empathy?
Closing the Deal/Setting Next Steps: How well did """ + user + """ work towards a close or establish clear next steps in the sales process?
Professionalism: Did """ + user + """ maintain a professional demeanor throughout the call?
                        """,
                },
                {"role": "user", "content": ""},
            ],
        )
        ans_docqa = completion.choices[0].message.content
        return ans_docqa

def allowed_file(filename, ALLOWED_EXTENSIONS):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/chat1", methods=["POST"])
def chat1():
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

@app.route("/chat2", methods=["POST"])
def chat2():
    query = request.form["prompt"]
    user = request.form["user"]
    industry = request.form["industry"]
    c_size = request.form["c_size"]
    c_title = request.form["c_title"]
    type = request.form["type"]
    result = chatgpt(query, user, industry, c_size, c_title, type)

    return {"state": "success", "answer": result}
if __name__ == "__main__":
    app.run(debug=True)
