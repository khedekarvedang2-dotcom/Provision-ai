import gradio as gr
import joblib

model = joblib.load("model.pkl")

responses = {
    "positive": "🔥 Positive",
    "negative": "🌧 Negative",
    "tech": "🤖 Tech",
    "cars": "🏎 Cars"
}

def predict(text):
    result = model.predict([text])[0]
    return responses.get(result, result)

app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    title="PocketVision AI"
)

app.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    inbrowser=True
)