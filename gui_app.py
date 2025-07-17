# main.py

import gradio as gr
from app_prompt import invoke_agent

available_models = [
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-27b-it:free"
]

with gr.Blocks(title="Telkom HC Assistant") as demo:
    gr.Markdown("### Telkom HC Assistant")
    gr.Markdown("Masukkan pertanyaan Anda .")

    output = gr.Textbox(label="Jawaban AI", lines=6, interactive=False)

    with gr.Column():
        user_input = gr.Textbox(label="Pertanyaan", placeholder="???", lines=4)
    with gr.Column():
        model_dropdown = gr.Dropdown(choices=available_models, label="Pilih Model", value=available_models[0])
        temp_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Temperature (Kreativitas)")
        submit_btn = gr.Button("Kirim", variant="primary")
 

    def run_agent(question, model, temp):
        return invoke_agent(question, model_name=model, temperature=temp)

    submit_btn.click(fn=run_agent, inputs=[user_input, model_dropdown, temp_slider], outputs=[output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
