import gradio as gr

with gr.Blocks(title="Evaluating many recordings") as demo:
    gr.Markdown("# Evaluating many recordings")

if __name__ == "__main__":
    demo.launch()
