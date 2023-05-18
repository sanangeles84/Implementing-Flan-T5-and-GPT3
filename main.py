
import gradio as gr
import openai

import os

openai.api_key = os.getenv("OPENAI_API_KEY")

#from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Flan-T5
#model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
#tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def generate_text_flan(input_string, max_length):
  inputs = tokenizer(input_string, return_tensors="pt")
  outputs = model.generate(**inputs, max_length=max_length)
  final_text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

  return (final_text)


def generate_text_gpt(input_string, max_length):
  response = openai.Completion.create(model="text-davinci-003",
                                      prompt=input_string,
                                      temperature=0,
                                      max_tokens=max_length,
                                      top_p=1,
                                      frequency_penalty=0,
                                      presence_penalty=0)
  answer = response.choices[0]['text']
  return (answer)


def to_gradio():
  demo = gr.Interface(fn=generate_text_gpt,
                      inputs=["text", gr.Slider(0, 250)],
                      outputs="text")
  demo.launch(debug=True, share=True)

if __name__ == "__main__":
  to_gradio()