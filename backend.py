from flask import Flask
from transformers import pipeline
import time

from langchain_openai import ChatOpenAI

from llm import *
from utils import call_ollama_api


app = Flask(__name__)

# initialize ollama
api_endpoint = "http://localhost:11434/api/generate"
initialize_data = {
    "model": "nexusraven:13b-v2-fp16",
    "keep_alive": -1,
    "prompt": "",
    "stream": False
}
call_ollama_api(api_endpoint, initialize_data)

# pipeline = pipeline(
#     "text-generation",
#     model="/home/a6000/huggingface/NexusRaven-V2-13B",
#     torch_dtype="auto",
#     device_map="auto",
# )

gpt4o_model = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.chatanywhere.tech",
    api_key="sk-aJCHfcIwEyjPaksvqkoJziuUMw9QsPJO0N4DFGatJRtCFBTS-ca",
    temperature=0.0,
)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/chat/<user_query>')
def chat(user_query):
    user_query = user_query.replace('+', '/')
    
    # 首先筛选可能需要的工具
    prompt = construct_prompt_for_tool_selection(user_query)
    now = time.time()
    response = gpt4o_model.invoke(prompt)
    print('gpt4: ', time.time() - now)
    tools, task_plan = get_tools_and_task_plan(response.content)

    # 然后构造prompt，进行function call
    prompt = construct_prompt(user_query=user_query, task_plan=task_plan, functions=tools)
    # response = pipeline(prompt, max_new_tokens=2048, return_full_text=False, do_sample=True, temperature=0.001, use_cache=True)[0]["generated_text"]
    data = {
        "model": "nexusraven:gis-agent",
        "prompt": prompt,
        "stream": False
    }
    now = time.time()
    response = call_ollama_api(api_endpoint, data)
    print('nexusraven: ', time.time() - now)
    if isinstance(response, dict):
        response = response['response']
    start = response.find('Call: ')
    end = response.find('Thought: ')
    call = response[start + 6:end].replace('\n', '')
    thought = response[end:]
    return {'call': call, 'thought': thought}


if __name__ == '__main__':
    app.run(debug=False)
