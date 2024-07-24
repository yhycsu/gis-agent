import argparse
import sys

from flask import Flask
from langchain_openai import ChatOpenAI
import openai

from llm import *
from utils import call_ollama_api


# some config about large language models
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model', default='gpt4', required=True)
arg_parser.add_argument('--openai_api_key', required=False)
args = arg_parser.parse_args()

if args.model == 'gpt4':
    # check if the user provides api key for gpt4
    if args.openai_api_key is None:
        print('Please provide an OpenAI API key')
        sys.exit(1)
    # check if the api key is valid
    try:
        # initialize gpt4
        llm = ChatOpenAI(
            model="gpt-4o",
            base_url="https://api.chatanywhere.tech",
            api_key=args.openai_api_key,
            temperature=0,
        )
        llm.invoke('hello')
    except openai.AuthenticationError as e:
        print('wrong api key, please provide a correct OpenAI api key')
        sys.exit(1)
    print('gpt4 is initialized')
else:  # if the user uses another llm from ollama
    # initialized the specified llm
    try:
        initialize_data = {
            "model": args.model,
            "keep_alive": -1,
            "prompt": "",
            "stream": False
        }
        call_ollama_api(initialize_data)
    except ValueError as e:
        sys.exit(1)
    print(f'{args.model} is initialized')

app = Flask(__name__)

# initialize nexusraven
try:
    initialize_data = {
        "model": "nexusraven:gis-agent",
        "keep_alive": -1,
        "prompt": "",
        "stream": False
    }
    call_ollama_api(initialize_data)
except ValueError as e:
    sys.exit(1)
print('nexusraven is initialized')


@app.route('/chat/<user_query>')
def chat(user_query):
    """
    input: user query
    output: llm's task plan and corresponding function calling
    """
    user_query = user_query.replace('+', '/')

    # first select potential tools
    prompt = construct_prompt_for_tool_selection(user_query)
    if args.model == 'gpt4':
        response = llm.invoke(prompt)
        response = response.content
    else:
        data = {
            "model": args.model,
            "prompt": prompt,
            "stream": False
        }
        response = call_ollama_api(data)
        response = response['response']

    # handle gpt4 parse error
    try:
        tools, task_plan = get_tools_and_task_plan(response)
    except Exception as e:
        return {'code': 1, 'error': 'error parsing tools and task plan', 'message': str(e)}

    # construct prompt，then perform function calling
    prompt = construct_prompt(user_query=user_query, task_plan=task_plan, functions=tools)
    data = {
        "model": "nexusraven:gis-agent",
        "prompt": prompt,
        "stream": False
    }
    # handle nexusraven parse error
    try:
        response = call_ollama_api(data)
        if isinstance(response, dict):
            response = response['response'].replace('<bot_end>', '')
        start = response.find('Call: ')
        end = response.find('Thought: ')
        call = response[start + 6:end].replace('\n', '')
        thought = response[end:]
        return {'code': 0, 'call': call, 'thought': thought}
    except Exception as e:
        return {'code': 1, 'error': 'error constructing function calls', 'message': str(e)}


if __name__ == '__main__':
    app.run(debug=False)
