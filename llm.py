import inspect
import ollama

from toolbox import *


def format_functions_for_prompt(*functions):
    formatted_functions = []
    for func in functions:
        source_code = inspect.getsource(func)
        docstring = inspect.getdoc(func)
        formatted_functions.append(
            f"OPTION:\n<func_start>{source_code}<func_end>\n<docstring_start>\n{docstring}\n<docstring_end>"
        )
    return "\n".join(formatted_functions)


def construct_prompt(user_query: str):
    formatted_prompt = format_functions_for_prompt(read_shp_file, create_buffer, compute_difference,
                                                   filter, get_path, get_poi)
    formatted_prompt += f"\n\nUser Query: Question: {user_query}\n"

    prompt = (
        "<human>:\n"
        + formatted_prompt
        + "Please pick a function from the above options that best answers the user query and fill in the appropriate "
          "arguments.<human_end>"
    )
    return prompt


def get_llm_output(input_text):
    response = ollama.generate(
        model='nexusraven:latest',
        prompt=construct_prompt(input_text)
    )
    response = response['response']
    start = response.find('Call: ')
    end = response.find('Thought: ')
    call = response[start + 6:end].replace('\n', '')
    thought = response[end:]
    return {'call': call, 'thought': thought}


if __name__ == "__main__":
    # prompt = '''
    # please filter "shapefiles/poi/中南大学校本部_长沙市/中南大学校本部_长沙市.shp" that is in the buffer of "shapefiles/results/buffer.shp", temp_dir为""
    # '''

    prompt = '''
    please filter out all locations with keyword "garden" that is in the buffer of the path from (122,30) to (133,60), file_dir is "", buffer size is 0.001
    '''
    print(get_llm_output(prompt))
