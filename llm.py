import inspect
import re
import sys
import json
import requests
from urllib.parse import quote

import ollama

from toolbox import *
from utils import *

api_endpoint = "http://localhost:11434/api/generate"

def format_functions_for_prompt(*functions):
    formatted_functions = []
    for func in functions:
        source_code = inspect.getsource(func)
        docstring = inspect.getdoc(func)
        formatted_functions.append(
            f"OPTION:\n<func_start>{source_code}<func_end>\n<docstring_start>\n{docstring}\n<docstring_end>"
            #f"OPTION:\n<docstring_start>\n{docstring}\n<docstring_end>"
        )
    return "\n".join(formatted_functions)


def construct_prompt(user_query: str, task_plan, functions):
    formatted_prompt = format_functions_for_prompt(*functions)
    formatted_prompt += f"\n\nUser Query: {user_query}\n"

    prompt = (
            "<human>:\n"
            + formatted_prompt
            + f"according to the task plan: {task_plan}\n"
            + "Please follow the task plan and pick corresponding functions from the above options that best answers the user query and fill in the appropriate "
              "arguments. Your answer should only contrain one call.<human_end>"
    )
    return prompt


def get_tools_and_task_plan(s):
    """
    提取大模型回答中的task plan 和 tools

    参数:
    s (str): 输入字符串

    返回:
    str: 第一个中括号内的内容，如果没有中括号则返回空字符串
    """
    tools = re.search(r'(?<=\[)[^\[\]]*(?=\])', s, re.DOTALL)[0]
    task_plan = re.search(r'(?<=<task plan>)(.*?)(?=</task plan>)', s, re.DOTALL)[0]
    tools = tools.replace(' ', '').replace('\n', '').split(',')
    tools = list(map(lambda x: eval(x), tools))

    if tools and task_plan:
        return tools, task_plan
    

def construct_prompt_for_tool_selection(user_query):
    prompt = f'''
    Your role: A professional Geo-information scientist and programmer good at Python. 
    You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. 
    You know well how to set up workflows for spatial analysis tasks. You have significant experence on how to plan. 
    You are also very experienced in using python and geopandas.

    Your task: Generate a specific task plan to solve the problem based on user instructions and requirements, and list the tools needed for each step.

    The available tools are:
    read_shp_file: Read a shapefile, only use this when a shapefile in provided in the user input
    create_buffer: create a buffer for a shapefile
    compute_difference: compute the difference between two shapefiles
    get_path: get the path from a start location to a end location, need to know their latitude and longitude
    get_poi: get many locations with a keyword, not suited for finding a  single location, useful when you want to find the locations and the user does not provide any shapefiles
    clip_shp: clip the shapefile
    union_features: compute the union of two shapefiles, if you want to combine two GeoDataFrames into one as a result
    compute_convex_hull: compute the convex hull of a shapefile
    dissolve_features: dissolve the features of two shapefiles
    intersection_features: compute the intersection of two shapefiles
    symmetrical_difference_features: The negated intersection of the two geodataframes
    count_points_in_polygons: count all the points in a polygon
    generate_centroids: Generates a point layer that contains the centroid of the geometry in the input layer
    calculate_line_length_in_polygons: calculate the length of lines in a polygon
    nearest_neighbor_analysis: Performs nearest neighbor analysis on a point layer
    filter: filter the shapefile that meets a certain condition
    get_latitude_and_longitude_of_a_location: get the latitude and longitude of a location
    get_current_latitude_and_longitude: get the current latitude and longitude
    calculate_the_area_of_polygon: using GeoPandas to calculate the area of a shapefile, or the result of other tools

    Your reply needs to meet these requirements: 
    1. Think step by step.
    2. You should select at most 8 tools that you think is helpful to solve the problem.
    3. return the tools in a list
    4. Your task plan should start with <task plan> and end with </task plan>
    
    
    Below are some examples that show how you should answer the question:

    ### examples ###
    Question:
    create buffer for the union of a.shp and b.shp
    Answer:
    tools: [read_shp_file, create_buffer, union_features]
    ###

    Here is the user's instruction and requirement:
    
    ### user instruction ### 
    {user_query}
    ###
    
    Think step by step.
    Generate a specific task plan to solve the problem based on user instructions and requirements, and list the tools needed for each step.
    '''
    return prompt


def select_tools_for_task(user_query):
    data = {
        "model": "llama3:gis-agent",
        "prompt": construct_prompt_for_tool_selection(user_query=user_query),
        "stream": False
    }
    llm_output = call_ollama_api(api_endpoint, data)
    if isinstance(llm_output, dict):
        llm_output = llm_output['response']
    tools = get_tools_and_task_plan(llm_output)
    tools = tools.replace(' ', '').split(',')
    tools = list(map(lambda x: eval(x), tools))
    return tools


def get_llm_output(user_query):
    # 将用户输入的查询语句中的斜杠替换为下划线，以便在URL中使用
    # 并且将空格编码
    user_query = user_query.replace('/', '+')
    user_query = quote(user_query)
    url = f"http://127.0.0.1:5000/chat/{user_query}"
    response = requests.get(url)

    # 检查响应状态码
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(f'请求失败，状态码: {response.status_code}')
        return None
    


if __name__ == "__main__":
    from langchain_community.llms import Ollama

    # function_calling_llm = Ollama(model='nexusraven:latest')
    # tool_filtering_llm = Ollama(model='llama3:latest')
    # prompt = '''
    # please filter "shapefiles/poi/中南大学校本部_长沙市/中南大学校本部_长沙市.shp" that is in the buffer of "shapefiles/results/buffer.shp", temp_dir为""
    # '''

    # prompt = '''
    # please filter out all locations with keyword "地铁站" that is in the buffer of the path from (112.93158,28.14308) to (112.94366,28.16827), file_dir is "", buffer size is 0.001
    # '''

    # prompt = '''
    #     Is there any "地铁站" within the path from (112.93158,28.14308) to (112.94366,28.16827)?
    # '''
    # prompt = '''
    # please filter out all locations with keyword "地铁站" that is inside the 'city.shp'
    # # '''
    # print(construct_prompt(prompt))
    #print(get_llm_output(prompt))

    # print(select_tools_for_task(
    #     'please filter out all locations with keyword "地铁站" that is in the buffer of the path from (112.93158,28.14308) to (112.94366,28.16827), file_dir is "", buffer size is 0.001'))
    # print(get_llm_output(
    #     'please filter out all locations with keyword "地铁站" that is in the buffer of the path from (112.93158,28.14308) to (112.94366,28.16827), file_dir is "", buffer size is 0.001'))
    #print(get_llm_output('compute the union of a.shp and the buffer of b.shp, buffer size is 0.0001'))
    s = """ 

    <task plan>

    1. **Get the current location**:
    - Use `get_current_latitude_and_longitude` to get the latitude and longitude of the current location.
    
    2. **Get the latitude and longitude of "长沙世界之窗"**:
    - Use `get_latitude_and_longitude_of_a_location` to get the latitude and longitude of "长沙世界之窗".
    
    3. **Get the path from the current location to "长沙世界之窗"**:
    - Use `get_path` with the current location and the destination coordinates to get the path.
    
    4. **Get the locations of subways**:
    - Use `get_poi` with the keyword "subway" to get the locations of subways in the area.
    
    5. **Create a buffer around the path**:
    - Use `create_buffer` to create a 1-kilometer buffer around the path.
    
    6. **Clip the subway locations with the buffer**:
    - Use `clip_shp` to clip the subway locations with the buffer to find subways within 1 kilometer of the path.
    </task plan>
    ### Tools:

    ```python
    tools = [
        get_current_latitude_and_longitude,
        get_latitude_and_longitude_of_a_location,
        get_path,
        get_poi,
        create_buffer,
        clip_shp
    ]
    """
    tools, task_plan = get_tools_and_task_plan(s)
    print(tools, task_plan)
