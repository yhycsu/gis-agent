import inspect
import re
from urllib.parse import quote

from toolbox import *


def format_functions_for_prompt(*functions):
    """
    Formats functions for prompt with their source code and docstring
    """
    formatted_functions = []
    for func in functions:
        source_code = inspect.getsource(func)
        docstring = inspect.getdoc(func)
        formatted_functions.append(
            f"OPTION:\n<func_start>{source_code}<func_end>\n<docstring_start>\n{docstring}\n<docstring_end>"
        )
    return "\n".join(formatted_functions)


def construct_prompt(user_query: str, task_plan, functions):
    """
    Constructs prompt to generate function calls with selected tools
    """
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


def get_tools_and_task_plan(response):
    """
    Extract task plan and tools from large language model answers

    param:
    response (str): the response from the LLM

    return:
    The content within the first square bracket, if there is no square bracket, None is returned.
    """
    tools = re.search(r'(?<=\[)[^\[\]]*(?=])', response, re.DOTALL)[0]
    task_plan = re.search(r'(?<=<task plan>)(.*?)(?=</task plan>)', response, re.DOTALL)[0]
    tools = tools.replace(' ', '').replace('\n', '').split(',')
    tools = list(map(lambda x: eval(x), tools))

    if tools and task_plan:
        return tools, task_plan
    return None, None


def construct_prompt_for_tool_selection(user_query):
    prompt = f''' Your role: A professional Geo-information scientist and programmer good at Python. You have worked 
    on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial 
    data and coding. You know well how to set up workflows for spatial analysis tasks. You have significant experence 
    on how to plan. You are also very experienced in using python and geopandas.

    Your task: Generate a specific task plan to solve the problem based on user instructions and requirements, 
    and list the tools needed for each step.

    The available tools are: read_shp_file: Read a shapefile, only use this when a shapefile in provided in the user 
    input create_buffer: create a buffer for a shapefile compute_difference: compute the difference between two 
    shapefiles get_path: get the path from a start location to a end location, need to know their latitude and 
    longitude get_poi: get many locations with a keyword, not suited for finding a  single location, useful when you 
    want to find the locations and the user does not provide any shapefiles clip_shp: clip the shapefile 
    union_features: compute the union of two shapefiles, if you want to combine two GeoDataFrames into one as a 
    result compute_convex_hull: compute the convex hull of a shapefile dissolve_features: dissolve the features of 
    two shapefiles intersection_features: compute the intersection of two shapefiles symmetrical_difference_features: 
    The negated intersection of the two geodataframes count_points_in_polygons: count all the points in a polygon 
    generate_centroids: Generates a point layer that contains the centroid of the geometry in the input layer 
    calculate_line_length_in_polygons: calculate the length of lines in a polygon nearest_neighbor_analysis: Performs 
    nearest neighbor analysis on a point layer filter: filter the shapefile that meets a certain condition 
    get_latitude_and_longitude_of_a_location: get the latitude and longitude of a location 
    get_current_latitude_and_longitude: get the current latitude and longitude calculate_the_area_of_polygon: using 
    GeoPandas to calculate the area of a shapefile, or the result of other tools

    Your reply needs to meet these requirements: 
    1. Think step by step.
    2. You should select at most 8 tools that you think is helpful to solve the problem.
    3. return the tools between tow brackets [].
    4. Your task plan should start with <task plan> and end with </task plan>


    Below are some examples that show how you should answer the question:

    ### examples ###
    Question:
    create buffer for the union of a.shp and b.shp
    Answer:
    <task plan>
    1. read shp file 'a.shp'
    2. read shp file 'b.shp'
    3. compute the unison of 'a.shp' and 'b.shp'
    4. compute the buffer of their union
    </task plan>
    <tools>
    [read_shp_file, create_buffer, union_features]
    </tools>
    ###

    Here is the user's instruction and requirement:

    ### user instruction ### 
    {user_query}
    ###

    Think step by step. Generate a specific task plan to solve the problem based on user instructions and 
    requirements, and list the tools needed for each step.'''
    return prompt


def get_llm_output(user_query):
    # Replace slashes in the input user's query with underscores for use in URLs
    # and encode spaces
    user_query = user_query.replace('/', '+')
    user_query = quote(user_query)
    url = f"http://127.0.0.1:5000/chat/{user_query}"

    try:
        response = requests.get(url)
    # error connecting to backend
    except requests.exceptions.ConnectionError as e:
        return {'code': 1, 'error': 'failed to connect to backend', 'message': str(e)}

    # Check response status code
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return {'code': 1, 'error': 'request failed', 'message': ""}
