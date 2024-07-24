import gradio as gr
import shutil
import time

from llm import *
from utils import *

# places to store temporary files
result_dir = 'shapefiles/results'
file_dir = 'files'
# map tiles
tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'


def move_files(input_files):
    """
    move the user uploaded files to one directory
    """
    moved_files = []
    for idx, file in enumerate(input_files):
        file_name = file.split('/')[-1]
        new_file_name = os.path.join(file_dir, file_name)
        moved_files.append(new_file_name)
        shutil.copy(file.name, new_file_name)
    return moved_files


def choose_example(example):
    return EXAMPLE_QUERIES[example]


def chat(input_files, input_text):
    # clear the function call graph
    clear_graph()
    # get llm output
    input_text += f' the file directory is {file_dir}'
    response = get_llm_output(input_text)
    # code == 1 represents failure
    if response['code'] == 1:
        map_html = gr.HTML(elem_id="map",
                           value=folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图',
                                            zoom_start=10)._repr_html_())
        return input_files, None, None, map_html, f"{response['error']}, error message: {response['message']}", None
    # else successfully get function calls
    call, thought = response['call'], response['thought']

    try:
        result = eval(call)
        if isinstance(result, tuple):
            result = result[0]

        # prepare outputs
        result_path = os.path.join(file_dir, 'result.shp')
        result.to_file(result_path, encoding='utf-8')

        # display result in files section, the user can download it
        if input_files is None:
            input_files = []
        input_files.append(result_path)
        # display result directly on a map
        map_html = display_in_map(result)
        # functions execution chain
        function_call_graph = graph_to_image()
        return input_files, call, thought, map_html, 'Done! result is saved at ' + result_path, function_call_graph
    # if the function calls go wrong
    except Exception as e:
        map = gr.HTML(elem_id="map", value=folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图',
                                                      zoom_start=10)._repr_html_())
        return input_files, call, thought, map, 'Error, error message: ' + str(e), None


title = """<h1 align="center">GIS Agent</h1>"""
description = 'Welcome to use GIS Agent!'

EXAMPLE_QUERIES = {
    "Geospatial Query": 'I want to go to "长沙世界之窗", list all the subway stations within 1 kilometer in the path to '
                        'there',
    "Compute the convex hull of points": 'compute the convex hull of all the points from "a.shp" that is in the area '
                                         'of "b.shp"',
    "Compute two regions' intersection": 'calculate the area of the intersection of "a.shp" and the buffer of '
                                         '"b.shp", buffer size is 0.0001',
    "Search for places in an area": "Get all the subways in the area of 'b.shp'",
}

css = """
#files {
    min-height:200px;
}
#func_call {
    min_height:20px;
}
"""

# themes of the main interface
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
)

# main interface layout
with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(elem_id='container', variant='compact'):
            # drop down box of examples
            examples = gr.Dropdown(
                choices=list(EXAMPLE_QUERIES.keys()), value=0, allow_custom_value=True, label="Examples",
                info="Choose one example to start"
            )
            # files input
            shp_input = gr.Files(elem_id='files', label='files', scale=2, elem_classes='files')

        with gr.Column():
            # interactive map to display results
            map = gr.HTML(elem_id="map", value=folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图',
                                                          zoom_start=10)._repr_html_())
    with gr.Row():
        # text input and text output section
        text_input = gr.Textbox(label='input text', interactive=True, lines=3, scale=1)
        text_output = gr.Textbox(label='output text', interactive=True, lines=3, scale=1)
    # submit and additional outputs
    submit_button = gr.Button("Submit", variant='primary', scale=1)
    function_call = gr.Code(elem_id='func_call', language='python', label='function calling', interactive=True, lines=1)
    task_planning = gr.Textbox(label='Task Planning', interactive=False)
    function_call_graph = gr.Image(label='function call graph')

    # when user presses enter, submit input to the system
    text_input.submit(
        chat,
        inputs=[shp_input, text_input],
        outputs=[shp_input, function_call, task_planning, map, text_output, function_call_graph],
    )

    # the user can also click submit to do the same thing
    submit_button.click(
        chat,
        inputs=[shp_input, text_input],
        outputs=[shp_input, function_call, task_planning, map, text_output, function_call_graph],
    )

    # when user upload files, automatically move them to the same place
    shp_input.upload(move_files, inputs=[shp_input], outputs=[shp_input])

    # when user click one of the examples, automatically fill the content in the files section and the text input
    # section.
    examples.select(
        fn=choose_example,
        inputs=[examples],
        outputs=text_input,
    )

demo.launch(share=True)
