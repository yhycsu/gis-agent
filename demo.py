"""
为要素创建缓冲区的示例，
需要同时上传shp和shx文件
"""
import gradio as gr
import shutil
import time

from toolbox import *
from llm import *
from utils import *

# result_dir = '/home/a6000/yhy/gis-agent/shapefiles/results'
# file_dir = '/home/a6000/yhy/gis-agent/files'
result_dir = 'shapefiles/results'
file_dir = 'files'
tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'

def example1():
    gdf1 = gpd.read_file('shapefiles/poi/日料_长沙市/日料_长沙市.shp')
    gdf2 = gpd.read_file('shapefiles/poi/电影院_长沙市/电影院_长沙市.shp')
    gdf3 = gpd.read_file('shapefiles/poi/商场_长沙市/商场_长沙市.shp')

    target, other_pois = calculate_minimum_distance_with_other_pois(gdf1, [gdf2, gdf3])

    # 创建一个基本的高德地图
    m = folium.Map(location=[target.geometry.y, target.geometry.x], zoom_start=10)

    # 在地图上添加一个标记
    folium.Marker(
        location=[target.geometry.y, target.geometry.x],
        popup=target['name'],
        icon=folium.Icon(color="red")
    ).add_to(m)
    for poi in other_pois:
        folium.Marker(location=[poi.geometry.y, poi.geometry.x], popup=poi['name']).add_to(m)

    # 保存地图为HTML字符串
    html_map = m._repr_html_()

    # 自定义HTML样式
    html_content = f"""<div>
            {html_map}
        </div>
        """

    text_output = f"""
        已找到符合条件的地点：
        目的地为：{target['name']}，在地图中用红色标出
        距其最近的电影院为：{other_pois[0]['name']}，在地图中用蓝色标出
        距其最近的地铁站为：{other_pois[1]['name']}，在地图中用蓝色标出
    """

    task_plan = '''
    任务规划如下：
    1. 搜索附近的日料店
    2. 筛选价格适中的店铺：
    3. 评估排队时间：
    4. 查找附近的电影院和地铁站：
    5. 结合日料店的位置、价格、排队时间，以及附近的电影院和地铁站，给出符合用户需求的推荐日料店。
    '''

    return html_content, text_output, task_plan


def move_files(input_files):
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
    # clear function call graph
    clear_graph()

    # get llm output
    try:
        input_text += f' the file directory is {file_dir}'
        now = time.time()
        response = get_llm_output(input_text)
        print('function call time: ', time.time() - now)
        
        call, thought = response['call'], response['thought']
    except Exception as e:
        map = gr.HTML(elem_id="map", value=folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图', zoom_start=10)._repr_html_())
        return input_files, None, None, map_html, 'Error, error message: ' + str(e), None

    #  如果生成的函数执行出错
    try:
        now = time.time()
        result = eval(call)
        print('function execute time: ', time.time() - now)
        if isinstance(result, tuple):
            result = result[0]
        result_path = os.path.join(file_dir, 'result.shp')
        result.to_file(result_path, encoding='utf-8')

        if input_files is None:
            input_files = []
        input_files.append(result_path)
        map_html = display_in_map(result)
        # function_call_graph = graph_to_html()
        function_call_graph = graph_to_image()
        return input_files, call, thought, map_html, 'Done! result is saved at ' + result_path, function_call_graph
        
    except Exception as e:
        map = gr.HTML(elem_id="map", value=folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图', zoom_start=10)._repr_html_())
        return input_files, call, thought, map, 'Error, error message: ' + str(e), None  


title = """<h1 align="center">GIS Agent</h1>"""
description = 'Welcome to use GIS Agent!'

EXAMPLE_QUERIES = {
    "Geospatial Query": 'I want to go to "长沙世界之窗", list all the subway stations within 1 kilometer in the path to there',
    "Convex Hull": 'compute the convex hull of all the points from "a.shp" that is in the area of "b.shp"',
    "Regions Intersection": 'calculate the area of the intersection of "a.shp" and the buffer of "b.shp", buffer size is 0.0001',
    "Search an Area": "Get all the subways in the area of 'b.shp'",
}

css = """
#files {
    min-height:200px;
}
#func_call {
    min_height:20px;
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
)

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(elem_id='container', variant='compact'):
            # with gr.Row():
            #     gr.Label("Examples")
            #     examples = [
            #         gr.Button(query_name) for query_name in EXAMPLE_QUERIES
            #     ]
            examples = gr.Dropdown(
                list(EXAMPLE_QUERIES.keys()), label="Examples", info="Choose one example to start"
            )
            shp_input = gr.Files(elem_id='files', label='files', scale=2, elem_classes='files')
            
            
        with gr.Column():
            map = gr.HTML(elem_id="map", value=folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图', zoom_start=10)._repr_html_())
    text_input = gr.Textbox(label='input text', interactive=True, lines=3, scale=2)
    submit_button = gr.Button("Submit", variant='primary', scale=1)
    function_call = gr.Code(elem_id='func_call', language='python', label='function calling', interactive=False, lines=1)
    task_planning = gr.Textbox(label='Task Planning', interactive=False)
    # function_call_graph = gr.HTML(label='function call graph')
    function_call_graph = gr.Image(label='function call graph')
    text_output = gr.Textbox(label='output text', interactive=True)

    text_input.submit(
        chat,
        inputs=[shp_input, text_input],
        outputs=[shp_input, function_call, task_planning, map, text_output, function_call_graph],
    )

    submit_button.click(
        chat,
        inputs=[shp_input, text_input],
        outputs=[shp_input, function_call, task_planning, map, text_output, function_call_graph],
    )

    shp_input.upload(move_files, inputs=[shp_input], outputs=[shp_input])

    # for i, button in enumerate(examples):
    #     button.click(
    #         fn=EXAMPLE_QUERIES.get,
    #         inputs=button,
    #         outputs=text_input,
    #         api_name=f"button_click_{i}",
    #     )

    examples.select(
        fn=choose_example,
        inputs=[examples],
        outputs=text_input,
    )


demo.launch(share=True)