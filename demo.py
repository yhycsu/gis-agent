"""
为要素创建缓冲区的示例，
需要同时上传shp和shx文件
"""
import json
import gradio as gr
import geopandas as gpd
import tempfile
import shutil
import os
import matplotlib.pyplot as plt
import folium

#from toolbox import *
from llm import *
from toolbox import calculate_minimum_distance_with_other_pois,calculate_distance

result_dir = '/home/yuan/gis-agent/shapefiles/results'
file_dir = '/home/yuan/gis-agent/files'


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


def chat(input_files, input_text):
    input_text += f' the file directory is {file_dir}'
    response = get_llm_output(input_text)
    call, thought = response['call'], response['thought']
    try:
        result = eval(call)
    except Exception as e:
        return input_files, call, thought, None, 'Error, error message: ' + str(e)
    result_path = os.path.join(file_dir, 'result.shp')
    result.to_file(result_path)
    input_files.append(result_path)
    return input_files, call, thought, None, 'Done! result is saved at ' + result_path

    # # execution chain
    # output = {}
    # for tool_data in response["tools"]:
    #     func_name = tool_data["tool"]
    #     func_input = tool_data["tool_input"]
    #     func_input = update_input(func_input, output)
    #     # 获取全局命名空间中的函数对象
    #     g = globals()
    #     func = globals().get(func_name)
    #
    #     if func is not None and callable(func):
    #         if func_name == 'read_shp_file':
    #             output = func(temp_dir, **func_input)
    #             original_gdf = output['gdf']
    #         else:
    #             output = func(**func_input)
    #     else:
    #         print(f"Unknown function: {func_name}")
    #
    # gdf = output['gdf']
    # save_path = os.path.join(result_dir, 'buffer.shp')
    # gdf.to_file(save_path)

    # 绘制Shapefile和缓冲区
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # original_gdf.plot(ax=axes[0])
    # gdf.plot(ax=axes[1])
    # axes[0].set_title('Original Shapefile')
    # axes[1].set_title('Result Shapefile')
    # plt.tight_layout()
    #
    # # 将地图保存为图片
    # combined_image_path = os.path.join(result_dir, "combined_map_image.png")
    # plt.savefig(combined_image_path, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    #
    # image_output.visible = True
    # return combined_image_path, f'任务已完成！结果文件已保存到{save_path}'


title = """<h1 align="center">GIS Agent</h1>"""
description = 'Welcome to use GIS Agent!'

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            shp_input = gr.Files(label='input files')
            text_input = gr.Textbox(label='input text', interactive=True)
            function_call = gr.Textbox(label='function calling', interactive=False)
            task_planning = gr.Textbox(label='Task Planning', interactive=False)

            with gr.Row():
                submit_button = gr.Button("Submit", variant='primary', size='sm', scale=0)
                example_button = gr.Button("Example1", variant='primary', size='sm', scale=0)

        with gr.Column():
            image_output = gr.Image(label='output image')
            map = gr.HTML(folium.Map(location=[28.2278, 112.9389], zoom_start=10)._repr_html_())
            text_output = gr.Textbox(label='output text', interactive=True)

    with gr.Row():
        gr.Examples(examples=[
            ["想找个好吃的日料店，最好不要太贵，不要排太久的队，附近有电影院和地铁站。"],
        ], inputs=[text_input], fn=example1,
            outputs=[map, text_output])

    text_input.submit(
        chat,
        inputs=[shp_input, text_input],
        outputs=[shp_input, function_call, task_planning, image_output, text_output],
    )

    submit_button.click(
        chat,
        inputs=[shp_input, text_input],
        outputs=[shp_input, function_call, task_planning, image_output, text_output],
    )

    example_button.click(example1, inputs=None, outputs=[map, text_output, task_planning])

    shp_input.upload(move_files, inputs=[shp_input], outputs=[shp_input])


demo.launch(share=True)
