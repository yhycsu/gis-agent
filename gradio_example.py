"""
为要素创建缓冲区的示例，
需要同时上传shp和shx文件
"""

import gradio as gr
import geopandas as gpd
import tempfile
import shutil
import os
import matplotlib.pyplot as plt

from toolbox import *


result_dir = '/Users/shuangzhiaishang/Documents/GISAgent/shapefiles/results'


def shp_to_image(shp_file_path):
    # 读取Shapefile文件
    gdf = gpd.read_file(shp_file_path)
    
    # 绘制Shapefile和缓冲区
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    gdf.plot(ax=axes[0])
    gdf_buffer.plot(ax=axes[1])
    axes[0].set_title('Original Shapefile')
    axes[1].set_title('Buffered Shapefile')
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, color='blue')

    # 绘制长沙市地图
    cities = gpd.read_file('shapefiles/china_shp/city/city.shp')
    city = cities[cities['ct_name'] == '长沙市']
    city.plot(ax=ax, color='green', alpha=0.5)

    ax.axis('off')  # 关闭坐标轴

    # 将地图保存为图片
    combined_image_path = os.path.join(result_dir, "combined_map_image.png")
    plt.savefig(combined_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return combined_image_path


def process(files, buffer_size):
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()

    # 将文件复制到临时文件夹
    for idx, file in enumerate(files):
        file_name = file.split('/')[-1]
        shutil.copy(file.name, os.path.join(temp_dir, file_name))

    # 读取临时文件夹中的文件
    files = os.listdir(temp_dir)
    for file in files:
        if file.endswith('.shp'):
            gdf = gpd.read_file(os.path.join(temp_dir, file))

    buffered_gdf = create_buffer(gdf, buffer_size)
    save_path = os.path.join(result_dir, 'buffer.shp')

    # 绘制Shapefile和缓冲区
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    gdf.plot(ax=axes[0])
    buffered_gdf.plot(ax=axes[1])
    axes[0].set_title('Original Shapefile')
    axes[1].set_title('Buffered Shapefile')
    plt.tight_layout()

    # 将地图保存为图片
    combined_image_path = os.path.join(result_dir, "combined_map_image.png")
    plt.savefig(combined_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return combined_image_path


demo = gr.Interface(
    fn=process,
    inputs=["files", 'number'],
    outputs=["image"],
)

demo.launch()
