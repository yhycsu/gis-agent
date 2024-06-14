import os
import requests
import json
from functools import wraps
import io

import folium
import networkx as nx
from matplotlib import pyplot as plt
from pyvis.network import Network
import numpy as np
from PIL import Image


graph = nx.DiGraph()
call_stack = []
img_dir = 'images'

# node type:
# data, function
node_colors = {
    'input': 'lime',
    'tool': 'deepskyblue',
    'result': 'blueviolet',
    'data': 'coral'
}
node_types = ['Data(input)', 'Tool', 'Data(result)', 'Data(intermediate)']


def gcj02_to_wgs84_one(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """

    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    pi = 3.1415926535897932384626  # π
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323  # 偏心率平方

    def _transformlng(lng, lat):
        lng, lat = np.array(lng), np.array(lat)
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * pi) + 20.0 *
                np.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * pi) + 40.0 *
                np.sin(lng / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * pi) + 300.0 *
                np.sin(lng / 30.0 * pi)) * 2.0 / 3.0
        return ret

    def _transformlat(lng, lat):
        lng, lat = np.array(lng), np.array(lat)
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * pi) + 20.0 *
                np.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * pi) + 40.0 *
                np.sin(lat / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * pi) + 320 *
                np.sin(lat * pi / 30.0)) * 2.0 / 3.0
        return ret

    lng, lat = np.array(lng), np.array(lat)
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat


def add_point(map, row):
    geometry = row.geometry
    folium.Marker(
        location=[geometry.y, geometry.x],
        popup=row['name'],
        icon=folium.Icon("red"),
    ).add_to(map)


def add_polygon(map, row):
    x, y = row.geometry.exterior.coords.xy
    x = x.tolist()
    y = y.tolist()
    coords = [[y[i], x[i]] for i in range(len(x))]

    folium.Polygon(
        locations=coords,
        color="blue",
        weight=6,
        fill_color="red",
        fill_opacity=0.5,
        fill=True,
    ).add_to(map)


def add_line(map, row):
    x, y = row.geometry.xy
    x = x.tolist()
    y = y.tolist()
    coords = [[y[i], x[i]] for i in range(len(x))]

    folium.PolyLine(
        locations=coords,
        color="#FF0000",
        weight=5,
    ).add_to(map)


def plot_shp(gdf, func_name):
    # 创建绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax)

    # 保存绘图为图片文件
    fig_name = os.path.join(img_dir, f'{func_name}.png')
    plt.savefig(fig_name, dpi=300)
    return fig_name


def display_in_map(result):
    tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'
    m = folium.Map(location=[28.2278, 112.9389], tiles=tiles, attr='高德-常规图', zoom_start=10)
    for i in range(len(result)):
        try:
            row = result.iloc[i]
            if row.geometry.geom_type == 'Polygon':
                add_polygon(m, row)
            elif row.geometry.geom_type == 'Point':
                add_point(m, row)
            elif row.geometry.geom_type == 'LineString':
                add_line(m, row)
        except:
            continue

    m.save("index.html")
    return m._repr_html_()


def call_graph(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 函数调用栈为空，该函数是在其它函数内部调用
        if call_stack:
            return func(*args, **kwargs)

        func_name = func.__name__
        n = graph.number_of_nodes()
        num_params = len(kwargs)
        function_id = n
        # 添加函数节点
        graph.add_node(function_id, name=func_name, node_type='tool')

        call_stack.append(func_name)
        add_param = [True for i in range(num_params)]

        def create_link(arg, i):
            if isinstance(arg, tuple):
                graph.add_edge(arg[1], function_id)
                add_param[i] = False
                return arg[0]
            return arg

        # args = list(map(lambda x: create_link(x), args))
        kwargs = {key: create_link(value, i) for i, (key, value) in enumerate(kwargs.items())}
        result = func(*args, **kwargs)

        # 函数执行完后，添加函数结果，从栈里去除
        # 添加输入，函数，输出节点，连接起来
        # 添加输入
        i = 1
        for s, (k, v) in enumerate(kwargs.items()):
            if add_param[s]:
                graph.add_node(n + i, name=k, node_type='data')
                i += 1
        # 将函数和输入连接
        j = 1
        for s, (k, v) in enumerate(kwargs.items()):
            if add_param[s]:
                graph.add_edge(n + j, n)
                j += 1
        # 添加输出，将函数和输出连接
        graph.add_node(n + i, name=f'{func_name}_result', node_type='data')
        graph.add_edge(n, n + i)
        call_stack.pop()

        return result, n + i

    return wrapper

def graph_to_image():
    draw_call_graph()
    # 将 Matplotlib 图形转换为 PIL 图像
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image = Image.open(buffer)
    return image


def draw_call_graph():
    colors = [node_colors[graph.nodes[node]['node_type']] for node in graph.nodes()]
    labels = {node: graph.nodes()[node]['name'] for node in graph.nodes()}
    sinks = find_sink_node(graph)
    sources = find_source_node(graph)
    for sink_node in sinks:
        colors[sink_node] = node_colors['result']
    for source_node in sources:
        colors[source_node] = node_colors['input']
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(15, 12), dpi=200)
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=3000, font_size=10, font_weight="bold",
            arrows=True, node_color=colors)
    plt.title("Function Call Graph")
    legend_handles=[plt.Line2D([], [], color=color, label=label) for color, label in zip(node_colors.values(), node_types)]
    # 绘制图例，并设置背景颜色
    legend = plt.legend(handles=legend_handles, fontsize=15)
    legend.get_frame().set_facecolor('lightgrey')  # 设置图例背景颜色为浅灰色
    # plt.show()
    # plt.savefig('graph.png')


def get_call_graph():
    return graph


def read_html_file(file_path):
    """
    读取HTML文件并返回其内容作为字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        return html_content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    

def call_ollama_api(api_endpoint, payload):
    response = requests.post(api_endpoint, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()  # assuming the API returns JSON
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org')
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to retrieve IP address:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def find_sink_node(G):
    """
    Find the sink node in a NetworkX directed graph.

    :param G: A NetworkX directed graph
    :return: The sink node, or None if not found
    """
    sinks = []
    for node in G.nodes():
        if G.out_degree(node) == 0 and G.in_degree(node) > 0:
            sinks.append(node)
    return sinks


# Function to find the source node
def find_source_node(graph):
    # Initialize an empty list to store potential source nodes
    source_nodes = []

    # Iterate over all nodes in the graph
    for node in graph.nodes():
        # Check if the node has no incoming edges
        if graph.in_degree(node) == 0:
            # Add the node to the list of source nodes
            source_nodes.append(node)

    # Return the source nodes
    return source_nodes


def graph_to_html():
    nt = Network(notebook=True,
                    cdn_resources="remote",
                    directed=True,
                    # bgcolor="#222222",
                    # font_color="white",
                    height="800px",
                    # width="100%",
                    #  select_menu=True,
                    # filter_menu=True,

    )
    nt.from_nx(graph)

    sinks = find_sink_node(graph)
    sources = find_source_node(graph)

    # Set node colors based on node type
    node_colors = []
    node_names = []
    for node in nt.nodes:
        node_names.append(node['name'])
        # print('node:', node)
        if node['node_type'] == 'data':
            # print('node:', node)
            if node['label'] in sinks:
                node_colors.append('violet')  # lightgreen
                # print(node)
            elif node['label'] in sources:
                node_colors.append('lightgreen')  #
                # print(node)
            else:
                node_colors.append('orange')

        elif node['node_type'] == 'tool':
            node_colors.append('deepskyblue')

            # Update node colorsb
    for i, color in enumerate(node_colors):
        nt.nodes[i]['color'] = color
        # nt.nodes[i]['shape'] = 'box'
        nt.nodes[i]['label'] = node_names[i]
        nt.nodes[i]['shape'] = 'dot'
        # nt.set_node_style(node, shape="box")

    nt.show('graph.html')
    return read_html_file('graph.html')


def km_to_degrees(distance_km):
    # 地球赤道周长约为 40075 公里
    earth_circumference_km = 40075
    # 每度约为赤道上的 111 公里
    km_per_degree = earth_circumference_km / 360
    # 计算距离对应的角度
    degrees = distance_km / km_per_degree
    return degrees


def clear_graph():
    global graph
    graph = nx.DiGraph()


if __name__ == '__main__':
    # 示例：将 1000 公里转换为度
    distance_km = 1
    degrees = km_to_degrees(distance_km)
    print(f"{distance_km} 公里对应的角度为 {degrees} 度")