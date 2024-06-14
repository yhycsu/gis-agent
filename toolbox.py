# -*- coding: utf-8 -*-
"""
包括GIS Agent所需要的工具

输入要求：
    为一个或多个GeoDataFrame，外加其它参数

输出要求：
    为一个或多个GeoDataFrame，外加其它结果
"""
import os
from typing import List, get_type_hints

import geopandas as gpd
import requests
from shapely import Point, LineString

from tools.poi_downloader import download_poi
from utils import *


import pandas as pd
from pyproj import CRS
from scipy.spatial import KDTree
from jinja2 import Template


api_key = '016ad9192e40fc6d1054a83a21400429'
poi_dir = 'shapefiles/poi'


@call_graph
def read_shp_file(file_dir: str, file_name: str):
    """
    read shapefile from the file directory and return as GeoDataFrame.

    Parameters:
    - temp_dir (str): the directory of shapefiles
    - file_name (str): the name of the shapefile

    Returns:
    - gpd.GeoDataFrame: the GeoDataFrame of the shapefile
    """""
    path = os.path.join(file_dir, file_name)
    gdf = gpd.read_file(path)
    return gdf


# 地理处理工具
@call_graph
def create_buffer(gdf: gpd.GeoDataFrame, buffer_size: float):
    """
    create buffer for the gdf with buffer_size

    Parameters:
    - gdf (gpd.GeoDataFrame): the gdf
    - buffer_size (float): the buffer size in meters.

    Returns:
    - gpd.GeoDataFrame: the GeoDataFrame that is buffered with buffer_size
    """
    buffer_size = km_to_degrees(buffer_size)
    buffer = gdf.buffer(buffer_size)
    return gpd.GeoDataFrame(geometry=buffer)


@call_graph
def compute_difference(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    compute the difference between two GeoDataFrames.

    Parameters:
    - gdf1 (gpd.GeoDataFrame): one GeoDataFrame
    - gdf2 (gpd.GeoDataFrame): the other GeoDataFrame

    Returns:
    - gpd.GeoDataFrame: the difference between gdf1 and gdf2
    """

    # 计算两个 GeoDataFrame 之间的差集
    difference = gdf1.geometry.difference(gdf2.unary_union)

    # 创建新的 GeoDataFrame
    difference_gdf = gpd.GeoDataFrame(geometry=difference)

    return difference_gdf


@call_graph
def filter(gdf, area):
    """
    filter the gdf in a specific area.

    Parameters:
    - gdf (gpd.GeoDataFrame): the GeoDataFrame to be filtered
    - gdf2 (gpd.GeoDataFrame): the area

    Returns:
    - gpd.GeoDataFrame: the filtered GeoDataFrame
    """
    points_within_polygons = gpd.sjoin(gdf, area, how="inner", op="within")
    return points_within_polygons


@call_graph
def get_path(origin: tuple, destination: tuple):
    """
    get the path from origin to destination.

    Parameters:
    - origin (tuple): contains the coordinates of the origin
    - destination (tuple): contains the coordinates of the destination

    Returns:
    - gpd.GeoDataFrame: the path from origin to destination
    """
    if not isinstance(origin, tuple) or not isinstance(destination, tuple):
        origin = tuple(origin)
        destination = tuple(destination)
    origin = f'{origin[0]},{origin[1]}'
    destination = f'{destination[0]},{destination[1]}'
    # 构造请求 URL
    url = 'https://restapi.amap.com/v3/direction/driving?key={}&origin={}&destination={}'.format(api_key, origin,
                                                                                                 destination)
    # 发送 HTTP 请求
    response = requests.get(url)

    # 解析 JSON 数据
    result = response.json()
    if result['status'] == '1':
        # 获取路径距离和时间
        distance = result['route']['paths'][0]['distance']
        duration = result['route']['paths'][0]['duration']

        # 获取路线节点
        steps = result['route']['paths'][0]['steps']
        points = []
        for step in steps:
            line = step['polyline']
            for point in line.split(';'):
                point = point.split(',')
                points.append(Point(eval(point[0]), eval(point[1])))

        line = LineString(points)
        gdf = gpd.GeoDataFrame(geometry=[line])
        # step['polyline']
        # '116.481196,39.989545;116.480939,39.989284;116.480886,39.989225'
        return gdf
    return None


@call_graph
def get_poi(keyword):
    """
    get the locations containing the specific keyword

    Parameters:
    - keyword (str): the desired keyword

    Returns:
    - gpd.GeoDataFrame: the locations that contains the specific keyword
    """
    if not os.path.exists(os.path.join(poi_dir, f'{keyword}_长沙市')):
        download_poi(keyword)
    return read_shp_file(poi_dir, f'{keyword}_长沙市/{keyword}_长沙市.shp')


@call_graph
def clip_shp(input_gdf: gpd.GeoDataFrame, clip_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    clip a GeoDataFrame with clip area

    Parameters:
    - input_gdf (gpd.GeoDataFrame): the GeoDataFrame to be clipped
    - clip_area (gpd.GeoDataFrame): clip area

    Returns:
    - gpd.GeoDataFrame: the clipped GeoDataFrame
    """

    clip_extent = clip_area.geometry.unary_union

    # 进行裁剪操作
    clipped = input_gdf.copy()
    clipped['geometry'] = input_gdf.intersection(clip_extent)

    return clipped


@call_graph
def compute_convex_hull(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    compute the convex hull of a GeoDataFrame

    Parameters:
    - gdf (gpd.GeoDataFrame): the gdf

    Returns:
    - gpd.GeoDataFrame: the convex hull of the GeoDataFrame
    """
    convex_hull = gdf.unary_union.convex_hull

    # 将凸包几何保存到一个新的GeoDataFrame
    convex_hull_gdf = gpd.GeoDataFrame(geometry=[convex_hull])
    return convex_hull_gdf


@call_graph
def union_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame):
    """
    compute the union of two GeoDataFrames

    Parameters:
    - input_gdf1 (gpd.GeoDataFrame): one gdf
    - input_gdf2 (gpd.GeoDataFrame): the other gdf

    Returns:
    - gpd.GeoDataFrame: the union of two GeoDataFrames
    """
    union = gpd.overlay(input_gdf1, input_gdf2, how='union')
    return union


@call_graph
def dissolve_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame, att):
    """
    融合两个 GeoDataFrame。

    参数:
    - input_gdf1 (GeoDataFrame): 第一个 GeoDataFrame。
    - input_gdf2 (GeoDataFrame): 第二个 GeoDataFrame。
    - att (str): 用于融合的属性列。

    返回:
        dissolved_union (GeoDataFrame): 融合后的 GeoDataFrame。
    """

    # 计算几何并集
    union = gpd.overlay(input_gdf1, input_gdf2, how='union')

    # Dissolve操作
    dissolved_union = union.dissolve(by=att)
    return dissolved_union


@call_graph
def intersection_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame):
    """
    compute the intersection of two GeoDataFrames

    Parameters:
    - input_gdf1 (gpd.GeoDataFrame): one gdf
    - input_gdf2 (gpd.GeoDataFrame): the other gdf

    Returns:
    - gpd.GeoDataFrame: the intersection of two GeoDataFrames
    """

    # 计算交集
    intersection = gpd.overlay(input_gdf1, input_gdf2, how='intersection')
    return intersection


@call_graph
def symmetrical_difference_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame):
    """
    两个 GeoDataFrame交集取反。

    参数:
        input_gdf1 (GeoDataFrame): 第一个 GeoDataFrame。
        input_gdf2 (GeoDataFrame): 第二个 GeoDataFrame。

    返回:
        merged_gdf (GeoDataFrame): 融合后的 GeoDataFrame。
    """
    symmetric_difference = gpd.overlay(input_gdf1, input_gdf2, how='symmetric_difference')
    return symmetric_difference


@call_graph
def calculate_distance(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame):
    # 遍历第一个shapefile中的每个点
    results = []
    for index1, row1 in gdf1.iterrows():
        point1 = row1.geometry

        result = []
        # 遍历第二个shapefile中的每个点
        for index2, row2 in gdf2.iterrows():
            point2 = row2.geometry

            # 计算两个点之间的距离
            distance = point1.distance(point2)

            result.append(distance)

        results.append(result)
    return results


@call_graph
def calculate_minimum_distance_with_other_pois(gdf: gpd.GeoDataFrame, other_gdfs: List[gpd.GeoDataFrame]):
    distance_with_other_pois = [calculate_distance(gdf, gdf_other) for gdf_other in other_gdfs]

    distance_index_with_others = []

    for distance_with_i in distance_with_other_pois:
        d = []
        for i, distance in enumerate(distance_with_i):
            min_index = np.argmin(distance)
            d.append(min_index)
        distance_index_with_others.append(d)

    min_dist = 1e9
    target = None
    for i in range(len(gdf)):
        total_dist = 0
        for j in range(len(distance_index_with_others)):
            min_index = distance_index_with_others[j][i]
            total_dist += distance_with_other_pois[j][i][min_index]

        if total_dist < min_dist:
            min_dist = total_dist
            target = i

    others = [other_gdfs[i].iloc[distance_index_with_others[i][target]] for i in range(len(other_gdfs))]

    return gdf.iloc[target], others


#统计多边形内点的数量
@call_graph
def count_points_in_polygons(point_layer, polygon_layer, weight_field=None, unique_field=None):
    """
    统计点在多边形中的数量，生成包含点数的新多边形图层。

    参数：
    point_layer (GeoDataFrame): 点图层。
    polygon_layer (GeoDataFrame): 多边形图层。
    weight_field (str, 可选): 点图层中的权重字段。
    unique_field (str, 可选): 点图层中的唯一类字段。

    返回：
    GeoDataFrame: 包含点数的新多边形图层。
    """

    # 复制多边形图层以生成输出图层
    result_polygons = polygon_layer.copy()

    # 初始化一个新的字段用于存储点数
    result_polygons['point_count'] = 0

    # 检查是否设置了权重字段和唯一类字段
    if weight_field and unique_field:
        print("权重字段优先，唯一类字段将被忽略。")
        unique_field = None

    # 遍历每个多边形
    for idx, polygon in result_polygons.iterrows():
        # 获取当前多边形的几何信息
        poly_geom = polygon.geometry

        # 查找在当前多边形内的所有点
        points_within_poly = point_layer[point_layer.within(poly_geom)]

        if weight_field:
            # 如果设置了权重字段，计算权重字段之和
            point_count = points_within_poly[weight_field].sum()
        elif unique_field:
            # 如果设置了唯一类字段，对点进行分类并只计数唯一类
            point_count = points_within_poly[unique_field].nunique()
        else:
            # 否则，直接计数点的数量
            point_count = len(points_within_poly)

        # 更新多边形的点数
        result_polygons.at[idx, 'point_count'] = point_count

    return result_polygons


# 示例用法
# point_layer = gpd.read_file('path_to_points.shp')
# polygon_layer = gpd.read_file('path_to_polygons.shp')
# result = count_points_in_polygons(point_layer, polygon_layer, weight_field='weight', unique_field=None)
# result.to_file('path_to_result.shp')


#线相交


@call_graph
def create_intersection_points(input_lines, intersect_lines):
    """
    创建“相交图层”中的线与“输入图层”中的线相交处的点要素。

    参数：
    input_lines (GeoDataFrame): 输入图层中的线。
    intersect_lines (GeoDataFrame): 相交图层中的线。

    返回：
    GeoDataFrame: 包含相交点的点图层。
    """

    # 初始化一个列表来存储相交点
    intersection_points = []

    # 遍历每条输入线
    for idx1, line1 in input_lines.iterrows():
        # 遍历每条相交线
        for idx2, line2 in intersect_lines.iterrows():
            # 查找线之间的交集
            intersection = line1.geometry.intersection(line2.geometry)

            # 如果交集是点，将其添加到列表中
            if intersection.is_empty:
                continue

            if intersection.geom_type == 'Point':
                intersection_points.append(intersection)

            # 如果交集是多个点（MultiPoint），则将每个点添加到列表中
            elif intersection.geom_type == 'MultiPoint':
                for point in intersection:
                    intersection_points.append(point)

    # 创建包含相交点的GeoDataFrame
    intersection_points_gdf = gpd.GeoDataFrame(geometry=intersection_points, crs=input_lines.crs)

    return intersection_points_gdf


# 示例用法
# input_lines = gpd.read_file('path_to_input_lines.shp')
# intersect_lines = gpd.read_file('path_to_intersect_lines.shp')
# result = create_intersection_points(input_lines, intersect_lines)
# result.to_file('path_to_result.shp')


#平均坐标


@call_graph
def generate_centroids(input_layer, weight_field=None, unique_id_field=None):
    """
    生成包含输入图层中几何图形质心的点图层，可以指定权重字段和唯一ID字段。

    参数：
    input_layer (GeoDataFrame): 输入图层。
    weight_field (str, 可选): 计算质心时各要素的权重字段。
    unique_id_field (str, 可选): 用于分组的唯一ID字段。

    返回：
    GeoDataFrame: 包含质心点的点图层。
    """

    # 如果指定了唯一ID字段，根据该字段进行分组
    if unique_id_field:
        grouped = input_layer.groupby(unique_id_field)
    else:
        grouped = [(None, input_layer)]

    centroids = []

    for group_name, group in grouped:
        if weight_field:
            # 使用加权质心计算
            weighted_x = sum(group.geometry.centroid.x * group[weight_field]) / group[weight_field].sum()
            weighted_y = sum(group.geometry.centroid.y * group[weight_field]) / group[weight_field].sum()
            centroid = Point(weighted_x, weighted_y)
        else:
            # 使用普通质心计算
            centroid = group.geometry.centroid.unary_union.centroid

        # 构建质心的属性字典
        properties = {}
        if unique_id_field:
            properties[unique_id_field] = group_name

        # 创建一个包含质心的GeoSeries
        centroid_geo = gpd.GeoSeries([centroid], crs=input_layer.crs)
        centroid_geo = centroid_geo.to_frame(name='geometry')
        centroid_geo = gpd.GeoDataFrame(centroid_geo, geometry='geometry')

        for key, value in properties.items():
            centroid_geo[key] = value

        centroids.append(centroid_geo)

    # 合并所有质心点为一个GeoDataFrame
    centroids_gdf = pd.concat(centroids, ignore_index=True)

    return centroids_gdf


# 示例用法
# input_layer = gpd.read_file('path_to_input_layer.shp')
# result = generate_centroids(input_layer, weight_field='weight', unique_id_field='id')
# result.to_file('path_to_result.shp')


#最近邻分析


@call_graph
def nearest_neighbor_analysis(point_layer, output_html):
    """
    对点图层执行最近邻分析，并生成包含统计值的HTML报告。

    参数：
    point_layer (GeoDataFrame): 点图层。
    output_html (str): 输出HTML文件路径。

    返回：
    None
    """

    # 提取点的坐标
    coords = np.array([(point.x, point.y) for point in point_layer.geometry])

    # 创建KDTree以进行快速最近邻搜索
    tree = KDTree(coords)

    # 查找每个点的最近邻距离
    distances, _ = tree.query(coords, k=2)
    nearest_distances = distances[:, 1]

    # 计算最近邻平均距离
    mean_nn_distance = np.mean(nearest_distances)

    # 计算点的密度
    area = point_layer.total_bounds
    width = area[2] - area[0]
    height = area[3] - area[1]
    study_area = width * height
    point_density = len(point_layer) / study_area

    # 计算期望的最近邻距离
    expected_nn_distance = 1 / (2 * np.sqrt(point_density))

    # 计算R指数
    r_index = mean_nn_distance / expected_nn_distance

    # 最近邻分析结果
    if r_index < 1:
        pattern = "Clustered"
    elif r_index > 1:
        pattern = "Dispersed"
    else:
        pattern = "Random"

    # 准备HTML模板
    html_template = """
    <html>
    <head>
        <title>Nearest Neighbor Analysis</title>
    </head>
    <body>
        <h1>Nearest Neighbor Analysis Report</h1>
        <p><strong>Number of Points:</strong> {{ num_points }}</p>
        <p><strong>Mean Nearest Neighbor Distance:</strong> {{ mean_nn_distance }}</p>
        <p><strong>Point Density:</strong> {{ point_density }}</p>
        <p><strong>Expected Nearest Neighbor Distance:</strong> {{ expected_nn_distance }}</p>
        <p><strong>R Index:</strong> {{ r_index }}</p>
        <p><strong>Distribution Pattern:</strong> {{ pattern }}</p>
    </body>
    </html>
    """

    # 渲染HTML模板
    template = Template(html_template)
    html_content = template.render(
        num_points=len(point_layer),
        mean_nn_distance=mean_nn_distance,
        point_density=point_density,
        expected_nn_distance=expected_nn_distance,
        r_index=r_index,
        pattern=pattern
    )

    # 写入HTML文件
    with open(output_html, "w") as f:
        f.write(html_content)


# 示例用法
# point_layer = gpd.read_file('path_to_point_layer.shp')
# nearest_neighbor_analysis(point_layer, 'output_report.html')


@call_graph
def calculate_line_length_in_polygons(line_layer_path, polygon_layer_path, output_layer_path,
                                      length_field_name='line_length', count_field_name='line_count'):
    # 读取线图层和多边形图层
    lines = gpd.read_file(line_layer_path)
    polygons = gpd.read_file(polygon_layer_path)

    # 检查坐标参考系统是否一致，不一致则转换
    if lines.crs != polygons.crs:
        lines = lines.to_crs(polygons.crs)

    # 创建新的字段并初始化为0
    polygons[length_field_name] = 0.0
    polygons[count_field_name] = 0

    # 遍历每个多边形，计算其内线的总长度和数量
    for idx, polygon in polygons.iterrows():
        # 获取与当前多边形相交的所有线
        intersecting_lines = lines[lines.intersects(polygon.geometry)]

        # 计算这些线的总长度
        total_length = intersecting_lines.geometry.length.sum()

        # 计算这些线的数量
        line_count = len(intersecting_lines)

        # 将计算结果赋值给当前多边形
        polygons.at[idx, length_field_name] = total_length
        polygons.at[idx, count_field_name] = line_count

    # 保存结果到新的图层文件
    polygons.to_file(output_layer_path)


'''
# 示例调用
line_layer_path = 'path/to/your/line_layer.shp'
polygon_layer_path = 'path/to/your/polygon_layer.shp'
output_layer_path = 'path/to/your/output_layer.shp'

calculate_line_length_in_polygons(line_layer_path, polygon_layer_path, output_layer_path)

'''

@call_graph
def get_current_latitude_and_longitude():
    """
    get the current latitude and longitude.

    Returns:
    - tuple: the latitude and longitude of the current place
    """
    public_ip = get_public_ip()
    url = f'https://restapi.amap.com/v3/ip?key=016ad9192e40fc6d1054a83a21400429&ip={public_ip}'
    res = requests.get(url)
    json_data = json.loads(res.text)
    rectangle = json_data['rectangle'].split(';')
    left_and_bottom = rectangle[0].split(',')
    right_and_top = rectangle[1].split(',')
    center_lon = (eval(left_and_bottom[0]) + eval(right_and_top[0])) / 2
    center_lat = (eval(left_and_bottom[1]) + eval(right_and_top[1])) / 2
    return [center_lon, center_lat]


@call_graph
def get_latitude_and_longitude_of_a_location(location):
    """
    get the current latitude and longitude of the target location

    Returns:
    - tuple: the latitude and longitude of the current place
    """
    url = f'https://restapi.amap.com/v3/geocode/geo?key=016ad9192e40fc6d1054a83a21400429&address={location}'
    res = requests.get(url)
    json_data = json.loads(res.text)
    location = json_data['geocodes'][0]['location'].split(',')
    location = list(map(lambda x: eval(x), location))
    return location


@call_graph
def calculate_the_area_of_polygon(gdf):
    """
    calculate the area of the polygon

    Parameters:
    - gdf (gpd.GeoDataFrame): the GeoDataFrame that contains the polygon

    Returns:
    - float: the area of the polygon in square meters
    """
    gdf.crs = 'EPSG:3857'
    target_crs = CRS.from_epsg(32649)  # UTM Zone 49N
    gdf_projected = gdf.to_crs(target_crs)

    area = gdf_projected.iloc[0].geometry.area
    return area


if __name__ == '__main__':
    # clip_shp(input_gdf=get_poi(keyword='subway station'), clip_area=create_buffer(gdf=get_path(origin=get_current_latitude_and_longitude(), destination=get_latitude_and_longitude_of_a_location(location='长沙世界之窗')), buffer_size=1000))
    # a = calculate_the_area_of_polygon(gdf=read_shp_file('', 'files/area.shp'))
    # print(a)
    a = clip_shp(input_gdf=get_poi(keyword='subway station'), clip_area=read_shp_file(file_dir='/home/a6000/yhy/gis-agent/files', file_name='area.shp'))
    print(a[0].head())
    display_in_map(a[0])
    print(a[0])