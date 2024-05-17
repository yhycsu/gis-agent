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
import numpy as np


def read_shp_file(file_dir: str, file_name: str) -> dict[str, gpd.GeoDataFrame]:
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
def create_buffer(gdf: gpd.GeoDataFrame, buffer_size: float):
    """
    create buffer for the gdf with buffer_size

    Parameters:
    - gdf (gpd.GeoDataFrame): the gdf
    - buffer_size (float): the buffer size in meters.

    Returns:
    - gpd.GeoDataFrame: the GeoDataFrame that is buffered with buffer_size
    """
    buffer = gdf.buffer(buffer_size)
    return gpd.GeoDataFrame(geometry=buffer)


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


def get_path(origin, destination):
    """
    get the path from origin to destination.

    Parameters:
    - origin (tuple): contains the coordinates of the origin
    - destination (tuple): contains the coordinates of the destination

    Returns:
    - gpd.GeoDataFrame: the path from origin to destination
    """

    return read_shp_file('', 'shapefiles/results/buffer.shp')


def get_poi(keyword):
    """
        get the locations containing the specific keyword

        Parameters:
        - keyword (str): the desired keyword

        Returns:
        - gpd.GeoDataFrame: the locations that contains the specific keyword
        """

    return read_shp_file('', 'shapefiles/results/buffer.shp')


def clip_shp(input_shp: gpd.GeoDataFrame, clip_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    裁剪一个shapefile

    输入：
        input_shp: 需要被裁剪的要素
        clip_area: 裁剪区域

    输出：
        裁剪完的要素
    """

    clip_extent = clip_area.geometry.unary_union

    # 进行裁剪操作
    clipped = input_shp.copy()
    clipped['geometry'] = input_shp.intersection(clip_extent)

    return clipped


def compute_convex_hull(gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """
    计算一个shapefile的凸包

    输入：
        gdf: 需要计算的要素

    输出：
        计算完的要素
    """
    convex_hull = gdf.unary_union.convex_hull

    # 将凸包几何保存到一个新的GeoDataFrame
    convex_hull_gdf = gpd.GeoDataFrame(geometry=[convex_hull])
    return {'gdf': convex_hull_gdf}


def union_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame):
    """
    联合两个 GeoDataFrame。

    参数:
        input_gdf1 (GeoDataFrame): 第一个 GeoDataFrame。
        input_gdf2 (GeoDataFrame): 第二个 GeoDataFrame。

    返回:
        merged_gdf (GeoDataFrame): 融合后的 GeoDataFrame。
    """
    union = gpd.overlay(input_gdf1, input_gdf2, how='union')
    return union


def dissolve_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame, att):
    """
    融合两个 GeoDataFrame。

    参数:
        input_gdf1 (GeoDataFrame): 第一个 GeoDataFrame。
        input_gdf2 (GeoDataFrame): 第二个 GeoDataFrame。
        att (str): 用于融合的属性列。

    返回:
        dissolved_union (GeoDataFrame): 融合后的 GeoDataFrame。
    """

    # 计算几何并集
    union = gpd.overlay(input_gdf1, input_gdf2, how='union')

    # Dissolve操作
    dissolved_union = union.dissolve(by=att)
    return dissolved_union


def intersection_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame):
    """
    计算两个 GeoDataFrame交集。
    参数:
        input_gdf1 (GeoDataFrame): 第一个 GeoDataFrame。
        input_gdf2 (GeoDataFrame): 第二个 GeoDataFrame。
    返回:
        intersection (GeoDataFrame): 交集的 GeoDataFrame。
    """

    # 计算交集
    intersection = gpd.overlay(input_gdf1, input_gdf2, how='intersection')
    return intersection


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


if __name__ == '__main__':
    # import geopandas as gpd
    # gdf1 = gpd.read_file('shapefiles/poi/日料_长沙市/日料_长沙市.shp')
    # gdf2 = gpd.read_file('shapefiles/poi/电影院_长沙市/电影院_长沙市.shp')
    # gdf3 = gpd.read_file('shapefiles/poi/商场_长沙市/商场_长沙市.shp')
    #
    # distance_1_2 = calculate_distance(gdf1, gdf2)
    # distance_1_3 = calculate_distance(gdf1, gdf3)
    #
    # distance_index_1_2 = []
    # distance_index_1_3 = []
    #
    # for i, distance in enumerate(distance_1_2):
    #     min_index = np.argmin(distance)
    #     distance_index_1_2.append(min_index)
    #
    # for i, distance in enumerate(distance_1_3):
    #     min_index = np.argmin(distance)
    #     distance_index_1_3.append(min_index)
    #
    # print(distance_index_1_2)
    # print(distance_index_1_3)
    #
    # min_dist = 1e9
    # target = None
    # for i in range(len(gdf1)):
    #     min_index1 = distance_index_1_2[i]
    #     min_index2 = distance_index_1_3[i]
    #
    #     total_dist = distance_1_2[i][min_index1] + distance_1_3[i][min_index2]
    #     if total_dist < min_dist:
    #         min_dist = total_dist
    #         target = i
    #
    # print(gdf1.iloc[target], distance_index_1_2[target], distance_index_1_3[target])
    #
    # target, other_poi = calculate_minimum_distance_with_other_pois(gdf1, [gdf2, gdf3])
    # print(target.geometry.x)
    # print(target['name'])
    #print(get_llm_output("请计算以下要素的的凸包，并使用它的缓冲区进行裁剪，文件路径为: changsha.shp，buffer_size为0.001"))
    gdf = gpd.read_file("/home/yuan/gis-agent/shapefiles/poi/中南大学校本部_长沙市/中南大学校本部_长沙市.shp")
    print(compute_difference(gdf, gdf))
