# -*- coding: utf-8 -*-
"""
包括GIS Agent所需要的工具

输入要求：
    为一个或多个GeoDataFrame，外加其它参数

输出要求：
    为一个或多个GeoDataFrame，外加其它结果
"""

from typing import List

import geopandas as gpd


# 地理处理工具
def create_buffer(gdf: gpd.GeoDataFrame, buffer_size: float) -> gpd.GeoDataFrame:
    """
    为一个要素创建缓冲区

    输入：
        gdf: 需要被创建缓冲区的要素
        buffer_size: 缓冲区大小

    输出：
        创建完缓冲区的要素
    """
    return gdf.buffer(buffer_size)


def create_multiple_buffers(gdfs: List[gpd.GeoDataFrame], buffer_sizes: List[float]) -> List[gpd.GeoDataFrame]:
    """
    为多个个要素创建缓冲区

    输入：
        gdfs: 需要被创建缓冲区的要素
        buffer_size: 缓冲区大小

    输出：
        创建完缓冲区的要素
    """
    results = []
    for gdf, buffer_size in zip(gdfs, buffer_sizes):
        results.append(gdf.buffer(buffer_size))
    return results
