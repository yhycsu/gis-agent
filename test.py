"""
用于测试的文件
"""
import os

import geopandas as gpd

from toolbox import *

# 测试结果存储位置
test_dir = 'shapefiles/test'


def test_buffer():
    # 测试单个要素缓冲区
    gdf = gpd.read_file('/shapefiles/poi/_长沙市/_长沙市.shp')
    buffer_size = 0.01
    buffered_gdf = create_buffer(gdf, buffer_size)
    buffered_gdf.to_file(os.join(test_dir, 'buffer.shp'))

    # 测试多个要素缓冲区
    gdfs = [gdf for _ in range(5)]
    buffer_sizes = [buffer_size * i for i in range(5)]
    buffered_gdfs = create_multiple_buffers(gdfs, buffer_sizes)
    for i in range(5):
        buffered_gdf.to_file(os.join(test_dir, f'buffer{i}.shp'))


if __name__ == "__main__":
    '''
    测试完后可以将得到的结果与用QGIS操作得到的结果进行比较
    '''
    test_buffer()
