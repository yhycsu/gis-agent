import requests
import geopandas as gpd
from shapely.geometry import LineString, Point


def download_route(origin, destination):
    # 构造请求 URL
    url = 'https://restapi.amap.com/v3/direction/driving?key={}&origin={}&destination={}'.format(app_key, origin,
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
        #'116.481196,39.989545;116.480939,39.989284;116.480886,39.989225'
        return gdf
    else:
        print('路径规划失败')


if __name__ == '__main__':
    from toolbox import *

    # 替换为您自己的 AppKey 和起点、终点坐标
    app_key = '016ad9192e40fc6d1054a83a21400429'
    origin = '116.481028,39.989643'
    destination = '116.434446,39.90816'

    gdf = download_route(origin, destination)
    buffered_gdf = create_buffer(gdf, 0.001)
    gdf.to_file('g.shp')
    buffered_gdf['gdf'].to_file('gg.shp')
    print(buffered_gdf)
