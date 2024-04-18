import requests
import os
import json
import geopandas as gpd
from shapely import geometry
import numpy as np


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


def download_poi(keywords='', types='', region=''):
    url = f'''https://restapi.amap.com/v5/place/text?keywords={keywords}&types={types}
        &region={region}&key=016ad9192e40fc6d1054a83a21400429'''
    res = requests.get(url)
    json_data = json.loads(res.text)
    pois = json_data['pois']
    for poi in pois:
        location = poi['location'].split(',')
        lon, lat = gcj02_to_wgs84_one(eval(location[0]), eval(location[1]))
        poi['geometry'] = geometry.Point(lon, lat)

    gdf = gpd.GeoDataFrame(pois)
    if not os.path.exists(os.path.join(f'../shapefiles/poi', f'{keywords}_{region}')):
        os.mkdir(os.path.join(f'../shapefiles/poi', f'{keywords}_{region}'))

    gdf.to_file(os.path.join(f'../shapefiles/poi/{keywords}_{region}', f'{keywords}_{region}.shp'), encoding='utf-8')


if __name__ == "__main__":
    keywords = "中南大学校本部"
    region = "长沙市"
    # types = "110100"
    download_poi(keywords=keywords, region=region)
    # download_poi('', types, region)

    data = gpd.read_file('E:\GISAgent\shapefiles\poi\中南大学校本部_长沙市\中南大学校本部_长沙市.shp', encoding='utf-8')
    print(data.head())
