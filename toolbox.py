# -*- coding: utf-8 -*-
"""
includes tools for GIS Agent to call
"""
from typing import List

import pandas as pd
from jinja2 import Template
from pyproj import CRS
from scipy.spatial import KDTree
from shapely import Point, LineString

from utils import *


# api to download data from Gaode map
api_key = '016ad9192e40fc6d1054a83a21400429'
# place to store downloaded poi data
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

    # Calculate the difference between two GeoDataFrames
    difference = gdf1.geometry.difference(gdf2.unary_union)

    #Create a new GeoDataFrame
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
    # Construct request URL
    url = 'https://restapi.amap.com/v3/direction/driving?key={}&origin={}&destination={}'.format(api_key, origin,
                                                                                                 destination)
    # send HTTP request
    response = requests.get(url)

    # parse JSON data
    result = response.json()
    if result['status'] == '1':
        # Get route nodes
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
    city = get_location_from_ip(get_public_ip())
    if not city:
        raise ValueError('Sorry, we currently only support cities in China because of some restrictions.')
    if not os.path.exists(os.path.join(poi_dir, f'{keyword}_{city}')):
        download_poi(keyword, city)
    return read_shp_file(poi_dir, f'{keyword}_{city}/{keyword}_{city}.shp')


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

    # Perform cropping operation
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

    # Save the convex hull geometry to a new GeoDataFrame
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
    Fusion of two GeoDataFrames.

    parameter:
    - input_gdf1 (GeoDataFrame): The first GeoDataFrame.
    - input_gdf2 (GeoDataFrame): Second GeoDataFrame.
    - att (str): attribute column used for fusion.

    return:
        dissolved_union (GeoDataFrame): Dissolved GeoDataFrame.
    """

    # Compute geometric union
    union = gpd.overlay(input_gdf1, input_gdf2, how='union')

    # Dissolve operation
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

    # calculate intersection
    intersection = gpd.overlay(input_gdf1, input_gdf2, how='intersection')
    return intersection


@call_graph
def symmetrical_difference_features(input_gdf1: gpd.GeoDataFrame, input_gdf2: gpd.GeoDataFrame):
    """
    The intersection of two GeoDataFrames is negated.

    parameter:
        input_gdf1 (GeoDataFrame): The first GeoDataFrame.
        input_gdf2 (GeoDataFrame): The second GeoDataFrame.

    return:
        merged_gdf (GeoDataFrame): Merged GeoDataFrame.
    """
    symmetric_difference = gpd.overlay(input_gdf1, input_gdf2, how='symmetric_difference')
    return symmetric_difference


@call_graph
def calculate_distance(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame):
    # Traverse each point in the first shapefile
    results = []
    for index1, row1 in gdf1.iterrows():
        point1 = row1.geometry
        result = []
        # Traverse each point in the second shapefile
        for index2, row2 in gdf2.iterrows():
            point2 = row2.geometry
            # Calculate the distance between two points
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


@call_graph
def count_points_in_polygons(point_layer, polygon_layer, weight_field=None, unique_field=None):
    """
    Count the number of points in a polygon and generate a new polygon layer containing the number of points.

    parameter:
    point_layer (GeoDataFrame): point layer.
    polygon_layer (GeoDataFrame): Polygon layer.
    weight_field (str, optional): The weight field in the point layer.
    unique_field (str, optional): The unique field in the point layer.

    return:
    GeoDataFrame: New polygon layer containing points.
    """
    # Copy the polygon layer to generate the output layer
    result_polygons = polygon_layer.copy()

    # Initialize a new field to store points
    result_polygons['point_count'] = 0

    # Check whether the weight field and unique class field are set
    if weight_field and unique_field:
        print("Weight fields take precedence, unique class fields will be ignored.")
        unique_field = None

    # Iterate through each polygon
    for idx, polygon in result_polygons.iterrows():
        # Get the geometric information of the current polygon
        poly_geom = polygon.geometry

        # Find all points within the current polygon
        points_within_poly = point_layer[point_layer.within(poly_geom)]

        if weight_field:
            # If the weight field is set, calculate the sum of the weight fields
            point_count = points_within_poly[weight_field].sum()
        elif unique_field:
            # If the unique class field is set, classify points and count only unique classes
            point_count = points_within_poly[unique_field].nunique()
        else:
            # Otherwise, count the number of points directly
            point_count = len(points_within_poly)

        # Update the number of points of the polygon
        result_polygons.at[idx, 'point_count'] = point_count

    return result_polygons


@call_graph
def create_intersection_points(input_lines, intersect_lines):
    """
    Creates point features where lines from the Intersect Layer intersect lines from the Input Layer.

    parameter:
    input_lines (GeoDataFrame): Lines in the input layer.
    intersect_lines (GeoDataFrame): Intersect lines in layers.

    return:
    GeoDataFrame: Point layer containing intersection points.
    """

    # Initialize a list to store intersection points
    intersection_points = []

    # Traverse each input line
    for idx1, line1 in input_lines.iterrows():
        # Traverse each intersecting line
        for idx2, line2 in intersect_lines.iterrows():
            # Find intersections between lines
            intersection = line1.geometry.intersection(line2.geometry)

            # If the intersection is a point, add it to the list
            if intersection.is_empty:
                continue

            if intersection.geom_type == 'Point':
                intersection_points.append(intersection)

            # If the intersection is multiple points (MultiPoint), add each point to the list
            elif intersection.geom_type == 'MultiPoint':
                for point in intersection:
                    intersection_points.append(point)

    #Create a GeoDataFrame containing intersection points
    intersection_points_gdf = gpd.GeoDataFrame(geometry=intersection_points, crs=input_lines.crs)

    return intersection_points_gdf


@call_graph
def generate_centroids(input_layer, weight_field=None, unique_id_field=None):
    """
    Generates a point layer containing the centroids of the geometry in the input layer. You can specify a weight field and a unique ID field.

    parameter:
    input_layer (GeoDataFrame): input layer.
    weight_field (str, optional): The weight field of each feature when calculating the centroid.
    unique_id_field (str, optional): Unique ID field used for grouping.

    return:
    GeoDataFrame: Point layer containing centroid points.
    """

    # If a unique ID field is specified, group according to this field
    if unique_id_field:
        grouped = input_layer.groupby(unique_id_field)
    else:
        grouped = [(None, input_layer)]

    centroids = []

    for group_name, group in grouped:
        if weight_field:
            # Use weighted centroid calculation
            weighted_x = sum(group.geometry.centroid.x * group[weight_field]) / group[weight_field].sum()
            weighted_y = sum(group.geometry.centroid.y * group[weight_field]) / group[weight_field].sum()
            centroid = Point(weighted_x, weighted_y)
        else:
            # Use normal centroid calculation
            centroid = group.geometry.centroid.unary_union.centroid

        # Construct the attribute dictionary of the centroid
        properties = {}
        if unique_id_field:
            properties[unique_id_field] = group_name

        # Create a GeoSeries containing centroids
        centroid_geo = gpd.GeoSeries([centroid], crs=input_layer.crs)
        centroid_geo = centroid_geo.to_frame(name='geometry')
        centroid_geo = gpd.GeoDataFrame(centroid_geo, geometry='geometry')

        for key, value in properties.items():
            centroid_geo[key] = value

        centroids.append(centroid_geo)

    # Merge all centroid points into a GeoDataFrame
    centroids_gdf = pd.concat(centroids, ignore_index=True)

    return centroids_gdf


@call_graph
def nearest_neighbor_analysis(point_layer, output_html):
    """
    Perform nearest neighbor analysis on point layers and generate an HTML report containing statistical values.

    parameter:
    point_layer (GeoDataFrame): point layer.
    output_html (str): Output HTML file path.

    return:
    None
    """

    #Extract the coordinates of the point
    coords = np.array([(point.x, point.y) for point in point_layer.geometry])

    # Create KDTree for fast nearest neighbor search
    tree = KDTree(coords)

    # Find the nearest neighbor distance of each point
    distances, _ = tree.query(coords, k=2)
    nearest_distances = distances[:, 1]

    # Calculate the average nearest neighbor distance
    mean_nn_distance = np.mean(nearest_distances)

    # Calculate the density of points
    area = point_layer.total_bounds
    width = area[2] - area[0]
    height = area[3] - area[1]
    study_area = width * height
    point_density = len(point_layer) / study_area

    # Calculate the expected nearest neighbor distance
    expected_nn_distance = 1 / (2 * np.sqrt(point_density))

    # Calculate R index
    r_index = mean_nn_distance / expected_nn_distance

    #Nearest neighbor analysis results
    if r_index < 1:
        pattern = "Clustered"
    elif r_index > 1:
        pattern = "Dispersed"
    else:
        pattern = "Random"

    # prepare HTML template
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

    # Render HTML template
    template = Template(html_template)
    html_content = template.render(
        num_points=len(point_layer),
        mean_nn_distance=mean_nn_distance,
        point_density=point_density,
        expected_nn_distance=expected_nn_distance,
        r_index=r_index,
        pattern=pattern
    )

    # Write HTML file
    with open(output_html, "w") as f:
        f.write(html_content)


@call_graph
def calculate_line_length_in_polygons(line_layer_path, polygon_layer_path, output_layer_path,
                                      length_field_name='line_length', count_field_name='line_count'):
    # Read line layer and polygon layer
    lines = gpd.read_file(line_layer_path)
    polygons = gpd.read_file(polygon_layer_path)

    # Check whether the coordinate reference system is consistent, if not, convert it
    if lines.crs != polygons.crs:
        lines = lines.to_crs(polygons.crs)

    # Create new fields and initialize them to 0
    polygons[length_field_name] = 0.0
    polygons[count_field_name] = 0

    # Traverse each polygon and calculate the total length and number of internal lines
    for idx, polygon in polygons.iterrows():
        # Get all lines that intersect the current polygon
        intersecting_lines = lines[lines.intersects(polygon.geometry)]

        # Calculate the total length of these lines
        total_length = intersecting_lines.geometry.length.sum()

        # Count the number of these lines
        line_count = len(intersecting_lines)

        # Assign the calculation result to the current polygon
        polygons.at[idx, length_field_name] = total_length
        polygons.at[idx, count_field_name] = line_count

    # Save the results to a new layer file
    polygons.to_file(output_layer_path)


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
