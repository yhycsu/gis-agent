import os

import pandas as pd
import requests
import json
from functools import wraps
import io

import folium
import networkx as nx
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from pyvis.network import Network
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely import geometry
from tqdm import tqdm
from shapely.geometry import Point, LineString


# Define a directed graph to represent function call relationships
graph = nx.DiGraph()
# Maintain a call stack to track nested function calls
call_stack = []
# Directory to save generated images
img_dir = 'images'

# Color scheme for different node types in the call graph
node_colors = {
    'input': 'lime',
    'tool': 'deepskyblue',
    'result': 'blueviolet',
    'data': 'coral'
}
# Predefined node types for clarity
node_types = ['Data(input)', 'Tool', 'Data(result)', 'Data(intermediate)']


def gcj02_to_wgs84_one(lng, lat):
    """
    Converts coordinates from the GCJ-02 (China Geodetic Coordinate System) to WGS-84 (World Geodetic System 1984).

    Args:
        lng (float): Longitude in GCJ-02 coordinates.
        lat (float): Latitude in GCJ-02 coordinates.

    Returns:
        tuple[float, float]: Converted longitude and latitude in WGS-84 coordinates.
    """
    # Constants for coordinate transformation calculations
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    pi = 3.1415926535897932384626  # π
    a = 6378245.0  # Earth's semi-major axis
    ee = 0.00669342162296594323  # Earth's eccentricity squared

    def _transformlng(lng, lat):
        """
        Calculates the transformed longitude based on GCJ-02 and WGS-84 parameters.

        Args:
            lng (float): Longitude in GCJ-02 coordinates.
            lat (float): Latitude in GCJ-02 coordinates.

        Returns:
            float: Transformed longitude in WGS-84 coordinates.
        """
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
        """
        Calculates the transformed latitude based on GCJ-02 and WGS-84 parameters.

        Args:
            lng (float): Longitude in GCJ-02 coordinates.
            lat (float): Latitude in GCJ-02 coordinates.

        Returns:
            float: Transformed latitude in WGS-84 coordinates.
        """
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


def download_poi(keywords, region):
    """Downloads POIs (Points of Interest) from AMap API and saves them as a GeoDataFrame.

    Args:
        keywords (str, optional): Keywords to search for POIs. Defaults to ''.
        region (str, optional): Region to search in. Defaults to '长沙市' (Changsha City).

    Returns:
        None
    """
    results = []  # List to store retrieved POI data
    for i in range(1, 21):
        url = f'''https://restapi.amap.com/v5/place/text?keywords={keywords}
            &region={region}&key=016ad9192e40fc6d1054a83a21400429&page_num={i}'''
        res = requests.get(url)
        json_data = json.loads(res.text)
        pois = json_data['pois']
        results += pois

    for poi in results:
        location = poi['location'].split(',')
        lon, lat = gcj02_to_wgs84_one(eval(location[0]), eval(location[1]))
        poi['geometry'] = geometry.Point(lon, lat)

    gdf = gpd.GeoDataFrame(pois)
    if not os.path.exists(os.path.join(f'../shapefiles/poi', f'{keywords}_{region}')):
        os.mkdir(os.path.join(f'../shapefiles/poi', f'{keywords}_{region}'))

    gdf.to_file(os.path.join(f'../shapefiles/poi/{keywords}_{region}', f'{keywords}_{region}.shp'), encoding='utf-8')


def add_point(map_, row):
    """
    Adds a point marker to the Folium map.

    Args:
        map_ (folium.Map): The Folium map object.
        row (pandas.Series): A row from a pandas DataFrame containing geometry data.

    Returns:
        None
    """
    geometry = row.geometry  # Extract geometry data from the row
    # Create a Folium marker with coordinates from the geometry
    folium.Marker(
        location=[geometry.y, geometry.x],
        popup=row['name'],  # Set popup text from the row's 'name' attribute
        icon=folium.Icon("red"),  # Set icon color to red
    ).add_to(map_)


def add_polygon(map_, row):
    """
    Adds a polygon shape to the Folium map.

    Args:
        map_ (folium.Map): The Folium map object.
        row (pandas.Series): A row from a pandas DataFrame containing geometry data.

    Returns:
        None
    """
    x, y = row.geometry.exterior.coords.xy  # Extract coordinates from the geometry
    x = x.tolist()  # Convert coordinates to lists
    y = y.tolist()

    coords = [[y[i], x[i]] for i in range(len(x))]  # Create list of coordinate pairs
    # Create a Folium polygon with specified color, weight, and fill properties
    folium.Polygon(
        locations=coords,
        color="blue",
        weight=6,
        fill_color="red",
        fill_opacity=0.5,
        fill=True,
    ).add_to(map_)


def add_line(map_, row):
    """
    Adds a line to the Folium map.

    Args:
        map_ (folium.Map): The Folium map object.
        row (pandas.Series): A row from a pandas DataFrame containing geometry data.

    Returns:
        None
    """
    x, y = row.geometry.xy  # Extract coordinates from the geometry
    x = x.tolist()  # Convert coordinates to lists
    y = y.tolist()

    coords = [[y[i], x[i]] for i in range(len(x))]  # Create list of coordinate pairs
    # Create a Folium PolyLine with specified color and weight
    folium.PolyLine(
        locations=coords,
        color="#FF0000",
        weight=5,
    ).add_to(map_)


def plot_shp(gdf, func_name):
    """
    Creates a Matplotlib plot of a GeoPandas DataFrame and saves it as a PNG image.

    Args:
        gdf (GeoPandas.GeoDataFrame): The GeoPandas DataFrame with geospatial data.
        func_name (str): The name of the function that generated the data (for image naming).

    Returns:
        str: The filename of the saved image.
    """
    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the GeoPandas DataFrame on the axes using the 'plot' method
    gdf.plot(ax=ax)

    # Create the image filename with the function name and '.png' extension
    fig_name = os.path.join(img_dir, f'{func_name}.png')
    # Save the figure as a PNG image with high resolution (300 dpi)
    plt.savefig(fig_name, dpi=300)
    return fig_name


def display_in_map(result):
    """
    Creates a Folium map displaying points, polygons, or lines based on geometry data.

    Args:
        result (pandas.DataFrame): A pandas DataFrame containing rows with geometry data.

    Returns:
        str: The HTML representation of the Folium map.
    """

    tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'
    # Create a Folium map with specified center coordinates, tiles, and zoom level
    m = folium.Map(
        location=[28.2278, 112.9389],
        tiles=tiles,
        attr='高德-常规图',
        zoom_start=10
    )

    # Iterate through each row in the DataFrame
    for i in range(len(result)):
        try:
            # Get the current row from the DataFrame
            row = result.iloc[i]

            # Check the geometry type of the current row
            if row.geometry.geom_type == 'Polygon':
                # If it's a polygon, add it to the map using the `add_polygon` function
                add_polygon(m, row)
            elif row.geometry.geom_type == 'Point':
                # If it's a point, add it to the map using the `add_point` function
                add_point(m, row)
            elif row.geometry.geom_type == 'LineString':
                # If it's a line, add it to the map using the `add_line` function
                add_line(m, row)
        except:
            # Handle any exceptions that may occur
            continue

    # Save the Folium map to an HTML file named "index.html"
    m.save("index.html")

    # Return the HTML representation of the map
    return m._repr_html_()


def call_graph(func):
    """
    Decorator function that wraps a function and tracks its call stack and call graph.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the call stack is empty, indicating that the function is being called
        # directly, not nested within another function
        if call_stack:
            return func(*args, **kwargs)

        # Get function name, number of nodes, and number of keyword arguments
        func_name = func.__name__
        n = graph.number_of_nodes()
        num_params = len(kwargs)
        # Assign a unique ID to the current function node
        function_id = n
        # Add the function node with its name and type ('tool') to the call graph
        graph.add_node(function_id, name=func_name, node_type='tool')

        # Append the function name to the call stack
        call_stack.append(func_name)
        # Create a list to track whether each parameter has been processed
        add_param = [True for i in range(num_params)]

        def create_link(arg, i):
            """
            Helper function to handle argument processing.

            Args:
                arg: The argument to be processed.
                i: The index of the argument.

            Returns:
                The processed argument or its corresponding node ID.
            """
            # If the argument is a tuple (meaning it's already linked in a previous call),
            # extract the node ID and update the `add_param` list
            if isinstance(arg, tuple):
                graph.add_edge(arg[1], function_id)  # Add edge between linked node and function
                add_param[i] = False  # Mark parameter as processed
                return arg[0]  # Return the actual value

            # Otherwise, return the argument as is
            return arg

        # Process keyword arguments
        kwargs = {key: create_link(value, i) for i, (key, value) in enumerate(kwargs.items())}
        # Execute the decorated function and store the result
        result = func(*args, **kwargs)

        # After function execution, add nodes for inputs, results, and connect them

        # 1. Add input nodes (data nodes) for unprocessed keyword arguments
        i = 1  # Counter for input node IDs
        for s, (k, v) in enumerate(kwargs.items()):
            if add_param[s]:  # Check if parameter hasn't been processed
                graph.add_node(n + i, name=k, node_type='data')
                i += 1

        # 2. Connect function node to input nodes
        j = 1  # Counter for connecting edges
        for s, (k, v) in enumerate(kwargs.items()):
            if add_param[s]:
                graph.add_edge(n + j, n)
                j += 1
        # 3. Add result node (data node) for the function's output
        graph.add_node(n + i, name=f'{func_name}_result', node_type='data')
        graph.add_edge(n, n + i)
        call_stack.pop()

        return result, n + i

    return wrapper


def graph_to_image():
    """
    Converts a function call graph (represented by a networkx graph object) into a PIL image (.png) for visualization.

    :return: The PIL image representing the function call graph.
    """
    draw_call_graph()
    # Convert Matplotlib figure to PIL image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image = Image.open(buffer)
    return image


def draw_call_graph():
    """
    Generates the visual representation of the function call graph using Matplotlib and NetworkX.

    :return: None
    """
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
    legend_handles = [plt.Line2D([], [], color=color, label=label) for color, label in
                      zip(node_colors.values(), node_types)]
    # 绘制图例，并设置背景颜色
    legend = plt.legend(handles=legend_handles, fontsize=15)
    legend.get_frame().set_facecolor('lightgrey')  # 设置图例背景颜色为浅灰色
    # plt.show()
    # plt.savefig('graph.png')


def get_call_graph():
    return graph


def read_html_file(file_path):
    """
    Reads the content of an HTML file and returns it as a string.

    :param file_path: Path to the HTML file.
    :return: The HTML file content as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        return html_content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def call_ollama_api(payload, api_endpoint="http://localhost:11434/api/generate"):
    """
    Makes a POST request to an OLlama API endpoint and returns the JSON response.

    :param api_endpoint: The URL of the OLlama API endpoint.
    :param payload: The data to send to the API in JSON format.
    :return: The JSON response from the API or None if an error occurs.
    """
    response = requests.post(api_endpoint, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()  # assuming the API returns JSON
    else:
        # remove brackets
        print(response.text[1:-1])
        raise ValueError


def get_public_ip():
    """
    get the user's ip address
    """
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


def get_location_from_ip(ip):
    """
    Get the location of the user's ip address'
    """
    url = f'https://restapi.amap.com/v3/ip?ip={ip}&key=016ad9192e40fc6d1054a83a21400429'
    res = requests.get(url)
    json_data = json.loads(res.text)
    return json_data['city']


def find_sink_node(g):
    """
    Find the sink node in a NetworkX directed graph.

    :param g: A NetworkX directed graph
    :return: The sink node, or None if not found
    """
    sinks = []
    for node in g.nodes():
        if g.out_degree(node) == 0 and g.in_degree(node) > 0:
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
        if node['node_type'] == 'data':
            if node['label'] in sinks:
                node_colors.append('violet')
            elif node['label'] in sources:
                node_colors.append('lightgreen')
            else:
                node_colors.append('orange')

        elif node['node_type'] == 'tool':
            node_colors.append('deepskyblue')

    # Update node colorsb
    for i, color in enumerate(node_colors):
        nt.nodes[i]['color'] = color
        nt.nodes[i]['label'] = node_names[i]
        nt.nodes[i]['shape'] = 'dot'

    nt.show('graph.html')
    return read_html_file('graph.html')


def km_to_degrees(distance_km):
    """
    Converts a distance in kilometers (km) to its corresponding angular measure in degrees (°).

    :param distance_km: The distance to convert, in kilometers.
    :return: The equivalent distance in degrees.
    """
    # Earth's equatorial circumference approximately 40,075 kilometers
    earth_circumference_km = 40075
    # Approximately 111 kilometers per degree along the equator
    km_per_degree = earth_circumference_km / 360
    # Calculate the corresponding angular measure (degrees)
    degrees = distance_km / km_per_degree
    return degrees


def clear_graph():
    """
    Resets the global graph and call stack variables to their initial states.
    """
    global graph
    global call_stack
    graph = nx.DiGraph()
    call_stack = []


class OsmDownloader:

    def __init__(self, area):
        """
        Initialize the `OsmDownloader` class with the target area.

        Args:
            area (str): The target area for data download.
        """
        self.area = area

    def __search_for_osm_id__(self):
        """
        Retrieves the OSM ID for the specified area using the Nominatim API.

        Returns:
            list: A list of matching area information.
        """
        # Send search request
        html = requests.get(f'https://nominatim.openstreetmap.org/search?format=json&q={self.area}')

        # Extract relevant city parameter information
        return [item for item in eval(html.text) if item['type'] == 'administrative']

    def __download__(self):
        """
        Downloads OSM data for the specified area and extracts nodes and ways.
        """
        self.search_result = self.__search_for_osm_id__()
        self.area_id = int(self.search_result[0]['osm_id'] + 36e8)

        url_front = 'https://overpass.kumi.systems/api/interpreter'
        url_parameters = {'data': f'[timeout:900][maxsize:1073741824][out:json];area({self.area_id});(._; )->.area;(way[highway](area.area); node(w););out skel;'}

        source = requests.post(url_front, data=url_parameters)

        raw_osm_json = eval(source.text)

        # Extract point layer and save
        points_contents = []
        for element in tqdm(raw_osm_json['elements'], desc=f'[{self.area}] Extracting point data'):
            if element['type'] == 'node':
                points_contents.append((str(element['id']), element['lon'], element['lat']))

        self.points = pd.DataFrame(points_contents, columns=['id', 'lng', 'lat'])

        self.points['geometry'] = self.points.apply(lambda row: Point([row['lng'], row['lat']]), axis=1)

        self.points = GeoDataFrame(self.points, crs='EPSG:4326')

        # Create id-to-point data dictionary
        self.id2points = {key: value for key, value in zip(self.points['id'], self.points['geometry'])}

        # Save line layer
        ways_contents = []
        for element in tqdm(raw_osm_json['elements'], desc=f'[{self.area}] Extracting line data'):
            if element['type'] == 'way':
                if element['nodes'].__len__() >= 2:
                    ways_contents.append((str(element['id']), LineString([self.id2points[str(_)]
                                                                          for _ in element['nodes']])))

        self.ways = gpd.GeoDataFrame(pd.DataFrame(ways_contents, columns=['id', 'geometry']),
                                     crs='EPSG:4326')

    def download_shapefile(self, path=''):
        """
        Downloads OSM data as shapefiles (nodes and ways) to the specified path.

        Args:
            path (str, optional): The path to the output directory. Defaults to the current directory.
        """
        try:
            self.search_result
        except AttributeError:
            print('=' * 200)
            print('Downloading data...')
            self.__download__()
            print('=' * 200)
            print('Data download complete!')

        print('=' * 200)
        print('Exporting data...')
        self.points.to_file(os.path.join(path, f'{self.area}_osm路网'), layer='节点')
        self.ways.to_file(os.path.join(path, f'{self.area}_osm路网'), layer='道路')
        print('=' * 200)
        print('Data export successful!')

    def download_geojson(self, path=''):
        """
        Downloads OSM data as GeoJSON files (nodes and ways) to the specified path.

        Args:
            path (str, optional): The path to the output directory. Defaults to the current directory.
        """
        try:
            self.search_result  # Check if data has been downloaded
        except AttributeError:
            print('=' * 200)
            print('Downloading data...')
            self.__download__()  # Download data if not already done
            print('=' * 200)
            print('Data download complete!')

        print('=' * 200)
        print('Exporting data...')

        # Export points as GeoJSON
        self.points.to_file(os.path.join(path, f'{self.area}_osm路网_节点.json'), driver='GeoJSON')

        # Export ways as GeoJSON
        self.ways.to_file(os.path.join(path, f'{self.area}_osm路网_道路.json'), driver='GeoJSON')

        print('=' * 200)
        print('Data export successful!')
