#from main_MIP import *
#from main_APPC import *
#import gurobipy as gp
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point
import json
import matplotlib.pyplot as plt
from concatenation import *
############################################
def plot_electric_network_with_substations(ground_file, air_file, substations_file):
    """
    Trace le réseau électrique en superposant deux types de lignes (aériennes et souterraines) et
    les postes sources en tant que points.

    Args:
        ground_file (str): Chemin vers le fichier CSV contenant les lignes souterraines.
        air_file (str): Chemin vers le fichier CSV contenant les lignes aériennes.
        substations_file (str): Chemin vers le fichier CSV contenant les positions des postes sources.

    Returns:
        (gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame):
        GeoDataFrames pour les lignes souterraines, aériennes et les postes sources.
    """
    def load_and_filter_lines(csv_file, color, label):
        bounding_box = {  # Belle-Ile
            "min_lon": -3.260,
            "max_lon": -3,
            "min_lat": 47.275,
            "max_lat": 47.600
        }

        try:
            data = pd.read_csv(csv_file, sep=",")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier CSV {csv_file} : {e}")

        required_columns = {'Code Commune', 'Geo Point', 'Geo Shape'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Le fichier {csv_file} est invalide. Colonnes manquantes : {missing_columns}")

        geometries = []
        filtered_rows = []
        for _, row in data.iterrows():
            try:
                geo_shape = json.loads(row['Geo Shape'])
                if geo_shape['type'] == 'LineString':
                    line = LineString(geo_shape['coordinates'])
                    if all(
                        bounding_box["min_lon"] <= coord[0] <= bounding_box["max_lon"] and
                        bounding_box["min_lat"] <= coord[1] <= bounding_box["max_lat"]
                        for coord in line.coords
                    ):
                        geometries.append(line)
                        filtered_rows.append(row)
            except Exception as e:
                print(f"Ligne ignorée dans {csv_file}: {row['Geo Shape']}, erreur : {e}")
                continue

        filtered_data = pd.DataFrame(filtered_rows)
        geo_df = gpd.GeoDataFrame(filtered_data, geometry=geometries, crs="EPSG:4326")
        return geo_df

    def load_and_filter_substations(substations_file):
        bounding_box = {  # Belle-Ile
            "min_lon": -3.260,
            "max_lon": -3,
            "min_lat": 47.275,
            "max_lat": 47.600
        }

        try:
            data = pd.read_csv(substations_file, sep=",")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier CSV {substations_file} : {e}")

        required_columns = {'Code Commune', 'Geo Point'}
        if 'Geo Point' not in data.columns:
            raise ValueError(f"Le fichier {substations_file} est invalide. Colonne `Geo Point` manquante.")

        geometries = []
        filtered_rows = []
        for _, row in data.iterrows():
            try:
                if pd.isna(row['Geo Point']) or ',' not in row['Geo Point']:
                    print(f"Ligne ignorée (coordonnées invalides) : {row['Geo Point']}")
                    continue

                lat, lon = map(float, row['Geo Point'].split(","))
                if bounding_box["min_lon"] <= lon <= bounding_box["max_lon"] and bounding_box["min_lat"] <= lat <= bounding_box["max_lat"]:
                    point = Point(lon, lat)
                    geometries.append(point)
                    filtered_rows.append(row)
            except Exception as e:
                print(f"Poste source ignoré : {row['Geo Point']}, erreur : {e}")
                continue

        filtered_data = pd.DataFrame(filtered_rows)
        geo_df = gpd.GeoDataFrame(filtered_data, geometry=geometries, crs="EPSG:4326")
        return geo_df

    ground_geo_df = load_and_filter_lines(ground_file, color="red", label="Lignes souterraines")
    air_geo_df = load_and_filter_lines(air_file, color="blue", label="Lignes aériennes")
    substations_geo_df = load_and_filter_substations(substations_file)

    fig, ax = plt.subplots(figsize=(10, 10))
    ground_geo_df.plot(ax=ax, color="red", linewidth=1, label="Lignes souterraines")
    air_geo_df.plot(ax=ax, color="blue", linewidth=1, label="Lignes aériennes")
    substations_geo_df.plot(ax=ax, color="green", markersize=50, label="Postes HT/BT")

    plt.legend()
    plt.title("Réseau électrique avec postes HTA/BT")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

    return ground_geo_df, air_geo_df, substations_geo_df

#################################################
def find_intersection_points(file1, file2, output_file):
    """
    Trouve les points d'intersection entre deux fichiers CSV basés sur la colonne 'geo_point_2d'
    et génère un nouveau fichier CSV avec les données communes.

    Args:
        file1 (str): Chemin vers le premier fichier CSV.
        file2 (str): Chemin vers le deuxième fichier CSV.
        output_file (str): Chemin pour le fichier CSV de sortie.

    Returns:
        None
    """
    # Charger les fichiers CSV
    df1 = pd.read_csv(file1, sep=";")
    df2 = pd.read_csv(file2, sep=";")

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = {'statut', 'geo_point_2d', 'geo_shape'}
    if not required_columns.issubset(df1.columns) or not required_columns.issubset(df2.columns):
        raise ValueError("Les colonnes 'statut', 'geo_point_2d', 'geo_shape' doivent être présentes dans les deux fichiers.")

    # Trouver les points d'intersection sur la colonne 'geo_point_2d'
    intersection = pd.merge(df1, df2, on='geo_point_2d', suffixes=('_file1', '_file2'))

    # Sélectionner les colonnes d'origine (ajuster si nécessaire pour garder d'autres colonnes pertinentes)
    result = intersection[['statut_file1', 'geo_point_2d', 'geo_shape_file1']].rename(
        columns={'statut_file1': 'statut', 'geo_shape_file1': 'geo_shape'}
    )

    # Sauvegarder le fichier CSV de sortie
    result.to_csv(output_file, index=False)
    print(f"Fichier d'intersection généré : {output_file}")
##################################################
def plot_electric_network(ground_file, air_file):
    """
    Trace le réseau électrique en superposant deux types de lignes :
    - Lignes souterraines (rouge) depuis `ground_file`.
    - Lignes aériennes (bleu) depuis `air_file`.

    Args:
        ground_file (str): Chemin vers le fichier CSV contenant les lignes souterraines.
        air_file (str): Chemin vers le fichier CSV contenant les lignes aériennes.

    Returns:
        (gpd.GeoDataFrame, gpd.GeoDataFrame): Deux GeoDataFrame pour les lignes souterraines et aériennes.
    """
    def load_and_filter(csv_file, color, label):
        """
        Charge un fichier CSV, filtre les géométries pour Ajaccio, et renvoie un GeoDataFrame.
        """
        # Définir la bounding box pour Ajaccio
        # bounding_box = { #Ajaccio
        #     "min_lon": 8.700,
        #     "max_lon": 8.760,
        #     "min_lat": 41.900,
        #     "max_lat": 41.940
        # }

        bounding_box = {  # Belle-Ile
            "min_lon": -3.260,  # Longitude minimale
            "max_lon": -3.150,  # Longitude maximale
            "min_lat": 47.280,  # Latitude minimale
            "max_lat": 47.400  # Latitude maximale
        }

        try:
            # Lecture du fichier CSV
            data = pd.read_csv(csv_file, sep=",")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier CSV {csv_file} : {e}")

        # Vérification des colonnes nécessaires
        required_columns = {'statut', 'geo_point_2d', 'geo_shape'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Le fichier {csv_file} est invalide. Colonnes manquantes : {missing_columns}")

        # Filtrage des géométries dans la bounding box
        geometries = []
        filtered_rows = []
        for _, row in data.iterrows():
            try:
                geo_shape = json.loads(row['geo_shape'])
                if geo_shape['type'] == 'LineString':
                    line = LineString(geo_shape['coordinates'])
                    if all(
                        bounding_box["min_lon"] <= coord[0] <= bounding_box["max_lon"] and
                        bounding_box["min_lat"] <= coord[1] <= bounding_box["max_lat"]
                        for coord in line.coords
                    ):
                        geometries.append(line)
                        filtered_rows.append(row)
                else:
                    raise ValueError("Type de géométrie non pris en charge.")
            except Exception as e:
                print(f"Ligne ignorée dans {csv_file}: {row['geo_shape']}, erreur : {e}")
                continue

        # Création d'un GeoDataFrame
        filtered_data = pd.DataFrame(filtered_rows)
        geo_df = gpd.GeoDataFrame(filtered_data, geometry=geometries, crs="EPSG:4326")
        return geo_df

    # Charger et filtrer les lignes souterraines et aériennes
    ground_geo_df = load_and_filter(ground_file, color="red", label="Lignes souterraines")
    air_geo_df = load_and_filter(air_file, color="blue", label="Lignes aériennes")

    # Tracé des deux types de lignes
    fig, ax = plt.subplots(figsize=(10, 10))
    ground_geo_df.plot(ax=ax, color="red", linewidth=1, label="Lignes souterraines")
    air_geo_df.plot(ax=ax, color="blue", linewidth=1, label="Lignes aériennes")

    # Personnalisation du graphique
    plt.legend()
    plt.title("Réseau électrique d'Ajaccio : Lignes aériennes et souterraines")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

    return ground_geo_df, air_geo_df
###################################################
def csv_to_graph_ajaccio(csv_file):
    """
    Convertit un fichier CSV avec des colonnes `statut`, `geo_point_2D` et `geo_shape` en un graphe géospatial,
    limité au réseau électrique d'Ajaccio.

    Args:
        csv_file (str): Chemin vers le fichier CSV à analyser.

    Returns:
        nx.Graph: Un graphe NetworkX représentant les lignes électriques.
        gpd.GeoDataFrame: Un GeoDataFrame contenant les géométries des lignes pour une visualisation future.
    """
    # Définir la bounding box pour Ajaccio

    # bounding_box = { #Ajaccio
    #     "min_lon": 8.700,
    #     "max_lon": 8.760,
    #     "min_lat": 41.900,
    #     "max_lat": 41.940
    # }

    bounding_box = {  # Belle-Ile
        "min_lon": -3.260,  # Longitude minimale
        "max_lon": -3.150,  # Longitude maximale
        "min_lat": 47.280,  # Latitude minimale
        "max_lat": 47.400  # Latitude maximale
    }

    try:
        # Lecture du fichier CSV avec le bon séparateur
        data = pd.read_csv(csv_file, sep=";")
        print("Colonnes détectées :", data.columns)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier CSV : {e}")

    # Vérification des colonnes nécessaires
    required_columns = {'statut', 'geo_point_2d', 'geo_shape'}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Les colonnes suivantes sont manquantes : {missing_columns}")

    # Création des géométries et filtrage pour Ajaccio
    geometries = []
    filtered_rows = []
    for _, row in data.iterrows():
        try:
            geo_shape = json.loads(row['geo_shape'])
            if geo_shape['type'] == 'LineString':
                line = LineString(geo_shape['coordinates'])

                # Vérifier si la ligne est dans la bounding box
                if all(
                        bounding_box["min_lon"] <= coord[0] <= bounding_box["max_lon"] and
                        bounding_box["min_lat"] <= coord[1] <= bounding_box["max_lat"]
                        for coord in line.coords
                ):
                    geometries.append(line)
                    filtered_rows.append(row)
            else:
                raise ValueError("Type de géométrie non pris en charge.")
        except Exception as e:
            print(f"Ligne ignorée : {row['geo_shape']}, erreur : {e}")
            continue

    # Créer un GeoDataFrame filtré
    filtered_data = pd.DataFrame(filtered_rows)
    geo_df = gpd.GeoDataFrame(filtered_data, geometry=geometries, crs="EPSG:4326")

    # Création du graphe
    graph = nx.Graph()
    for _, row in geo_df.iterrows():
        line = row.geometry
        start_node = tuple(line.coords[0])
        end_node = tuple(line.coords[-1])
        graph.add_node(start_node, statut=row['statut'])
        graph.add_node(end_node, statut=row['statut'])
        graph.add_edge(start_node, end_node, statut=row['statut'])

    # Tracé
    fig, ax = plt.subplots(figsize=(10, 10))
    geo_df.plot(ax=ax, color="blue", linewidth=1, label="Lignes électriques")
    pos = {node: node for node in graph.nodes()}
    #nx.draw(graph, pos, ax=ax, node_size=10, edge_color="red", with_labels=False, label="Graph")
    plt.legend()
    plt.title("Réseau électrique d'Ajaccio")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

    return graph, geo_df
##################################################
if __name__ == '__main__':
    ground_file = "belleile/hta_ground.csv"
    air_file = "belleile/hta_air.csv"
    substations_file = "belleile/postes_hta_bt.csv"
    print(length(substations_file))

    ground_geo_df, air_geo_df, substations_geo_df = plot_electric_network_with_substations(
        ground_file, air_file, substations_file)

    # Save the plot as an SVG
    fig, ax = plt.subplots(figsize=(10, 10))
    ground_geo_df.plot(ax=ax, color="red", linewidth=1, label="Lignes souterraines")
    air_geo_df.plot(ax=ax, color="blue", linewidth=1, label="Lignes aériennes")
    substations_geo_df.plot(ax=ax, color="green", markersize=50, label="Postes HT/BT")

    plt.legend()
    plt.title("Réseau électrique avec postes HTA/BT")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the figure to SVG
    plt.savefig("plot.svg", format="svg")

    # Display the plot in the script (optional)
    plt.show()