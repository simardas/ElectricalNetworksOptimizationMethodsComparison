# Fonction pour extraire les coordonnées d'une ligne
def extract_coordinates(row):
    try:
        geo_data = json.loads(row["geo_shape"])
        return geo_data["coordinates"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return []

def merge_continuous_lines(df):
    visited = set()
    merged_lines = []

    def find_and_merge(start_index, current_line):
        if start_index in visited:
            return
        visited.add(start_index)
        coordinates = extract_coordinates(df.iloc[start_index])
        if coordinates:
            # Ajoute les coordonnées actuelles
            current_line.extend(coordinates)
            # Vérifie les connexions aux autres lignes
            for i, row in df.iterrows():
                if i not in visited:
                    other_coords = extract_coordinates(row)
                    if other_coords:
                        # Vérifie si les lignes partagent des points d'extrémité
                        if tuple(coordinates[-1]) == tuple(other_coords[0]):
                            find_and_merge(i, current_line)
                        elif tuple(coordinates[0]) == tuple(other_coords[-1]):
                            find_and_merge(i, current_line[::-1])  # Inverser si connexion à l'autre extrémité

    for i, row in df.iterrows():
        if i not in visited:
            line = []
            find_and_merge(i, line)
            if line:
                merged_lines.append(line)

    return merged_lines
