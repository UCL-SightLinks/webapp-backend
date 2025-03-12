import folium
import json
import os

def generateMap(jsonFile, outputFile):
    """
    Generate a folium map with polygons and confidence markers from the given JSON data.
    The JSON file has to be in the same format as the output of our main function. 
    The outputFile should end with .html

    """
    
    # Load JSON data
    with open(jsonFile, "r") as f:
        data = json.load(f)

    mapCenter = data[0]["coordinates"][0][0]
    # Create a folium map centered at the provided mapCenter
    m = folium.Map(location=mapCenter, zoom_start=16)
    # Extract coordinates and confidence values
    coordinates = data[0]["coordinates"]
    confidenceValues = data[0]["confidence"]

    # Add polygons for the squares and confidence markers
    for i, (square, conf) in enumerate(zip(coordinates, confidenceValues)):
        folium.Polygon(
            locations=square,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.4,
            popup=f"Confidence: {conf}",
        ).add_to(m)
        
        # Add a marker at the center of each square
        centerLat = sum(point[0] for point in square) / len(square)
        centerLon = sum(point[1] for point in square) / len(square)
        folium.Marker(
            location=[centerLat, centerLon],
            popup=f"Confidence: {conf}",
            icon=folium.Icon(color="blue"),
        ).add_to(m)

    # Save the map to the specified outputFile
    m.save(outputFile)
    print(f"Map saved as {outputFile}")
