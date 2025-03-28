from osgeo import osr

def georefereceJGW(x1,y1,x2,y2,x3,y3,x4,y4,pixelSizeX,pixelSizeY,topLeftXGeo,topLeftYGeo):
    """
    This function will receive all of the x y points from a bounding box, their respective pixel sizes, and the
    georeferencing data of the image's top left corner.

    Using this information, it will calculate the coordinates of the four corners of the bounding box.

    Args:
        x(i) (int): The x pixel location of the ith bounding box corner in the image.
        y(i) (int): The y pixel location of the ith bounding box corner in the image.
        pixelSizeX (int): The real-world width of each pixel
        pixelSizeY (int): The real-world height of each pixel
        topLeftXGeo (int): The real-world x-coordinate of the top left pixel in the image (In our case it is easting and northings)
        topLeftYGeo (int): The real-world y-coordinate of the top left pixel in the image (In our case it is easting and northings)
    
    Returns:
        list: It is a list of coordinates for each corner of the box.
    """
    x1 = topLeftXGeo + pixelSizeX * x1
    y1 = topLeftYGeo + pixelSizeY * y1
    x2 = topLeftXGeo + pixelSizeX * x2
    y2 = topLeftYGeo + pixelSizeY * y2
    x3 = topLeftXGeo + pixelSizeX * x3
    y3 = topLeftYGeo + pixelSizeY * y3
    x4 = topLeftXGeo + pixelSizeX * x4
    y4 = topLeftYGeo + pixelSizeY * y4
    return([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])


def BNGtoLatLong(listOfPoints):
    """
    This function takes a list of coordinates, representing a corner of a bounding box in BNG. It will then convert it
    to latitude and longitude.

    Args:
        listOfPoints (list): It is a list of coordinates for each corner of the box in BNG.
    
    Returns:
        latLongList (list): It is a list of latitude and longitudes, representing the corners of a bounding box.
    """
    bng = osr.SpatialReference()
    bng.ImportFromEPSG(27700)
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(bng, wgs84)
    latLongList = []
    for xBNG, yBNG in listOfPoints:
        lat, long, _ = transform.TransformPoint(xBNG, yBNG)
        latLongList.append((lat, long))
    return latLongList



def georeferenceTIF(croppedTifImage, x1,y1,x2,y2,x3,y3,x4,y4):
    """
    This function takes a croppedTifImage, and the four corners of a bounding box. It will then use the data stored in the
    TIF image to find the coordinates of the corners, and it is then converted to latitude and longitude.

    Args:
        croppedTifImage (tif object): It is a tif image which was already cropped and stored in memory.
        x(i) (int): The x pixel location of the ith bounding box corner in the image.
        y(i) (int): The y pixel location of the ith bounding box corner in the image.
    
    Returns:
        outputList (list): It is a list of coordinates for each corner of the box.
    """
    print(f"\n=== Georeferencing Detection Points ===")
    print(f"Input points (pixel coordinates):")
    print(f"Point 1: ({x1}, {y1})")
    print(f"Point 2: ({x2}, {y2})")
    print(f"Point 3: ({x3}, {y3})")
    print(f"Point 4: ({x4}, {y4})")
    
    geotransform = croppedTifImage.GetGeoTransform()
    if not geotransform:
        raise ValueError(f"No geotransform found in the file: {croppedTifImage}")
        
    projection = croppedTifImage.GetProjection()
    print(f"Geotransform: {geotransform}")
    print(f"Projection: {projection}")

    # Convert pixel coordinates to real-world coordinates in the file's CRS
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    print(f"Origin: ({origin_x}, {origin_y})")
    print(f"Pixel size: width={pixel_width}, height={pixel_height}")

    # More accurate calculation of world coordinates
    x1_world = origin_x + x1 * pixel_width
    y1_world = origin_y + y1 * pixel_height
    x2_world = origin_x + x2 * pixel_width
    y2_world = origin_y + y2 * pixel_height
    x3_world = origin_x + x3 * pixel_width
    y3_world = origin_y + y3 * pixel_height
    x4_world = origin_x + x4 * pixel_width
    y4_world = origin_y + y4 * pixel_height
    
    print(f"World coordinates:")
    print(f"Point 1: ({x1_world}, {y1_world})")
    print(f"Point 2: ({x2_world}, {y2_world})")
    print(f"Point 3: ({x3_world}, {y3_world})")
    print(f"Point 4: ({x4_world}, {y4_world})")

    # Set up transformation to WGS84 (latitude/longitude)
    source_crs = osr.SpatialReference()
    source_crs.ImportFromWkt(projection)

    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(4326)  # WGS84 EPSG code
    
    # For older versions of GDAL
    source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(source_crs, target_crs)

    outputList = []
    try:
        # Transform all points
        lat1, lon1, _ = transform.TransformPoint(x1_world, y1_world)
        lat2, lon2, _ = transform.TransformPoint(x2_world, y2_world)
        lat3, lon3, _ = transform.TransformPoint(x3_world, y3_world)
        lat4, lon4, _ = transform.TransformPoint(x4_world, y4_world)
        
        outputList.append((lat1, lon1))
        outputList.append((lat2, lon2))
        outputList.append((lat3, lon3))
        outputList.append((lat4, lon4))
        
        print(f"WGS84 coordinates (lat, lon):")
        print(f"Point 1: ({lat1}, {lon1})")
        print(f"Point 2: ({lat2}, {lon2})")
        print(f"Point 3: ({lat3}, {lon3})")
        print(f"Point 4: ({lat4}, {lon4})")
    except Exception as e:
        print(f"Error during coordinate transformation: {str(e)}")
        # Fallback: if transformation fails, try using the world coordinates directly
        print(f"Falling back to world coordinates without transformation")
        outputList.append((x1_world, y1_world))
        outputList.append((x2_world, y2_world))
        outputList.append((x3_world, y3_world))
        outputList.append((x4_world, y4_world))

    return outputList