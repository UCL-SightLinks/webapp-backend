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
    geotransform = croppedTifImage.GetGeoTransform()
    if not geotransform:
        raise ValueError(f"No geotransform found in the file: {croppedTifImage}")
    projection = croppedTifImage.GetProjection()

    # Convert pixel coordinates to real-world coordinates in the file's CRS
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    x1 = origin_x + x1 * pixel_width + y1 * geotransform[2]
    y1 = origin_y + x1 * geotransform[4] + y1 * pixel_height
    x2 = origin_x + x2 * pixel_width + y2 * geotransform[2]
    y2 = origin_y + x2 * geotransform[4] + y2 * pixel_height
    x3 = origin_x + x3 * pixel_width + y3 * geotransform[2]
    y3 = origin_y + x3 * geotransform[4] + y3 * pixel_height
    x4 = origin_x + x4 * pixel_width + y4 * geotransform[2]
    y4 = origin_y + x4 * geotransform[4] + y4 * pixel_height

    # Set up transformation to WGS84 (latitude/longitude)
    source_crs = osr.SpatialReference()
    source_crs.ImportFromWkt(projection)

    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(4326)  # WGS84 EPSG code

    transform = osr.CoordinateTransformation(source_crs, target_crs)

    outputList = []
    lat1, lon1, _ = transform.TransformPoint(x1, y1)
    lat2, lon2, _ = transform.TransformPoint(x2, y2)
    lat3, lon3, _ = transform.TransformPoint(x3, y3)
    lat4, lon4, _ = transform.TransformPoint(x4, y4)
    outputList.append((lat1, lon1))
    outputList.append((lat2, lon2))
    outputList.append((lat3, lon3))
    outputList.append((lat4, lon4))

    return outputList