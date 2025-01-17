from pyproj import Transformer

def georeferecePoints(x1,y1,x2,y2,x3,y3,x4,y4,imagePath):
    try:
        with open(imagePath.replace('jpg', 'jgw'), 'r') as jgwFile:
            lines = jgwFile.readlines()
        pixelSizeX = float(lines[0].strip())
        pixelSizeY = float(lines[3].strip())
        topLeftXGeo = float(lines[4].strip())
        topLeftYGeo = float(lines[5].strip())
        x1 = topLeftXGeo + pixelSizeX * x1
        y1 = topLeftYGeo + pixelSizeY * y1
        x2 = topLeftXGeo + pixelSizeX * x2
        y2 = topLeftYGeo + pixelSizeY * y2
        x3 = topLeftXGeo + pixelSizeX * x3
        y3 = topLeftYGeo + pixelSizeY * y3
        x4 = topLeftXGeo + pixelSizeX * x4
        y4 = topLeftYGeo + pixelSizeY * y4
        return([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
    except Exception as e:
        print(f"There was an issue with opening {imagePath.replace('jpg', 'jgw')}: {e}")
        return(None)

def BNGtoLatLong(listOfPoints):
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    longLatList = []
    for xBNG, yBNG in listOfPoints:
        long, lat = transformer.transform(xBNG, yBNG)
        longLatList.append((long, lat))
    return longLatList