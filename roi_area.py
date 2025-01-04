from geographiclib.geodesic import Geodesic

# Define the polygon coordinates (in latitude, longitude order)
coordinates = [
    [40.181759, 8.629761],
    [40.194348, 8.702888],
    [40.14004, 8.736877],
    [40.098559, 8.651047],
    [40.130591, 8.570709],
    [40.171004, 8.598862],
    [40.181759, 8.629761]
]

# Initialize the WGS84 geodesic calculator
geod = Geodesic.WGS84

# Start a polygon calculation
polygon = geod.Polygon()

# Add each point to the polygon
for lat, lon in coordinates:
    polygon.AddPoint(lat, lon)

# Finalize and compute the polygon's area and perimeter
num, perimeter, area = polygon.Compute()

# Area is in square meters, convert to square kilometers
area_sqkm = abs(area) / 1e6

print(f"Area of the polygon: {abs(area):.2f} square meters ({area_sqkm:.4f} square kilometers)")
