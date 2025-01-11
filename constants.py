import os
from typing import Literal

###################### Types ##########################

Granularity = Literal["block", "block_group", "tract", "county"]
GRANULARITY_LIST = ["block", "block_group", "tract", "county"]
State = Literal["LA", "AL"]
STATE_LIST = ["LA", "AL"]
Fips = Literal["22", "01"]
FIPS_LIST = ["22", "01"]
CenterSelectionMethod = Literal[
    "uniform_random",
    "random_iterative",
    "capacitated_random_iterative",
    "uncapacitated_kmeans",
    "random_method",
]
CapacitiesAssignmentMethod = Literal["match", "compute"]
CapacityWeights = Literal["fractional", "voronoi"]
Mode = Literal["partition", "master", "both"]
VapCol = Literal[
    "VAP",  # Total voting age population (P011001)
    "HVAP",  # Hispanic or Latino, any number of races (P011002)
    "WVAP",  # White alone, not hispanic or latino (P011005)
    "BVAP",  # Black or African American alone, not hispanic or latino (P011006)
    "AMINVAP",  # American Indian or Alaska Native alone, not hispanic or latino (P011007)
    "ASIANVAP",  # Asian alone, not hispanic or latino (P011008)
    "NHPIVAP",  # Native Hawaiian or other Pacific Islander alone, not hispanic or latino (P011009)
    "OTHERVAP",  # One other race alone, not hispanic or latino (P011010)
    "2MOREVAP",  # Two or more races, not hispanic or latino (P011011)
]
PopCol = Literal["POP"]  # Total population (P008001)
IPStr = Literal[
    "base", "maj_cvap_approx", "maj_cvap_explicit", "maj_cvap_exact"
]

# def is_granularity  TODO?
# def is_state  TODO?
# def is_fips  TODO?
###################### Paths ##########################

GERRYPY_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "demographics")
CENSUS_SHAPE_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "shapefiles")
# ACS_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "acs_data") #TODO: delete?
OPT_DATA_PATH = os.path.join(
    GERRYPY_BASE_PATH, "data", "optimization_caches"
)  # TODO: delete?
RESULTS_PATH = os.path.join(GERRYPY_BASE_PATH, "results")

###################### Other ##########################

FIPS_DICT = {"LA": "22", "AL": "01"}
COL_DICT_DEC_2010 = {
    "GEO_ID": "GEOID",
    "P008001": "POP",
    "P011001": "VAP",
    "P011002": "HVAP",
    "P011005": "WVAP",
    "P011006": "BVAP",
    "P011007": "AMINVAP",
    "P011008": "ASIANVAP",
    "P011009": "NHPIVAP",
    "P011010": "OTHERVAP",
    "P011011": "2MOREVAP",
}
CENSUS_API_KEY = "e70c2da2298439c24a3bb24f6dd24a03fb30189b"
# https://api.census.gov/data/2010/dec/sf1?get=GEO_ID,P008001,P011001,P011002,P011005,P011006,P011007,P011008,P011009,P011010,P011011&for=block%20group:*&in=county:*&in=state:22&key=e70c2da2298439c24a3bb24f6dd24a03fb30189b


def flatten(lis_of_lis):
    return [element for lis in lis_of_lis for element in lis]
