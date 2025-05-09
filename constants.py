import os
from typing import Literal
import numpy as np


################################## Types #######################################

State = Literal[
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

Fips = Literal[
    "01",
    "02",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
]

Granularity = Literal["block", "block_group", "tract", "county"]

CenterSelectionMethod = Literal[
    "uniform_random",
    "random_iterative",
    "capacitated_random_iterative",
    "uncapacitated_kmeans",
    "random_method",
]

CapacitiesAssignmentMethod = Literal["match", "compute"]

CapacityWeights = Literal["fractional", "voronoi"]

VapCol = Literal[
    "VAP",  # Total voting age population (P011001)
    "HVAP",  # Hispanic or Latino, any number of races (P011002)
    "WVAP",  # White alone, not hispanic or latino (P011005)
    "BVAP",  # Black or African American alone, not hispanic or latino
    # (P011006)
    "AMINVAP",  # American Indian or Alaska Native alone, not hispanic or
    # latino (P011007)
    "ASIANVAP",  # Asian alone, not hispanic or latino (P011008)
    "NHPIVAP",  # Native Hawaiian or other Pacific Islander alone, not
    # hispanic or latino (P011009)
    "OTHERVAP",  # One other race alone, not hispanic or latino (P011010)
    "2MOREVAP",  # Two or more races, not hispanic or latino (P011011)
    "POCVAP",  # VAP - WVAP
]

PopCol = Literal["POP"]  # Total population (P008001)

IPStr = Literal[
    "base", "maj_cvap_approx", "maj_cvap_explicit", "maj_cvap_exact"
]

PriorityFuncStr = Literal["average_geq_threshold_excess"]

PartitionFuncStr = Literal["maj_cvap_ip"]

CguMapFunc = Literal[
    "solid_colored",
    "cvap_prop_shaded",
    "pop_prop_shaded",
    "six_colored",
    "show_disconnected",
]

DistrictMapFunc = Literal[
    "colored_maj_cvap_outlined",
    "colored_cvap_prop_shaded",
    "cvap_prop_grayscale",
    "districts_colored_hashed_cgus_shaded",
]

ComparativeDistrictMapFunc = Literal[
    "colored_cvap_prop_shaded", "maj_cvap_outlined_cvap_prop_shaded_cgus"
]

Interpretation = Literal["any_part", "single"]

############################ State Information #################################

FIPS_DICT = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

SEATS_DICT = {
    "AL": {"house": 7, "state_senate": 35, "state_house": 105},
    "AK": {"house": 1, "state_senate": 20, "state_house": 40},
    "AZ": {"house": 9, "state_senate": 30, "state_house": 60},
    "AR": {"house": 4, "state_senate": 35, "state_house": 100},
    "CA": {"house": 53, "state_senate": 40, "state_house": 80},
    "CO": {"house": 7, "state_senate": 35, "state_house": 65},
    "CT": {"house": 5, "state_senate": 36, "state_house": 151},
    "DE": {"house": 1, "state_senate": 21, "state_house": 41},
    "FL": {"house": 27, "state_senate": 40, "state_house": 120},
    "GA": {"house": 14, "state_senate": 56, "state_house": 180},
    "HI": {"house": 2, "state_senate": 25, "state_house": 51},
    "ID": {"house": 2, "state_senate": 35, "state_house": 70},
    "IL": {"house": 18, "state_senate": 59, "state_house": 118},
    "IN": {"house": 9, "state_senate": 50, "state_house": 100},
    "IA": {"house": 4, "state_senate": 50, "state_house": 100},
    "KS": {"house": 4, "state_senate": 40, "state_house": 125},
    "KY": {"house": 6, "state_senate": 38, "state_house": 100},
    "LA": {"house": 6, "state_senate": 39, "state_house": 105},
    "ME": {"house": 2, "state_senate": 35, "state_house": 151},
    "MD": {"house": 8, "state_senate": 47, "state_house": 141},
    "MA": {"house": 9, "state_senate": 40, "state_house": 160},
    "MI": {"house": 14, "state_senate": 38, "state_house": 110},
    "MN": {"house": 8, "state_senate": 67, "state_house": 134},
    "MS": {"house": 4, "state_senate": 52, "state_house": 122},
    "MO": {"house": 8, "state_senate": 34, "state_house": 163},
    "MT": {"house": 1, "state_senate": 50, "state_house": 100},
    "NE": {"house": 3, "state_senate": 49, "state_house": 0},
    "NV": {"house": 4, "state_senate": 21, "state_house": 42},
    "NH": {"house": 2, "state_senate": 24, "state_house": 400},
    "NJ": {"house": 12, "state_senate": 40, "state_house": 80},
    "NM": {"house": 3, "state_senate": 42, "state_house": 70},
    "NY": {"house": 27, "state_senate": 63, "state_house": 150},
    "NC": {"house": 13, "state_senate": 50, "state_house": 120},
    "ND": {"house": 1, "state_senate": 47, "state_house": 94},
    "OH": {"house": 16, "state_senate": 33, "state_house": 99},
    "OK": {"house": 5, "state_senate": 48, "state_house": 101},
    "OR": {"house": 5, "state_senate": 30, "state_house": 60},
    "PA": {"house": 18, "state_senate": 50, "state_house": 203},
    "RI": {"house": 2, "state_senate": 38, "state_house": 75},
    "SC": {"house": 7, "state_senate": 46, "state_house": 124},
    "SD": {"house": 1, "state_senate": 35, "state_house": 70},
    "TN": {"house": 9, "state_senate": 33, "state_house": 99},
    "TX": {"house": 36, "state_senate": 31, "state_house": 150},
    "UT": {"house": 4, "state_senate": 29, "state_house": 75},
    "VT": {"house": 1, "state_senate": 30, "state_house": 150},
    "VA": {"house": 11, "state_senate": 40, "state_house": 100},
    "WA": {"house": 10, "state_senate": 49, "state_house": 98},
    "WV": {"house": 3, "state_senate": 34, "state_house": 100},
    "WI": {"house": 8, "state_senate": 33, "state_house": 99},
    "WY": {"house": 1, "state_senate": 30, "state_house": 60},
}

################################## Paths #######################################

GERRYPY_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "demographics")
CENSUS_SHAPE_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "shapefiles")
OPT_DATA_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "optimization_caches")
SEED_PATH = os.path.join(GERRYPY_BASE_PATH, "data", "seeds")
RESULTS_PATH = os.path.join(GERRYPY_BASE_PATH, "results")

###################### Census Data Query Inforamtion ###########################

CENSUS_API_KEY = "e70c2da2298439c24a3bb24f6dd24a03fb30189b"

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

COL_DICT_DEC_2020_ANY_PART = {  # 2020/dec/pl
    "GEOID": ["GEO_ID"],
    "POP": ["P1_001N"],
    "VAP": ["P3_001N"],
    "HVAP": ["P4_002N"],
    "WVAP": ["P3_003N"],  # 1
    "BVAP": ["P3_004N"]  # 1
    + ["P3_011N", "P3_016N", "P3_017N", "P3_018N", "P3_019N"]  # 2
    + [
        "P3_027N",
        "P3_028N",
        "P3_029N",
        "P3_030N",
        "P3_037N",
        "P3_038N",
        "P3_039N",
        "P3_040N",
        "P3_041N",
        "P3_042N",
    ]  # 3
    + [
        "P3_048N",
        "P3_049N",
        "P3_050N",
        "P3_051N",
        "P3_052N",
        "P3_053N",
        "P3_058N",
        "P3_059N",
        "P3_060N",
        "P3_061N",
    ]  # 4
    + ["P3_064N", "P3_065N", "P3_066N", "P3_067N", "P3_069N"]  # 5
    + ["P3_071N"],  # 6
    "AMINVAP": ["P3_005N"]  # 1
    + ["P3_012N", "P3_016N", "P3_020N", "P3_021N", "P3_022N"]  # 2
    + [
        "P3_027N",
        "P3_031N",
        "P3_032N",
        "P3_033N",
        "P3_037N",
        "P3_038N",
        "P3_039N",
        "P3_043N",
        "P3_044N",
        "P3_045N",
    ]  # 3
    + [
        "P3_048N",
        "P3_049N",
        "P3_050N",
        "P3_054N",
        "P3_055N",
        "P3_056N",
        "P3_058N",
        "P3_059N",
        "P3_060N",
        "P3_062N",
    ]  # 4
    + ["P3_064N", "P3_065N", "P3_066N", "P3_068N", "P3_069N"]  # 5
    + ["P3_071N"],  # 6
    "ASIANVAP": ["P3_006N"]  # 1
    + ["P3_013N", "P3_017N", "P3_020N", "P3_023N", "P3_024N"]  # 2
    + [
        "P3_028N",
        "P3_031N",
        "P3_034N",
        "P3_035N",
        "P3_037N",
        "P3_040N",
        "P3_041N",
        "P3_043N",
        "P3_044N",
        "P3_046N",
    ]  # 3
    + [
        "P3_048N",
        "P3_051N",
        "P3_052N",
        "P3_054N",
        "P3_055N",
        "P3_057N",
        "P3_058N",
        "P3_059N",
        "P3_061N",
        "P3_062N",
    ]  # 4
    + ["P3_064N", "P3_065N", "P3_067N", "P3_068N", "P3_069N"]  # 5
    + ["P3_071N"],  # 6
    "NHPIVAP": ["P3_007N"]  # 1
    + ["P3_014N", "P3_018N", "P3_021N", "P3_023N", "P3_025N"]  # 2
    + [
        "P3_029N",
        "P3_032N",
        "P3_034N",
        "P3_036N",
        "P3_038N",
        "P3_040N",
        "P3_042N",
        "P3_043N",
        "P3_045N",
        "P3_046N",
    ]  # 3
    + [
        "P3_049N",
        "P3_051N",
        "P3_053N",
        "P3_054N",
        "P3_056N",
        "P3_057N",
        "P3_058N",
        "P3_060N",
        "P3_061N",
        "P3_062N",
    ]  # 4
    + ["P3_065N", "P3_066N", "P3_067N", "P3_068N", "P3_069N"]  # 5
    + ["P3_071N"],  # 6
}

COL_DICT_DEC_2020_SINGLE = {  # 2020/dec/pl
    "GEO_ID": "GEOID",
    "P1_001N": "POP",
    "P4_001N": "VAP",
    "P4_002N": "HVAP",
    "P4_005N": "WVAP",
    "P3_004N": "BVAP",  # "P4_006N": "BVAP",
    "P4_007N": "AMINVAP",
    "P4_008N": "ASIANVAP",
    "P4_009N": "NHPIVAP",
    "P4_010N": "OTHERVAP",
    "P4_011N": "2MOREVAP",
}

# COL_DICT_DEC_2020 = { # 2020/dec/cd119
#     "GEO_ID": "GEOID",
#     "P1_001N": "POP",
#     "P11_001N": "VAP",
#     "P11_002N": "HVAP",
#     "P11_005N": "WVAP",
#     "P11_006N": "BVAP",
#     "P11_007N": "AMINVAP",
#     "P11_008N": "ASIANVAP",
#     "P11_009N": "NHPIVAP",
#     "P11_010N": "OTHERVAP",
#     "P11_011N": "2MOREVAP",
# }

################################ Other Info ####################################


def flatten(lis_of_lis):
    return [element for lis in lis_of_lis for element in lis]


six_colors = [
    [204, 12, 12, 256],  # [232, 23, 23, 256],
    [232, 138, 23, 256],
    [252, 226, 25, 256],
    [40, 138, 45, 256],
    [23, 131, 232, 256],
    [114, 66, 245, 256],
]

SIX_COLORS = [np.array(color, dtype=float) / 256 for color in six_colors]

smooth_six_colors = [
    [230, 41, 11, 256],  # [237, 83, 20, 256],
    [252, 148, 43, 256],  # [255, 185, 42, 256],
    [254, 235, 8, 256],
    [155, 202, 62, 256],
    [58, 187, 201, 256],
    [102, 109, 203, 256],
]

SMOOTH_SIX_COLORS = [
    np.array(color, dtype=float) / 256 for color in smooth_six_colors
]

SHORTBURSTS_MAXIMUMS = {
    "LA": {
        2010: {"BVAP": 33, "POCVAP": 46},
        2020: {"BVAP": 33, "POCVAP": 50},
    },
    "TX": {
        2010: {"BVAP": 6, "HVAP": 48, "POCVAP": 94},
    },
    "VA": {
        2010: {"BVAP": 12, "POCVAP": 35},
    },
    "NM": {
        2010: {"HVAP": 38, "POCVAP": 58},
    },
}
