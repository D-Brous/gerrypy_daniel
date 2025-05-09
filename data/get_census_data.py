import os
import requests
import pandas as pd
import sys
import math

sys.path.append(".")
import constants
from data.config import StateConfig

import os
from io import BytesIO
from ftplib import FTP
import socket
from zipfile import ZipFile
from urllib.request import urlopen


def get_decennial_census_data(
    config: StateConfig, interpretation: constants.Interpretation
):
    # Build the url prefix
    if config.year == 2010:
        col_dict = constants.COL_DICT_DEC_2010
        url_prefix = "https://api.census.gov/data/2010/dec/sf1?get="
    elif config.year == 2020:
        if interpretation == "any_part":
            col_dict = constants.COL_DICT_DEC_2020_ANY_PART
        elif interpretation == "single":
            col_dict = constants.COL_DICT_DEC_2020_SINGLE
        else:
            raise ValueError("Unrecognized value provided for interpretation")
        url_prefix = "https://api.census.gov/data/2020/dec/pl?get="
    else:
        raise ValueError(
            f"The given year, {config.year}, is currently unsupported for census data."
        )

    # Build the url suffix
    url_suffix = "&for="
    granularity = config.granularity
    if granularity == "block":
        url_suffix += "block:*&in=county:*"
        geoid_length = 15
    elif granularity == "block_group":
        url_suffix += "block%20group:*&in=county:*"
        geoid_length = 12
    elif granularity == "tract":
        url_suffix += "tract:*"
        geoid_length = 11
    elif granularity == "county":
        url_suffix += "county:*"
        geoid_length = 5

    # Gather the data in pieces (Can only specify up to 50 columns to
    # grab at once from census api, so this is done in pieces)
    state_fips = constants.FIPS_DICT[config.state]
    url_suffix += "&in=state:%s" % state_fips
    url_suffix += "&key=%s" % constants.CENSUS_API_KEY
    census_col_set = set(constants.flatten(list(col_dict.values())))
    census_geoid = col_dict["GEOID"][0]
    census_col_set.remove(census_geoid)
    census_cols = list(census_col_set)
    n_cols = len(census_cols)
    piece_length = 40
    n_pieces = math.ceil(n_cols / piece_length)
    piece_dfs = {}
    for piece in range(n_pieces):
        piece_cols = census_geoid + ","
        for col_ix in range(
            piece * piece_length, min((piece + 1) * piece_length, n_cols)
        ):
            piece_cols += "%s," % census_cols[col_ix]
        url = url_prefix + piece_cols[:-1] + url_suffix
        # Grab the data
        request = requests.get(url)
        try:
            json_df = request.json()
        except:
            raise ConnectionError(
                f"Failed to download table.\nurl: {url}\ntext: {request.text}"
            )
        piece_dfs[piece] = pd.DataFrame(json_df[1:], columns=json_df[0])

    # Merge the pieces
    census_df = piece_dfs[0]
    for piece in range(1, n_pieces):
        census_df = pd.merge(census_df, piece_dfs[piece], on=census_geoid)

    # print(census_df[census_geoid])
    # Clean up the data
    demo_df = pd.DataFrame(census_df[census_geoid])
    demo_df.rename(columns={census_geoid: "GEOID"}, inplace=True)
    # demo_df = pd.DataFrame(census_df[census_geoid], columns=["GEOID"])

    for col, census_cols in col_dict.items():
        if col != "GEOID":
            demo_df[col] = census_df[census_cols].astype(int).sum(axis=1)
    # df.rename(columns=col_dict, inplace=True)
    # for col in col_dict.values():
    #     if col != "GEOID":
    #         df[col] = df[col].astype(int)
    # check = df["VAP"].copy().to_numpy()
    # for col in list(col_dict.values())[3:]:
    #     check -= df[col].to_numpy()
    # if check.any():
    #     raise ValueError(
    #         "Citizen voting age population columns don't add up to the total voting age population"
    #     )

    demo_df["GEOID"] = demo_df["GEOID"].apply(lambda x: x.split("US")[1])
    demo_df["GEOID"] = demo_df["GEOID"].apply(lambda x: x.zfill(geoid_length))
    demo_df = demo_df.sort_values(
        "GEOID"
    )  # demo_df[list(col_dict.values())].sort_values("GEOID")
    demo_df["POCVAP"] = demo_df["VAP"] - demo_df["WVAP"]
    demo_df = demo_df.set_index("GEOID")
    return demo_df


def download_decennial_census_data(
    config: StateConfig, interpretation: constants.Interpretation = "any_part"
):
    save_path = os.path.join(constants.DEMO_DATA_PATH, config.get_dirname())
    csv_path = os.path.join(save_path, f"pops_{interpretation}.csv")
    if os.path.exists(csv_path):
        print(
            f"Ignored download since the data asked for already exists at the following path: {csv_path}"
        )
        return
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    demo_df = get_decennial_census_data(config, interpretation)
    demo_df.to_csv(csv_path)
    print(
        f"Successfully downloaded decennial census data for {config.state} in {config.year} at the {config.granularity} level"
    )


def download_shapefiles(config: StateConfig):
    # Check if shapefiles are already downloaded
    save_path = os.path.join(constants.CENSUS_SHAPE_PATH, config.get_dirname())
    if os.path.exists(save_path):
        print(
            f"Ignored download since the data asked for already exists at the following directory: {save_path}"
        )
        return

    # Build url
    state = config.state
    year = config.year
    granularity = config.granularity
    url = "/geo/tiger/TIGER%d/" % year

    if granularity == "block":
        url += "TABBLOCK/"
    elif granularity == "block_group":
        url += "BG/"
    elif granularity == "tract":
        url += "TRACT/"
    elif granularity == "county":
        url += "COUNTY/"

    if year == 2010:
        url += "%d/" % year

    print(url)

    # Login to census FTP server and download the desired shapefiles
    try:
        ftp = FTP("ftp.census.gov", timeout=1000)
    except socket.timeout:
        raise ConnectionError(f"Failed to connect to the census database")
    ftp.login()
    ftp.cwd(url)
    os.mkdir(save_path)
    state_fips = constants.FIPS_DICT[state]
    for file_name in ftp.nlst():
        fips = file_name.split("_")[2]
        if fips != state_fips:
            continue
        resp = urlopen("https://www2.census.gov" + url + file_name)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(save_path)
        print(
            f"Successfully downloaded shapefiles for {config.state} in {year} at the {granularity} level"
        )
    ftp.close()


if __name__ == "__main__":
    config = StateConfig("NM", 2020, "block_group")
    # download_decennial_census_data(config)
    download_shapefiles(config)
