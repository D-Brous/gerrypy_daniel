import os
import requests
import pandas as pd
import sys

sys.path.append(".")
import constants
from data.config import StateConfig

import os
from io import BytesIO
from ftplib import FTP
import socket
from zipfile import ZipFile
from urllib.request import urlopen


def get_decennial_census_data(config: StateConfig):
    # Build the url
    if config.year == 2010:
        col_dict = constants.COL_DICT_DEC_2010
        url = "https://api.census.gov/data/2010/dec/sf1?get="
        # If I wanted to implement 2020, I should use dec/pl for that year
    else:
        raise ValueError(
            f"The given year, {config.year}, is currently unsupported for census data."
        )
    for census_col in col_dict.keys():
        url += "%s," % census_col
    url = url[:-1] + "&for="

    granularity = config.granularity
    if granularity == "block":
        url += "block:*&in=county:*"
        geoid_length = 15
    elif granularity == "block_group":
        url += "block%20group:*&in=county:*"
        geoid_length = 12
    elif granularity == "tract":
        url += "tract:*"
        geoid_length = 11
    elif granularity == "county":
        url += "county:*"
        geoid_length = 5

    state_fips = constants.FIPS_DICT[config.state]
    url += "&in=state:%s" % state_fips
    url += "&key=%s" % constants.CENSUS_API_KEY

    # Grab the data
    request = requests.get(url)
    try:
        json_df = request.json()
    except:
        raise ConnectionError(
            f"Failed to download table.\nurl: {url}\ntext: {request.text}"
        )
    df = pd.DataFrame(json_df[1:], columns=json_df[0])

    # Clean up the data
    df.rename(columns=col_dict, inplace=True)
    for col in col_dict.values():
        if col != "GEOID":
            df[col] = df[col].astype(int)
    check = df["VAP"].copy().to_numpy()
    for col in list(col_dict.values())[3:]:
        check -= df[col].to_numpy()
    if check.any():
        raise ValueError(
            "Citizen voting age population columns don't add up to the total voting age population"
        )

    df["GEOID"] = df["GEOID"].apply(lambda x: x.split("US")[1])
    df["GEOID"] = df["GEOID"].apply(lambda x: x.zfill(geoid_length))
    df = df[list(col_dict.values())].sort_values("GEOID")
    df["POCVAP"] = df["VAP"] - df["WVAP"]
    df = df.set_index("GEOID")
    return df


def download_decennial_census_data(config: StateConfig):
    save_path = os.path.join(constants.DEMO_DATA_PATH, config.get_dirname())
    csv_path = os.path.join(save_path, "pops.csv")
    if os.path.exists(csv_path):
        print(
            f"Ignored download since the data asked for already exists at the following path: {csv_path}"
        )
        return
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df = get_decennial_census_data(config)
    df.to_csv(csv_path)
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


if __name__ == "__main__":
    config = StateConfig("VA", 2010, "block_group")
    download_decennial_census_data(config)
    download_shapefiles(config)
