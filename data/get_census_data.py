import os
import requests
import pandas as pd
import sys

sys.path.append(".")
from data.config import StateConfig
import constants


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

    # url = "https://api.census.gov/data/%d/dec/sf1?get=" % config.year
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
    df["GEOID"] = df["GEOID"].apply(lambda x: x.split("US")[1])
    df["GEOID"] = df["GEOID"].apply(lambda x: x.zfill(geoid_length))
    df = df[list(col_dict.values())].sort_values("GEOID")
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
    print(
        f"Successfully downloaded decennial census data for {config.state} in {config.year} at the {config.granularity} level"
    )
    df.to_csv(csv_path)


if __name__ == "__main__":
    config = StateConfig("LA", 2010, "block_group")
    download_decennial_census_data(config)
