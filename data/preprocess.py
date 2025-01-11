import networkx as nx
import sys
from scipy.spatial.distance import pdist, squareform

sys.path.append("../gerrypy_daniel")
from data.config import StateConfig
from data.df import ShapeDataFrame, DemoDataFrame
from data.graph import Graph


def prepare_optimization_cache(config: StateConfig):
    shape_df = ShapeDataFrame.from_config(config)
    # demo_df = DemoDataFrame
    G = Graph.from_shape_df(shape_df)
    edge_dists = dict(nx.all_pairs_shortest_path_length(G))

    centroids = shape_df[shape_df.centroid.x, shape_df.centroid.y].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(
        constants.OPT_DATA_PATH, granularity, state, str(year)
    )
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, "state_df.csv"), index=False)
    np.save(os.path.join(save_path, "lengths.npy"), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, "G.p"))
    pickle.dump(edge_dists, open(os.path.join(save_path, "edge_dists.p"), "wb"))
    cgus = load_cgus(state, year, granularity)
    state_df = pd.DataFrame(
        {
            "x": cgus.centroid.x,
            "y": cgus.centroid.y,
            "area": cgus.area / 1000**2,  # sq km
            "GEOID": cgus.GEOID.apply(lambda x: str(x).zfill(11)),
        }
    )

    # Join location data with demographic data
    granularity_path = ""
    if granularity == "block":
        granularity_path = constants.BLOCK_DATA_PATH
    elif granularity == "block_group":
        granularity_path = constants.BLOCK_GROUP_DATA_PATH
    elif granularity == "tract":
        granularity_path = constants.TRACT_DATA_PATH
    elif granularity == "county":
        granularity_path = constants.COUNTY_DATA_PATH

    demo_data = pd.read_csv(
        os.path.join(
            granularity_path, "%d_acs5" % year, "%s_tract.csv" % state
        ),
        low_memory=False,
    )
    demo_data["GEOID"] = (
        demo_data["GEOID"].astype(str).apply(lambda x: x.zfill(11))
    )  # might need to modify this to accommadte other granularities
    demo_data = demo_data.set_index("GEOID")
    if use_name_map:
        demo_data = demo_data[list(CENSUS_VARIABLE_TO_NAME[str(year)])]
        demo_data = demo_data.rename(columns=CENSUS_VARIABLE_TO_NAME[str(year)])
    else:
        demo_data = demo_data[["TOTPOP", "VAP", "BVAP", "WVAP"]]
        demo_data.rename(
            columns={"TOTPOP": "population", "WVAP": "p_white"}, inplace=True
        )
    demo_data[demo_data < 0] = 0

    state_df = state_df.set_index("GEOID")
    state_df = state_df.join(demo_data)
    state_df = state_df.reset_index()

    shape_list = cgus.geometry.to_list()
    adj_graph = libpysal.weights.Rook.from_iterable(shape_list).to_networkx()

    if not nx.is_connected(adj_graph):
        adj_graph = connect_components(cgus)

    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[["x", "y"]].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(
        constants.OPT_DATA_PATH, granularity, state, str(year)
    )
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, "state_df.csv"), index=False)
    np.save(os.path.join(save_path, "lengths.npy"), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, "G.p"))
    pickle.dump(edge_dists, open(os.path.join(save_path, "edge_dists.p"), "wb"))
