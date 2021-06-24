import pickle
import resource
import gc
from multiprocessing import Pool
from itertools import chain, groupby
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import numpy as np

# These are from the DeepGoPlus paper
# maximum sequence length
MAX_LEN = 2000
# minimum number of times a GO term has to appear in order for it to be included
MIN_GO = 50

DATA_PATH = "/home/tcoard/git/attention_go_data_prep/local_run"
DEEPGO_DATA_PATH = "/home/tcoard/Downloads/data2016"
OBO_FILE = f"{DEEPGO_DATA_PATH}/go.obo"

# It was easier to debug if this was generated before the script ran, so I left it hardcoded
AA_TO_ONE_HOT = {
    "A": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "C": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "D": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "E": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "F": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "G": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "H": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "I": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "K": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "L": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "M": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "N": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "P": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "Q": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
    "R": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype="double"),
    "S": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype="double"),
    "T": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype="double"),
    "V": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype="double"),
    "W": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype="double"),
    "Y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype="double"),
    "X": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype="double"),
    "empty": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="double"),
}


def get_one_hot_encodings(seqs: list[str]) -> np.ndarray:
    """Returns an array with a one hot endcoded vector for each amino acid position in the sequence.
    If the sequence is shorter than the longest sequence,
    the difference in length is filled in with an empty vector
    """
    most_freq_aas = "ACDEFGHIKLMNPQRSTVWYX"
    embeddings = np.zeros([len(seqs), MAX_LEN, 21], dtype="double")
    for i, seq in enumerate(seqs):
        for ii, aa in enumerate(seq):
            # aa = aa.upper()
            aa = aa if aa in most_freq_aas else "X"
            embeddings[i][ii][most_freq_aas.find(aa)] = 1
    return embeddings


# def get_one_hot_encodings(seq: str) -> np.ndarray:
#    """Returns an array with a one hot endcoded vector for each amino acid position in the sequence.
#    If the sequence is shorter than the longest sequence,
#    the difference in length is filled in with an empty vector
#    """
#    most_freq_aas = "ACDEFGHIKLMNPQRSTVWYX"
#    embedding = np.zeros([MAX_LEN, 21], dtype="double")
#    for i, aa in enumerate(seq):
#        #aa = aa.upper()
#        aa = aa if aa in most_freq_aas else "X"
#        embedding[i][most_freq_aas.find(aa)] = 1


#    '''
#    for aa in seq:
#        #aa = aa.upper()
#        aa = aa if aa in most_freq_aas else "X"
#        embedding.append(AA_TO_ONE_HOT[aa])
#    # pad the remaining positions with the empty encoding
#    #embedding.extend([AA_TO_ONE_HOT["empty"]] * (actual_max_len - len(seq)))
#    for _ in range(MAX_LEN - len(seq)):
#        embedding.append(AA_TO_ONE_HOT["empty"])
#    return np.array(embedding, dtype="double")
#    '''
#    return embedding


def count_go_terms(
    seq_annos: list[list[str]], go_term_namespace: dict[str, str]
) -> tuple[dict[str, list[str]], list[str]]:
    """make a dictionary of {namespace: [list of go ids in that namespace]}"""
    # count how many times the GO ID appears
    go_id_counts = Counter(chain.from_iterable(seq_annos))
    # Filter out GO IDs that appear < MIN_GO times
    filtered_go_ids = dict(filter(lambda tup: tup[1] >= MIN_GO, go_id_counts.items())).keys()
    # Sort GO IDs by namespace
    sorted_go_ids = sorted(filtered_go_ids, key=lambda value: go_term_namespace[value])
    # Create a new list for each namespace
    grouped_go_ids = groupby(sorted_go_ids, key=lambda value: go_term_namespace[value])
    # Create a dict of namespace to list of GO IDs
    final_go_ids = {k: sorted(list(v)) for k, v in grouped_go_ids}
    return final_go_ids, sorted_go_ids


def get_valid_list_index(seq_annos: list[list[str]], go_term_namespace: dict[str, str]) -> list[np.ndarray]:
    """For each GO namespace, make a boolean list for if each GO term is in that namespace"""
    bp = np.full(len(seq_annos), False)
    cc = np.full(len(seq_annos), False)
    mf = np.full(len(seq_annos), False)
    for i, anno in enumerate(seq_annos):
        for go_term in anno:
            namespace = go_term_namespace[go_term]
            if namespace == "biological_process":
                bp[i] = True
            elif namespace == "cellular_component":
                cc[i] = True
            elif namespace == "molecular_function":
                mf[i] = True

    valid_index_list = list()
    valid_index_list.append(np.array(bp))
    valid_index_list.append(np.array(cc))
    valid_index_list.append(np.array(mf))
    return valid_index_list


def obo_to_dict() -> dict[str, str]:
    """creates a dict of {goid: namespace}"""
    go_namespace = dict()
    go_id = ""
    with open(OBO_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id:"):
                go_id = line.split(" ")[1]
            if line.startswith("namespace:"):
                namespace = line.split(" ")[1]
                go_namespace[go_id] = namespace
                go_id = ""
                namespace = ""
    # pickle.dump(go_namespace, open(f"{DATA_PATH}/go.pkl", "wb"))
    return go_namespace


def get_seq_and_anno(dataset: Literal["train", "test"]) -> tuple[list[str], list[list[str]]]:
    # load the train/test dataset
    with open(f"{DEEPGO_DATA_PATH}/{dataset}_data.pkl", "rb") as f:
        data = pickle.load(f)
        # filter to sequences below maximum length
        seqs = data[(data.sequences.str.len() <= MAX_LEN)]["sequences"].values.tolist()
        seq_annos = data[(data.sequences.str.len() <= MAX_LEN)]["annotations"].values.tolist()
        del data
        gc.collect()
        return seqs, seq_annos


def get_test_labels(
    seq_annos: list[list[str]], go_term_namespace: dict[str, str], go_counts: dict[str, list[str]]
) -> list[np.ndarray]:
    namespaces = ["biological_process", "molecular_function", "cellular_component"]
    bp = np.zeros([len(seq_annos), len(go_counts["biological_process"])], dtype="double")
    mf = np.zeros([len(seq_annos), len(go_counts["molecular_function"])], dtype="double")
    cc = np.zeros([len(seq_annos), len(go_counts["cellular_component"])], dtype="double")
    test_labels = [bp, mf, cc]
    for i, anno in enumerate(seq_annos):
        for go_term in anno:
            namespace = go_term_namespace[go_term]
            # test_labels[      namespace          ][our sequence][ go term one hot encoding position ]
            test_labels[namespaces.index(namespace)][i][go_counts[namespace].index(go_term)] = 1
    return test_labels


def filter_out_unused_go_terms(
    seqs: list[str], seq_annos: list[list[str]], go_term_namespace: dict[str, str], usable_go_ids: list[str]
):
    filtered_go_term_namespace = {k: v for k, v in go_term_namespace.items() if k in usable_go_ids}
    filtered_seqs = list()
    filtered_seq_annos = list()
    for seq, anno in zip(seqs, seq_annos):
        new_annos = list()
        for go_term in anno:
            if go_term in usable_go_ids:
                new_annos.append(go_term)
        if new_annos:
            filtered_seqs.append(seq)
            filtered_seq_annos.append(new_annos)
    return filtered_seqs, filtered_seq_annos, filtered_go_term_namespace


def compile_data(
    dataset: Literal["train", "test"],
    go_counts: Optional[dict[str, list[str]]] = None,
    usable_go_ids: Optional[list[str]] = None,
    go_term_namespace: Optional[dict[str, str]] = None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Create the train/test files to be input into AttentioGo"""
    final = dict()
    pool = Pool()
    # this hack runs get_seq_and_anno in a new process, therefore deleting all variables when it leaves
    # the function. If not done this way, internal references in out of scope variabels will linger
    print("before function", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
    seqs, seq_annos = pool.map(get_seq_and_anno, [dataset])[0]

    if go_term_namespace is None:
        go_term_namespace = obo_to_dict()
    if go_counts is None or usable_go_ids is None:
        go_counts, usable_go_ids = count_go_terms(seq_annos, go_term_namespace)

    orig_seq_num = len(seqs)
    print("starting filter", flush=True)
    seqs, seq_annos, go_term_namespace = filter_out_unused_go_terms(seqs, seq_annos, go_term_namespace, usable_go_ids)
    print("finished filter. Lost this many seqs: ", orig_seq_num - len(seqs), flush=True)

    final["X_test_length"] = np.array([len(seq) for seq in seqs])
    # actual_max_len = len(max(seqs, key=len))
    # print(f"{actual_max_len=}")
    # final["X_train"] = np.array(pool.map(get_one_hot_encodings, seqs), dtype="double")

    final["X_train"] = get_one_hot_encodings(seqs)
    final["y_FULL_TERM"] = np.array(seq_annos, dtype=object)
    final["valid_index_list"] = get_valid_list_index(seq_annos, go_term_namespace)
    print("after get_valid_list_index", flush=True)

    final["test_labels"] = get_test_labels(seq_annos, go_term_namespace, go_counts)

    print("after test_labels", flush=True)

    data = [
        final["X_train"],
        final["X_test_length"],
        final["test_labels"],
        final["y_FULL_TERM"],
        final["valid_index_list"],
    ]
    pickle.dump(data, open(f"{DATA_PATH}/deepgo_{dataset}_data.pkl", "wb"))
    return go_counts, usable_go_ids, go_term_namespace


def make_go_meta_files(mt: list) -> None:
    # mt = pickle.load(open(f"{DATA_PATH}/mt_dgp.pkl", "rb"))
    for i, ns_nsid in enumerate([("bp", "GO:0008150"), ("mf", "GO:0003674"), ("cc", "GO:0005575")]):
        namespace, ns_id = ns_nsid
        ns_data = list()
        ns_data.append(ns_id)
        ns_data.append(mt[0])  # it is possible that I need to filter this per namespace
        ns_data.append(set(mt[i + 1]))
        ns_data.append(mt[i + 1])
        ns_data.append(set().union(*mt[1:]))  # all go terms?
        ns_data.append({go_id: i for i, go_id in enumerate(mt[i + 1])})
        pickle.dump(ns_data, open(f"{DATA_PATH}/{namespace}_go_meta.pkl", "wb"))


def make_meta_data(go_counts: dict[str, list[str]]) -> None:
    """create MT_dgp.pkl file, which is a dictionary with GO data and relationships"""

    @dataclass
    class GoData:
        name: str = ""
        children: set[str] = field(default_factory=set)
        id: str = ""
        regulates: list[str] = field(default_factory=list)
        is_a: list[str] = field(default_factory=list)
        part_of: list[str] = field(default_factory=list)
        is_obsolete: bool = False

    go_namespace = defaultdict(GoData)
    # go_namespace['new_id'].children.append('new_child')
    go_id = ""
    with open(OBO_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id:"):
                go_id = line.split(" ")[1]
                go_namespace[go_id].id = go_id
            elif line.startswith("name:"):
                go_namespace[go_id].name = (" ").join(line.split(" ")[1:])
            elif line.startswith("is_a:"):
                parent_id = line.split(" ")[1]
                go_namespace[go_id].is_a.append(parent_id)
                go_namespace[parent_id].children.add(go_id)
            elif line.startswith("relationship:"):
                line_elems = line.split(" ")
                if line_elems[1] == "regulates":
                    go_namespace[go_id].regulates.append(line_elems[2])
                elif line_elems[1] == "part_of":
                    go_namespace[go_id].part_of.append(line_elems[2])
            elif line.startswith("is_obsolete"):
                if line.split(" ")[1] == "true":
                    go_namespace[go_id].is_obsolete = True

    go_namespace = {k: asdict(v) for k, v in go_namespace.items()}
    mt = [
        go_namespace,
        np.array(go_counts["biological_process"]),
        np.array(go_counts["molecular_function"]),
        np.array(go_counts["cellular_component"]),
    ]
    pickle.dump(mt, open(f"{DATA_PATH}/MT_dgp.pkl", "wb"))
    make_go_meta_files(mt)


def main() -> None:
    go_counts, usable_go_ids, go_term_namespace = compile_data("train")
    gc.collect()
    make_meta_data(go_counts)
    gc.collect()
    compile_data("test", go_counts, usable_go_ids, go_term_namespace)


if __name__ == "__main__":
    main()
