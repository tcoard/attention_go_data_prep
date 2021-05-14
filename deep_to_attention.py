import pickle
import gc
from multiprocessing import Pool
from functools import partial
from itertools import chain, groupby
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
import numpy as np

# These are from the DeepGoPlus paper
MAX_LEN = 2000
MIN_GO = 50

# It was easier to debug if this was generated before the script ran, so I left it hardcoded
aa_to_one_hot = {
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


def get_one_hot_encodings(seq, max_len):
    most_freq_aas = "ACDEFGHIKLMNPQRSTVWYX"
    embedding = list()
    for aa in seq:
        aa = aa if aa in most_freq_aas else "X"
        embedding.append(aa_to_one_hot[aa])
    for _ in range(max_len - len(seq)):
        embedding.append(aa_to_one_hot["empty"])
    return np.array(embedding, dtype="double")


def count_go_terms(seq_annos, go_id_namespace):
    """make a dictionary of {namespace: [list of go ids in that namespace]}"""
    # count how many times the GO ID appears
    go_id_counts = Counter(chain.from_iterable(seq_annos))
    # Filter out GO IDs that appear < MIN_GO times
    filtered_go_ids = dict(filter(lambda tup: tup[1] >= MIN_GO, go_id_counts.items())).keys()
    # Sort GO IDs by namespace
    sorted_go_ids = sorted(filtered_go_ids, key=lambda value: go_id_namespace[value])
    # Create a new list for each namespace
    grouped_go_ids = groupby(sorted_go_ids, key=lambda value: go_id_namespace[value])
    # Create a dict of namespace to list of GO IDs
    final_go_ids = {k: sorted(list(v)) for k, v in grouped_go_ids}
    return final_go_ids


def get_valid_list_index(seq_annos, go_counts):
    valid_index_list = [[], [], []]
    for anno in seq_annos:
        for go_term in anno:
            bp = False
            cc = False
            mf = False
            if go_term in go_counts["biological_process"]:
                bp = True
            if go_term in go_counts["cellular_component"]:
                cc = True
            if go_term in go_counts["molecular_function"]:
                mf = True
        valid_index_list[0].append(bp)
        valid_index_list[1].append(cc)
        valid_index_list[2].append(mf)

    valid_index_list[0] = np.array(valid_index_list[0])
    valid_index_list[1] = np.array(valid_index_list[1])
    valid_index_list[2] = np.array(valid_index_list[2])
    return valid_index_list


def compile_data(dataset, go_counts=None):
    """Create the train/test files to be input into AttentioGo"""
    final = dict()
    pool = Pool()
    with open(f"/home/tcoard/Downloads/data2016/{dataset}_data.pkl", "rb") as f:
        data = pickle.load(f)
        seqs = data[(data.sequences.str.len() < MAX_LEN)]["sequences"].values.tolist()

        final["X_test_length"] = np.array([len(seq) for seq in seqs])
        max_len = len(max(seqs, key=len))
        print(len(seqs))
        final["X_train"] = np.array(pool.map(partial(get_one_hot_encodings, max_len=max_len), seqs), dtype="double")
        print("X_train done")

        # note, we are ignoring external
        with open("go.pkl", "rb") as go_f:
            seq_annos = data[(data.sequences.str.len() < MAX_LEN)]["annotations"].values.tolist()
            final["y_FULL_TERM"] = np.array(seq_annos, dtype=object)
            go_term_namespace = pickle.load(go_f)
            if go_counts is None:
                go_counts = count_go_terms(seq_annos, go_term_namespace)
            valid_index_list = get_valid_list_index(seq_annos, go_counts)
            final["valid_index_list"] = valid_index_list
            del valid_index_list
            print("gc")
            gc.collect()
            print([len(go_counts[i]) for i in go_counts])

            namespaces = ["biological_process", "molecular_function", "cellular_component"]
            final["test_labels"] = [[], [], []]
            for anno in seq_annos:
                for i, namespace in enumerate(namespaces):
                    final["test_labels"][i].append(np.array([0] * len(go_counts[namespace]), dtype="double"))
                for go_term in anno:
                    namespace = go_term_namespace[go_term]
                    # test_labels[      namespace          ][our sequence][ go term one hot encoding position ]
                    if go_term in go_counts[namespace]:
                        final["test_labels"][namespaces.index(namespace)][-1][go_counts[namespace].index(go_term)] = 1
            for i in range(len(final["test_labels"])):
                final["test_labels"][i] = np.array(final["test_labels"][i])
    print("gc")
    gc.collect()
    data = [
        final["X_train"],
        final["X_test_length"],
        final["test_labels"],
        final["y_FULL_TERM"],
        final["valid_index_list"],
    ]
    pickle.dump(data, open(f"deepgo_{dataset}_data.pkl", "wb"))
    return go_counts


def make_meta_data(go_counts):
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
    with open("/home/tcoard/Downloads/data2016/go.obo", "r") as f:
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
    pickle.dump(
        [
            go_namespace,
            np.array(go_counts["biological_process"]),
            np.array(go_counts["molecular_function"]),
            np.array(go_counts["cellular_component"]),
        ],
        open("MT_dgp.pkl", "wb"),
    )


def main():
    go_counts = compile_data("train")
    make_meta_data(go_counts)
    gc.collect()
    compile_data("test", go_counts)


if __name__ == "__main__":
    main()
