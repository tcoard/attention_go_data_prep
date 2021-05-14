import pickle

mt = pickle.load(open("local_run/MT_dgp.pkl", "rb"))

FUNCTION_CAND = ["bp", "mf", "cc"]
for i, ns_nsid in enumerate([("bp", "GO:0008150"), ("mf", "GO:0003674"), ("cc", "GO:0005575")]):
    namespace, ns_id = ns_nsid
    ns_data = list()
    ns_data.append(ns_id)
    ns_data.append(mt[0]) # it is possible that I need to filter this per namespace
    ns_data.append(set(mt[i+1]))
    ns_data.append(mt[i+1])
    ns_data.append(set().union(*mt[1:])) # all go terms?
    ns_data.append({go_id: i for i, go_id in enumerate(mt[i+1])})
    pickle.dump(ns_data, open(f"{namespace}_go_meta.pkl", "wb"))
