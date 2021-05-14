import pickle


def main():
    go_namespace = dict()
    go_id = ""
    with open("/home/tcoard/Downloads/data2016/go.obo", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id:"):
                go_id = line.split(" ")[1]
            if line.startswith("namespace:"):
                namespace = line.split(" ")[1]
                go_namespace[go_id] = namespace
                go_id = ""
                namespace = ""
    pickle.dump(go_namespace, open( "go.pkl", "wb" ) )



if __name__ == '__main__':
    main()
