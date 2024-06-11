import pickle

def fileLister(file):
    f = open(file, "rb")
    k = pickle.load(f)
    f.close()
    return k


def merge(dest, fileNames):
    tot_list = []
    for file in fileNames:
        tot_list.append(fileLister(file))
    f = open(dest, "wb")
    pickle.dump(tot_list, f)
    f.close()
    print("List: ", tot_list)
    return tot_list

merge("DS2_3_Shot_res.txt", ["Merged_few_Shot_DS2.txt", "res16150.txt"])

