import os
import numpy as np
from plinkio import plinkfile
import json
import pandas as pd


"""
Script to generate the SNP Histogram Embeddings and save as npy. 
In addition, we will save te train/test/val data sets at npy files (todo).

In order to run, 
    cd here 
    python preprocess.py

"""

def generate_1000_genomes_hist(transpose=False, label_splits=None,
                               feature_splits=None, fold=0, perclass=False,
                               path = "", prefix="", phenotype_file=""):

    """ 
    generated the histogram embedding of the SNP data
    Args:
        label_split: percent to split for train/val
        feature_split: not implemented
        perclass: Bool-- hist on 3*26 classes?
        path: path to data /usr/local/dietnet/1000G
        prefix: name of data e.g. genotypes
        phenotype_file: the panel file for the phenotype data
    Returns:
        N/a: it saves a npy of the histogram, train, valid, and test data sets.
    """
    
    train, valid, test = load_1000_genomes(transpose=transpose,
                                                 label_splits=label_splits,
                                                 feature_splits=feature_splits,
                                                 fold=fold, path=path, prefix=prefix,
                                                 norm=False, nolabels='raw',
                                                 phenotype_file=phenotype_file)

    # Generate no_label: fuse train and valid sets
    nolabel_orig = (np.vstack([train[0], valid[0]])).transpose()
    nolabel_y = np.vstack([train[1], valid[1]])

    nolabel_y = nolabel_y.argmax(axis=1)

    filename = 'histo3x26' 
    testfilename = 'histo3x26.npy'

    if perclass and not os.path.isfile(os.path.join(path,testfilename)):
        print("making the histogram embedding...")
        # the first dimension of the following is length 'number of snps'
        nolabel_x = np.zeros((nolabel_orig.shape[0], 3*26))
        for i in range(nolabel_x.shape[0]):
            if i % 5000 == 0:
                print "processing snp no: ", i
            for j in range(26):
                nolabel_x[i, j*3:j*3+3] += \
                    np.bincount(nolabel_orig[i, nolabel_y == j ].astype('int32'), minlength=3)
                nolabel_x[i, j*3:j*3+3] /= \
                    nolabel_x[i, j*3:j*3+3].sum()
        print("made.")
        np.save(os.path.join(path, filename), nolabel_x)
        nolabel_x = nolabel_x.astype('float32')
    else:
        print("hist3x26 already exists")
    

    print("saving embedding matrix, train, test, valid as npy in: " + str(path))
    # saved as [x,y] format 
    np.save(os.path.join(path, "trainX"), train[0])
    np.save(os.path.join(path, "validX"), valid[0])
    np.save(os.path.join(path, "testX"), test[0])
    np.save(os.path.join(path, "trainY"), train[1])
    np.save(os.path.join(path, "validY"), valid[1])
    np.save(os.path.join(path, "testY"), test[1])
    print("saved.")



def load_1000_genomes(transpose=False, label_splits=None, feature_splits=None,
                      nolabels='raw', fold=0, norm=True,
                      path="", prefix="", phenotype_file="" ):
    """
    This function is almost set up for k-fold. for now it simply pulls the 
    test set out so that we don't build our embedding with the test set
    Args:
        label_splits: train/val split
        fold: which fold to use at test
        norm: is not implemented
        path: path to all the data 1000G
        prefix: name of the three plink files e.g. genotypes
        phenotype_file: path to the pheno file for load_data
    returns:
        rvals: list of train, val, test 
    """

    if nolabels == 'raw' or not transpose:
        # Load raw data either for supervised or unsupervised part
        x, y = load_data(path, prefix, phenotype_file)
        x = x.astype("float32")

        print("shuffling data...")
        (x, y) = shuffle((x, y))  # seed is fixed, shuffle is always the same

        # Prepare training and validation sets
        assert len(label_splits) == 1  # train/valid split
        # 5-fold cross validation: this means that test will always be 20%
        all_folds = split([x, y], [.2, .2, .2, .2])
        assert fold >= 0
        assert fold < 5

        # Separate choosen test set
        test = all_folds[fold]
        # collapse the remaining folds for train val
        all_folds = all_folds[:fold] + all_folds[(fold + 1):]

        # maybe print shape to debug
        x = np.concatenate([el[0] for el in all_folds])
        y = np.concatenate([el[1] for el in all_folds])
    
    print("just made the test, train, valid sets")
    # Data used for supervised training
    train, valid = split([x, y], label_splits)
    rvals = [train, valid, test]
    
    # save the train, valid and test to npy arrays here. Makes more sense than in load_data

    return rvals


def load_data(path, prefix, phenotype_file):
    """
    loads the prefixed files: prefix.bed, prefix.fam, ...
    and saves, but it may make more sense to save in the load_1000 function above.
    Args:
        prefix: path with last elem as prefix of .bed, .fam, ...
    Returns:
        genomic_data: numpy array
        label_data: nuumpy array of labels
    """
    prefix = os.path.join(path, prefix)
    
    print("loading plink files...")
    Xt_plink = plinkfile.open(prefix)
    num_snps = len(Xt_plink.get_loci())
    num_ind = len(Xt_plink.get_samples())
    num_class = 26
    print("loaded.")

    # save metafile for info
    print("writing meta file...")
    with open(os.path.join(path,"_metadata.json"), 'w') as f:
        json.dump({'num_snps': num_snps,
                       'num_ind': num_ind,
                       'num_class': num_class}, f)    
    print("written.")

    # have to transpose the plinkfile to get X
    trans_filename = os.path.join(path,"trans")
    print("transposing plink file...")
    assert Xt_plink.transpose(trans_filename), "transpose failed"
    print("done.")

    # Now Open the transpose as X
    print("make genomic_data matrix...")
    X_plink = plinkfile.open(trans_filename)
    assert not X_plink.one_locus_per_row(), "Plink file should be transposed"

    # save the data as a npy file:
    genomic_data = np.zeros((num_ind,num_snps), np.int8)
    for i, row in enumerate(X_plink):
            genomic_data[i,:]=list(row)
    print("made.")	
    
    # lets save labels
    print("loading labels and making one-hot rep...")
    pheno = pd.read_csv(os.path.join(path,phenotype_file), sep=None, engine= "python")
    pheno_list = pheno.iloc[:, 1]
    pheno_list_cat = pheno_list.astype('category').cat
    pheno_list_values = pheno_list_cat.categories.values
    pheno_map = pd.DataFrame({'Phenotype': pheno_list_values,
                                                     "Codes": range(len(pheno_list_values))},
                                                     columns=('Phenotype','Codes'))
    pheno_map.to_csv(os.path.join(path,"pheno_map"))
    # okay get labels now that we have a map
    labels = pheno_list_cat.codes.astype(np.uint8)
    nb_class = len(pheno_list_values)
    targets = np.array(labels).reshape(-1)
    
    # makes one hot matrix for label data class1 = [1,0,...,0]
    label_data = np.eye(nb_class)[targets]
    print("just made the one-hot matrix for labels")

    return genomic_data, label_data



def shuffle(data_sources, seed=21):
    """
    Shuffles the data so that the labels stay with the corresponding snps
    """

    np.random.seed(seed)
    indices = np.arange(data_sources[0].shape[0])
    np.random.shuffle(indices)

    return [d[indices] for d in data_sources]



def split(data_sources, splits):
    """
    Splits the given data sources (numpy arrays) according to the provided
    split boundries.
    Ex : if splits is [0.6], every data source will be separated in two parts,
    the first containing 60% of the data and the other containing the
    remaining 40%.
    """

    if splits is None:
        return data_sources

    split_data_sources = []
    nb_elements = data_sources[0].shape[0]
    start = 0
    end = 0

    for s in splits:
        end += int(nb_elements * s)
        split_data_sources.append([d[start:end] for d in data_sources])
        start = end
    split_data_sources.append([d[end:] for d in data_sources])

    return split_data_sources



if __name__=="__main__":
    """
    fold=0: determines which fold to make test set. will not be used to construct 
    the histogram. test is 20% data perclass: true => histogram embedding will be 
    (px78) 78 = 3*26classes label_splits: train will be 75% val will be 25% of 80% 
    total (since test)

    """
    generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                               feature_splits=[1.], fold=0, perclass=True, 
                               path = "/usr/local/diet_code/1000G", prefix="genotypes",
                               phenotype_file="affy_samples.20141118.panel")
