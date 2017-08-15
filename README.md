## Diet_Code:
This dir contains the implementation of [Diet Networks](https://arxiv.org/abs/1611.09340). The particular embedding we are going to use is the Histogram. The idea is to preprocess the data set by contructing a matrix s.t. each cell is the frequency of SNP k taking on value j for class i (where k is the SNP; j is 0,1,2; and i is 1,...,26). This is the embedding that is used for the auxilary networks which are used to construct the weight matrix in such a way that we greatly reduce the number of parameters. That is, we lean an embedding on the transpose of the data and construct a matrix that represents the first layer of the discrimative network (the net that makes predictions and optionally reconstructs using an autoencoder). 


#### Preprocess:
We assume you have {.panel file for labels, the 3 plinkfiles}. We also are working with Docker. (Dockerfile provided)

- Build environment image:
```
sudo docker build -t diet_code_env -f Dockerfile.gpu .
```
- launch the container and place add the volume with the data we are using:
```
sudo nvidia-docker run -it -p 81:6006 -v /path/to/4files:/usr/local/diet_code/1000G diet_code_env
```

- Preprocess the data. Assumes you have the files: `affy_samples.20141118.panel`, `genotypes.bim/bed/fam` in `/usr/local/diet_code/1000G`. In the container, we call:
```
 python preprocess.py
```
- We get these files:
  * `hist3x26.npy`: (px78) matrix of the freq of snp k taking on value j for class i. Built from the train/val set only.
  * `train{}.npy`: 75% of the remaining 80% of the data in the format where {} = X or Y for genomic data and labels resp..
  * `valid{}.npy`: 25% of the remaining 80% of the data in the format where {} = X or Y for genomic data and labels resp.
  * `test.npy`: 20% of the overall data (not used in constructing the histogram embedding.

#### Train the Model:
- Grab the docker tf environment image if it doesnt build above:
```
docker pull ljstrnadiii/diet_code_env:0.1
```
- Then, run the preprocess described above.
- Finally, train the model like so:
```
sudo nvidia-docker run -it -p 81:6006 -v /path/to/this/repo:/usr/local/diet_code  diet_code_env

```
```
python train.py
```

### TODO: 
- build the skeleton of the tensorflow model
- build numpy pipeline for data entry
- output the loss and scores to tensorboard
