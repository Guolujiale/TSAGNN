# TSAGNN
Prediction of store star using Time Sequential Attention GNN.

## Environment

Please install the necessary libraries according to the following requirements.  
This program need run in the Linux system.

### Dependency library

Please refer to the requirements.txt file to install the following libraries:

- `pytorch==2.5.1`
- `python==3.12`
- `pandas==2.23`
- `pytorch_geometric==2.6.1`
- `tqdm`
- `matplotlib`
- `sklearn`
- `numpy`
- `captum`

You can use' pip' to install the project's dependency library:

```bash
pip install -r requirements.txt
```
### Run

You can run the model by typing the following from the command line:

```bash
python run.py
```
Please make sure your data in the dir 'data' as name **'20-21data_encoded.csv'**, you can rename it in the **line 45** of **run.py**  

You can also prepare yourself data by using the **predata.py**  

If you want to modify the Hyperparameter of this model, you can change the number of **win_size, batch_size, n_layer** in **run.py**  

**win_size:** Use the data of **win_size** months in the past to predict the current data, which will change the result of data preprocessing.  

**n_layer:** Increasing the value of **n_layer** will deepen the model, prolong the training time and improve the accuracy. 

### Result
The result will be save in the dir: **run/**.

### Program structure
```bash
.
├── run.py                    # Main run 
├── requirements.txt          # Libraries and versions required for the project
├── data/                     # The folder where the dataset is stored.
├── run/                      # Result dir
├── README.md                 # Description document of the project
├── layer.py                  # model layer
├── model.py                  # model
├── dataset.py                # prepare gnn dataset
```
