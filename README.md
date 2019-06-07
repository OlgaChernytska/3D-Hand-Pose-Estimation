# 3D Hand Pose Estimation from Single RGB Camera
Master's Thesis in Ukrainian Catholic University (2018)

All the details on the data, preprocessing, model architecture and training details can be found in [thesis text](http://er.ucu.edu.ua/bitstream/handle/1/1327/Chernytska%20-%203D%20Hand%20Pose%20Estimation%20from%20Single%20RGB%20Camera%20-%20master%20thesis.pdf).

## Requirements

Requirements are specified in `requirements.txt`.
```bash
pip install -r requirements.txt
```

Model works only on cuda.

## Usage

There are two main scripts - `trainer.py` and `evaluate.py`, which are used for training and evaluation.

```bash
python trainer.py {experiment_name}
``` 
```bash
python evaluate.py {experiment_name} ({dataset_name})
```
All parameters are specified in `config.yaml` file in corresponding experiment folder.

Examples:

```bash
python trainer.py e010
``` 

```bash
python evaluate.py e010 dexter+object
```

## Project structure

Folders `dataset`, `model`, `criterion`, `metric`, `optimizer` contain datasets, models, losses, metrics for evaluation and optimizers, respectively. 

To get a specific dataset, model, loss, metric or optimizer, call functions `get_dataloder`, `get_model`, `get_criterion`, `get_metric` or `get_optimizer`, respectively. Functions are defined in `__init__.py` files in corresponding folders. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

