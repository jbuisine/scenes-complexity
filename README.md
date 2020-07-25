# Scenes complexity

## Description

Project developed in order to study the complexity of few scenes.

## Installation


```bash
git clone https://github.com/prise-3d/scenes-complexity.git
```


```bash
pip install -r requirements.txt
```

## How it works ?

Generate estimators data on your own images data:
```bash
python run/scenes_classification_data.py --folder /path/to/scenes --estimators estimator_1,estimator2 --output estimators_data.csv
```

Data file is saved into `data/generated` folder.

You can try to clusterize images using:
```bash
python run/scenes_classification.py --data data/generated/estimators_data.csv --clusters 3 --output estimated_clusters.png
```



## Contributors

* [jbuisine](https://github.com/jbuisine)

## License

[MIT](LICENSE)