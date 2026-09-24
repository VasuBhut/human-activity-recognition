# Human Activity Recognition from dual-accelerometer data

Classifying everyday activities (walking, running, sitting, stairs, and so on) from two body-worn
accelerometers, comparing a CNN, K-Means and a Random Forest before and after tuning.

MSc Artificial Intelligence coursework, COMP6246 Machine Learning Technologies, University of
Southampton, autumn 2025.

## The data

Dual-accelerometer time-series recordings (back and thigh sensors) supplied by the module, one CSV
per participant. **The dataset is not in this repository.** The notebook expects it at
`./MLT-CW-Dataset`.

After cleaning, the training data is about 5 million rows; the full set before cleaning is over
6 million.

## What the notebook does

1. **Exploration and class balance** - counts per activity across the whole dataset.
2. **Cleaning**
   - removes all cycling data,
   - merges stairs-ascending and stairs-descending into a single class (9),
   - fixes participant S007, whose file contains an undocumented label 10 that appears nowhere else
     in the data or the documentation, and which I removed.
3. **Windowing** - the raw signal is cut into 2-second sliding windows, which become the model inputs.
4. **Three pipelines, each baseline then tuned**
   - 1D CNN (TensorFlow/Keras)
   - K-Means (scikit-learn), clusters mapped to activity labels
   - Random Forest (scikit-learn)

## Results

Test set: 22,107 windows, 7 activity classes.

| Model | Accuracy (baseline) | Weighted F1 (baseline) | Accuracy (tuned) | Weighted F1 (tuned) |
|---|---|---|---|---|
| Random Forest | 0.8936 | 0.8928 | **0.9059** | **0.9021** |
| CNN | 0.8201 | 0.8330 | 0.8447 | 0.8581 |
| K-Means | 0.7246 | 0.6451 | 0.7246 | 0.6451 |

The tuned Random Forest is the best model, and it meets the coursework's business target (per-class
precision of at least 75% and recall of at least 50%) on every class except shuffling and stairs,
the two classes the coursework flagged as hard.

Tuning was light for the two classical models. The Random Forest went from 10 to 100 trees. K-Means
only changed its random seed and number of restarts, which is why it barely moves: its seven
clusters still map onto just four activities, separating the high-movement ones and merging the
rest. Allowing more clusters than classes would be the real lever there.

Macro F1 is much lower than weighted F1 for every model (0.77 against 0.90 for the best one), because
the rare classes are the hard ones.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook human_activity_recognition.ipynb
```

Put the dataset in `./MLT-CW-Dataset` first, with the test set in `./MLT-CW-Dataset/test-set`.

## Note

This is my own coursework code, published with my tutor's confirmation that the code I wrote is mine
to share. The assignment brief, the module's reading list and the dataset are not included.
