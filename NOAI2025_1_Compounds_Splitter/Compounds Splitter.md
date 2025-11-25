## 2025 IOAI—Compounds Splitter

### I. Problem Overview

Compounds refer to the phenomenon of forming new words by combining multiple short words, which is particularly common in German. For example, "Fußball" is formed by combining "Fuß" and "Ball," which mean "foot" and "ball," respectively. "Autobahnanschlussstelle" (highway interchange) is formed by combining "Autobahn," "Anschluss," and "Stelle," which mean "highway," "connection," and "place," respectively.

In this task, we need to split compound words in a German sentence into separate words with spaces. For example, "Fußballspieler" should be split into "Fußball" and "spieler."

### II. Dataset

`data/train.json` contains over 90,000 German compound words, each split into individual words. Each data entry consists of two fields: the compound word and its segmentation labels.

The validation set (`val.json`) and test set (`test.json`) each contain over 10,000 German compound words. The specific dataset sizes are as follows:

- **Training set**: 94,306 entries
- **Validation set**: 11,788 entries
- **Test set**: 11,789 entries

Example data:

```json
{
    "Sprachbereich": [0,0,0,0,0,1,0,0,0,0,0,0,1]
}
```

This indicates that "Sprachbereich" is split into "Sprach" and "bereich," where `1` represents a split point.

[Dataset Link](https://chatgpt.com/c/67a77b2f-a9b8-8013-b7da-88800b09a06e)

### III. Task

Implement a compound word splitter using a deep learning model. You may use any deep learning framework you are familiar with.

### IV. Submission

Participants need to submit:

1. **Model Training and Inferencing code**
2. **Predicted output file** (run inference on the test set and output the results)

Baseline code reference: [baseline.ipynb](https://chatgpt.com/c/67a77b2f-a9b8-8013-b7da-88800b09a06e)

### V. Evaluation

The final score will be calculated based on the following metrics:

**Accuracy**: Measures the match between predicted segmentation points and ground truth labels.