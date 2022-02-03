# Image Classifier for clothing patterns

### Data:
the data was forked from here https://github.com/lstearns86/clothing-pattern-dataset

### Implemented:
  - Pre-processing. Basically get all the images and convert it to a data frame
  - First model using MLP (Multilayer Perceptron)
  - Function to measure accuracy
  - Confusion matrix
  - Use numbers instead of strings for the classes to implove efficiency
  - switch to tensorflow

### To be implemented:
  - Use a model with deeplearning
  - Cross validation

### Tecnologies:
- Opencv
- Sklearn
- numpy
- pandas

### Results:

#### First run: simple MLP with 3 hidden layers with 50 neurons each, using black and white version of the images

The first run was made for comparison purposes with more refined models.

accuracy per class:

  dotted: 0.444444444444444
  solid: 0.4166666666666667
  striped: 0.1
  checkered: 0.16
  floral: 0.5161290322580645
  zig zag: 0.08695652173913043
  
#### Second run with tensorflow:

  Results are can be seem on the plots on folder version 2. Needs  further investigation.
