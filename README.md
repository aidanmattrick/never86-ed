# Never86'ed Machine Learning Overview

The purpose of this ReadMe is to provide more insight into the machine learning aspect of this project (August/September 2019).

In order to create a prototype to see if this was a viable business, we want to create a simplified version to answer the questions of feasibility (technical aspects) and viability (business aspects).

While we knew that a production-ready MVP would require both image classification (to determine type of bottle) and linear regression (fill-level), we had only a couple weeks to complete a working prototype for our final project for the Lighthouse Labs web development bootcamp. With this in mind, we put together a model that used image classification to (given a single type of bottle  - a bottle of Bulleit Bourbon) determine set fill levels (0%, 20%, 40%, 60%, 80%, and 100% full).

For this prototype, we built the backend in Ruby/Rails and the frontend in React. For the machine learning portions, Python (3) was used. 

Our machine learning platform was TensorFlow. 
We ended up applying transfer learning from ResNet50. We tried importing other models such as ResNet100, VGG16, Xception but had better results with ResNet50. We also applied weights from Imagenet.

We had best results when freezing the layers of the base model and adding two additional layers of 2000 nodes each and lowered the dropout rate between them (0.25).

For our optimizer, we had best results with Adam but also tried out other optimizers (i.e. SGT) while experimenting with different learning rates (best success was with 0.001).

For our dataset we collected few thousand images and had a split 80/10/10 between the training/validation/test. Through this we were able to get our model in a narrow use case (single type of bottle with controlled fill levels) up to 95% accuracy.

We also experimented with using top 2 accuracy which further boosted the accuracy of our model.


