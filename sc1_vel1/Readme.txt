Readme.txt

========================
Preprocess:

feature vectors:
0. position and velocity
1. deformation gradient
2. ratio of edge length
3. ratio of angle
4. ratio of cos(a)

5. precompute the geodesic distance matrix


========================
Smooth layer:
Learn per triangle weights with a mask.



========================
Pooling layer:
Learning pooling weights. Initialize as cotangent weights.

loss_pool = min E



========================
Loss function:

loss_mse + loss_pool + regularization 