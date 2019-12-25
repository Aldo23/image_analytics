import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2
import warnings
from skimage import  transform, color
warnings.filterwarnings("ignore")


print("Getting Positive Set Data ")
# Positive set of labelled people in the wild
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=120, resize=0.4)
positive_patches = faces.images # 13233 face images to use for training
print("Positive Set Done")


print("Making Negative Set data")
# Negative set
from skimage import  transform, color

imgs_to_use = ['hubble_deep_field', 'text', 'coins', 'moon',
               'page', 'clock','coffee','chelsea','horse']
images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]

# To make up the negative set we extract patches from images shipped
# with Scikit-Image at various scales
from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])
# 27000 images for negative set
print("Negative Set Done")




print("Combining the positive and negative set and extracting hog features")
# Combine positive and negative sets and extract hog features
from skimage import feature   # To use skimage.feature.hog()
from itertools import chain

X_train = np.array([feature.hog(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1




print("Training SVM classifier")
# Training a SVM classifier.
# We do gridsearch over some choices of SVM's C parameter to get best result
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]},cv=3)
grid.fit(X_train, y_train)
# grid.best_score_ to see the score
print("Best training score :  ",grid.best_score_)

print("Training the best estimator on full dataset")
# Taking the best estimator and train it on full dataset
model = grid.best_estimator_
model.fit(X_train, y_train)



print("Testing on a New Image")
# Detect Faces in a New Image
test_img = cv2.imread('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/examples/sc.jpg')
test_img = color.rgb2gray(test_img)
test_img = transform.rescale(test_img, 0.5)
test_img = test_img[0:180, 100:260]

# Plotting image
plt.imshow(test_img, cmap='gray')
plt.axis('off')



# Sliding Window function - Goes Over the image patch by patch
# and computes the HOG features for each patch.
def sliding_window(img, patch_size=positive_patches[0].shape,
                   istep=1, jstep=1, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


print("Extracting features from test image......")
# Apply sliding window function to test_img
indices, patches = zip(*sliding_window(test_img))
patches_hog = np.array([feature.hog(patch) for patch in patches])


print("Using model to evaluate if the patches contain a face or not")
# Use the model to evaluate if HOG patches of the image
# contains a face
labels = model.predict(patches_hog)
# labels.sum() for  number of face detections from all the patches in the image


print("Visualizing the result")
# Visualize the detections
fig, ax = plt.subplots()
ax.imshow(test_img, cmap='gray')
ax.axis('off')

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=5, facecolor='none'))
plt.savefig('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/examples/find_santa.png')
plt.show()




