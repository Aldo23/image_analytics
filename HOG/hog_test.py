import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, exposure


# image = data.chelsea()
image = cv2.imread('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/examples/sc.jpg')
image_det = cv2.imread('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/examples/find_santa.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image_det = cv2.cvtColor(image_det, cv2.COLOR_RGB2BGR)
# image = image[0:220, 200:560]
# cv2.imshow('test',image)
# cv2.waitKey(0)

# create hog
fd, hog_image = hog(image, orientations=5, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

# create viz
fig, axs = plt.subplots(3, figsize=(9, 20), sharex=True, sharey=True,gridspec_kw={'hspace': 0.1})

axs[0].axis('on')
axs[0].imshow(image, cmap=plt.cm.gray)
axs[0].set_title('Input Image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 8))

axs[1].axis('on')
axs[1].imshow(hog_image_rescaled, cmap=plt.cm.gray)
axs[1].set_title('Histogram of Oriented Gradients')

axs[2].axis('on')
axs[2].imshow(image_det,cmap=plt.cm.gray)
axs[2].set_title('Face Recognition')

plt.savefig('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/examples/image_santa.png')
plt.show()


