from skimage import feature
import numpy as np
from PIL import Image
import cv2


B_path = "/Users/fireis/Documents/masters/masters_video_gen/Em1_Fala1_CarolinaHolly_00001.jpg"
out_path = "/Users/fireis/Documents/masters/masters_video_gen/canny.jpg"

img = Image.open(B_path)
im_edge = np.zeros((256,256), np.uint8)
cv2.putText(im_edge, "TESTE", (128,230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

edges = feature.canny(np.array(img.convert('L')))
#edges = edges * (part_labels == 0)  # remove edges within face
im_edges = (edges * 255).astype(np.uint8)
out = Image.fromarray((im_edge))
#image_pil = Image.fromarray(out)
out.save(out_path)