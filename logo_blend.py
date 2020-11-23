# %%
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise

# %%

img_marv = plt.imread('./ICs/marvell.png')
img_real = plt.imread('./ICs/realtek.png')
img_ti = plt.imread('./ICs/ti.png')
img_xil = plt.imread('./ICs/xilinx.png')

logo_3m = np.asarray(Image.open('./Logos/3M/0.png'))


vert_marv = [[99, 338], [467, 505]]
vert_real = [[25, 22], [139, 109]]
vert_ti = [[60, 19], [119, 75]]
vert_xil = [[514, 120], [566, 288]]


img_dst = img_ti
vert = vert_ti
img = logo_3m
# %%
plt.imshow(img_ti)
# %%
#plt.imshow(logo_marv)
#logo_marv.shape

c_sum =np.zeros(img.shape[1])
r_sum =np.zeros(img.shape[0])


for i in range(img.shape[0]):
    r_sum[i] = np.mean(img[i,:])
for j in range(img.shape[1]):
    c_sum[j] = np.mean(img[:,j])


# %%
thresh = 0.5
thresh_c_sum = (c_sum > 0.5)*1
thresh_r_sum = (r_sum > 0.5)*1

for i in range(img.shape[0]):
    if thresh_r_sum[i] == 1:
        r_index_min = i
        break 
for i in range(img.shape[1]):
    if thresh_c_sum[i] == 1:
        c_index_min = i
        break

for i in range(img.shape[0]-1, 0,-1):
    if thresh_r_sum[i] == 1:
        r_index_max = i
        break 
for i in range(img.shape[1]-1, 0,-1):
    if thresh_c_sum[i] == 1:
        c_index_max = i
        break

im_crop = img[r_index_min:r_index_max,c_index_min:c_index_max]

plt.imshow(im_crop)

# %%
plt.gray()
plt.figure()
plt.imshow(img)
plt.title("Original Image")
plt.figure()
plt.imshow(im_crop)
plt.title("Cropped Image")
plt.figure()
#plt.plot(c_sum)
plt.scatter(range(1,img.shape[1]+1),c_sum)
plt.title("Column Sum")
plt.figure()
#plt.plot(r_sum)
plt.scatter(range(1,img.shape[0]+1),r_sum)
plt.title("Row Sum")
# %%
plt.gray()
im_inv = (im_crop!=1)*1
plt.imshow(im_inv)
# %% Inpaint

src = im_inv
dst = img_dst
#src_mask = im_inv
dst = (dst*255).astype(np.uint8)
#mask = np.zeros(dst.shape)
plt.imshow(dst)
# %%
gray_dst = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)
hist_eq = cv2.equalizeHist(gray_dst)
plt.imshow(gray_dst)
# %%
mask = np.zeros(gray_dst.shape, gray_dst.dtype)
mask = (mask[:,:]).astype(np.uint8)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
mask = cv2.rectangle(mask,tuple(vert[0]),tuple(vert[1]),(255,255,255),-1)
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
# %%
plt.imshow(mask)
mask.shape
# %%
#dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
#rgb_dst = cv2.cvtColor(gray_dst, cv2.COLOR_GRAY2RGB)
#plt.imshow(rgb_dst)
#print(dst.shape)
# %%

#output = cv2.inpaint(dst,mask[:,:,0],5,cv2.INPAINT_TELEA)
#output = cv2.inpaint(dst,mask[:,:,0],3,cv2.INPAINT_NS)
output = cv2.inpaint(gray_dst,mask,3,cv2.INPAINT_NS)

print(output.shape)
plt.imshow(output)

# %%
before_reshape = (im_inv*255).astype(np.uint8)
after_reshape = cv2.resize(before_reshape,(vert[1][0]-vert[0][0],vert[1][1]-vert[0][1]))
plt.imshow(after_reshape)
# %%
src = after_reshape 
dst = output
dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

# %%
plt.imshow(src)

# %%
hist_full = cv2.calcHist([dst],[0],None,[256],[0,256])
#plt.hist(dst.ravel(),256,[0,256]); plt.show()
plt.plot(range(0,256),hist_full[:,0])

# %%
for i in range(0,255):
    if hist_full[i]>1e+02:
        text_intensity = i

print(text_intensity)
# %%
idx_peak =np.argmax(hist_full)
idx_peak
# %%
## Alternate code for alpha composting
logo_mask = (src[:,:,0]>150)*255
plt.imshow(logo_mask)
#alpha =idx_peak/255
alpha =text_intensity/255
print(alpha)


for i in range(vert[0][1],vert[1][1]):
    for j in range(vert[0][0],vert[1][0]):
        if logo_mask[i-vert[0][1],j-vert[0][0]]==255:
            dst[i,j] = alpha* src[i-vert[0][1],j-vert[0][0]][0]

plt.imshow(dst)
# %%
logo_mask[0,0]
# %% Place to change image variace based scaling

#img_var_cap = 255/(np.mean(dst)+2*np.std(dst))
#plt.imshow(dst)
#temp_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
#temp = temp_dst*img_var_cap
#temp.shape
#temp =temp.astype(np.uint8)
# %%
#for i in range(temp.shape[0]):
#    for j in range(temp.shape[1]):
#        if temp[i,j,]>255:
#            temp[i][j]=255

#lt.imshow(temp)
#print(temp.shape)

#temp_clr = cv2.cvtColor(temp,cv2.COLOR_GRAY2RGB)
# %%
thresh_mask = (src<125)*255
plt.imshow(thresh_mask)
# %%

#mask_sc = (idx_peak/255)*thresh_mask
#mask_sc = thresh_mask
#mask_sc = mask_sc.astype(np.uint8)
#print(mask_sc.shape)
#mask_sc = 

mask_sc = idx_peak*np.ones(src.shape, np.uint8)
print(mask_sc[:,:].shape)
plt.imshow(mask_sc[:,:])

# %%

#mask = 255 * np.ones(src.shape, src.dtype)
#mask = np.asarray((src>125)*255).astype(np.uint8)
center = ( (vert[0][0]+vert[1][0])//2, (vert[0][1]+vert[1][1])//2)
output_clone = cv2.seamlessClone(src, dst, mask_sc[:,:,0], center, cv2.MIXED_CLONE)
#output_clone=cv2.seamlessClone(src, dst, mask_sc[:,:,0], center, cv2.NORMAL_CLONE)

plt.imshow(output_clone)


# %%
cv2.imwrite("3m_synth_img_ti.jpg", output_clone)


#%%Read in image and preprocess

gray_im = cv2.cvtColor(output_clone,cv2.COLOR_RGB2GRAY)

plt.imshow(gray_im)
src_img = output_clone
#%%Adding noise to the logo region

src_img = gray_im
#src_img = np.array(src_img, dtype = 'float32')
#noise_img_rgb =np.zeros((vert[1][1]-vert[0][1],vert[1][0]- vert[0][0],3))
#noise_img_rgb = np.array(noise_img_rgb, dtype = 'float32')

noise_img = random_noise(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]], mode='speckle', var= np.std(src_img)/255)
#noise_img_rgb[:,:,0] = random_noise(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],0], mode='speckle', var=np.std(src_img[:,:,0])/255)
#noise_img_rgb[:,:,1] = random_noise(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],1], mode='speckle', var=np.std(src_img[:,:,1])/255)
#noise_img_rgb[:,:,2] = random_noise(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],2], mode='speckle', var=np.std(src_img[:,:,2])/255)

#noise = noise_img[:,:,0] - src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],0]


# %%

# %%
#noise_img_rgb[:,:,0] = src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],0]+noise
#noise_img_rgb[:,:,1] = src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],1]+noise
#noise_img_rgb[:,:,2] = src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0],2]+noise

#noise_img_rgb = np.array(noise_img_rgb*255, dtype = 'uint8')
noise_img = np.array(noise_img*255, dtype = 'uint8')
plt.figure()
#plt.imshow(noise_img_rgb)
plt.imshow(noise_img)
plt.title("Speckled Image")
# %%

plt.imshow(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]].astype(np.uint8))
plt.title("Source Image")
plt.figure()
plt.imshow(src_img.astype(np.uint8))
plt.title("Source Image Full")
plt.figure()
plt.imshow(gray_im)
plt.title("Speckled Image")
#%% Blurring the logo region

src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]] = cv2.GaussianBlur(noise_img, (3,3), 0.5)

plt.imshow(src_img)
plt.title("Blurred Image")

# %%
