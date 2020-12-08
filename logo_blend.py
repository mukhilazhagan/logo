# %%
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise
import os


def logo_create(ic_list, logo_list,file_name):

    len_list = len(logo_list)
    file_src = "./pcb_images/"+file_name+".png"
    img = plt.imread(file_src)
    #plt.imshow(img)

    ic_vert = ic_list[1]
    logo_vert = logo_list[1]
    ic_vert =[int(x) for x in ic_vert]
    logo_vert =[int(x) for x in logo_vert]
    img_dst = img[min(ic_vert[1::2]):max(ic_vert[1::2]),min(ic_vert[0::2]):max(ic_vert[0::2]),:]

    # Relative location to IC
    lg_vert = [[min(logo_vert[0::2]),min(logo_vert[1::2])],[max(logo_vert[0::2]),max(logo_vert[1::2])]]

    i_vert = [[min(ic_vert[0::2]),min(ic_vert[1::2])],[max(ic_vert[0::2]),max(ic_vert[1::2])]]

    # relative vertices
    rel_vert = [[lg_vert[0][0]-i_vert[0][0],lg_vert[0][1]-i_vert[0][1]],[lg_vert[0][0]-i_vert[0][0]+lg_vert[1][0]-lg_vert[0][0],lg_vert[0][1]-i_vert[0][1]+lg_vert[1][1]-lg_vert[0][1]]]

    print("Relative vertices of logo:", rel_vert)
    vert = rel_vert

    plt.imshow(img_dst)
    print(vert)
    l = vert[1][1]-vert[0][1]
    h = vert[1][0]-vert[0][0]

    print("Length:"+str(l)+"\n Height:"+str(h))
    if l/h>1.2 or l/h<0.8:
        print("Logo Spot is Rectangle\n")
        if l<50 and h<50:
            dir_str = "./sorted-logos-test/rectangleImages/A-BothAreLessThan50px"
            print("Logo is Small")
            blend_logo(img_dst, vert, dir_str, 0)
        elif l>h and h<50 :
            dir_str = "./sorted-logos-test/rectangleImages/B-LengthIsLarger"
            print("Logo Length is Large")
            blend_logo(img_dst, vert, dir_str, 0)
        elif h>l and l<50 :
            dir_str = "./sorted-logos-test/rectangleImages/C-HeightIsLarger"
            print("Logo Height is Large")
            blend_logo(img_dst, vert, dir_str, 0)
        else:
            dir_str = "./sorted-logos-test/rectangleImages/D-BothAreLarge"
            print("Logo is Large")
            blend_logo(img_dst, vert, dir_str, 1)
    else:
        print("Logo Spot is Square\n")
        if l<50 and h<50:
            dir_str = "./sorted-logos-test/squareImages/A-BothAreLessThan50px"
            print("Logo is Small")
            blend_logo(img_dst, vert, dir_str, 0)
        elif l>h and h<50 :
            dir_str = "./sorted-logos-test/squareImages/B-LengthIsLarger"
            print("Logo Length is Large")
            blend_logo(img_dst, vert, dir_str, 0)
        elif h>l and l<50 :
            dir_str = "./sorted-logos-test/squareImages/C-HeightIsLarger"
            print("Logo Height is Large")
            blend_logo(img_dst, vert, dir_str, 0)
        else:
            dir_str = "./sorted-logos-test/squareImages/D-BothAreLarge"
            print("Logo is Large")
            blend_logo(img_dst, vert, dir_str, 1)
    return

def blend_logo(img_ic, logo_vert, dir_str, option):
    
    # Debug
    #option = 0

    files_list = os.listdir(dir_str)
    vert = logo_vert
    #for i in range(len(files_list)):

    #for file_idx in range(len(files_list)):
    for file_idx in range(len(files_list)):
        # Logo to be Implanted
        src = plt.imread(dir_str+"/"+files_list[file_idx])
        plt.figure()
        plt.imshow(src)
        plt.title("Image to be implanted")

        plt.figure()
        plt.imshow(img_ic)
        plt.title("IC on which to be implanted")

        # Destination where Logo will be implanted
        dst = img_ic
        # RGB Image is loaded as float, converting to uint8
        dst = (dst*255).astype(np.uint8)
        plt.figure()
        plt.imshow(dst)
        plt.title("Destination where logo will be implanted")

        # Converting to Gray to perform histeq
        gray_dst = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)
        hist_eq = cv2.equalizeHist(gray_dst)
        plt.imshow(gray_dst)
            # Creating a 1 channel mask 
        mask = np.zeros(gray_dst.shape, gray_dst.dtype)
        mask = (mask[:,:]).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = cv2.rectangle(mask,tuple(vert[0]),tuple(vert[1]),(255,255,255),-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        plt.figure()
        plt.imshow(mask)
        plt.title("Mask region where logo will be implanted")

        output = cv2.inpaint(gray_dst,mask,3,cv2.INPAINT_NS)
        plt.imshow(output)

        # converting source to uint8
        before_reshape = (src*255).astype(np.uint8)
        # Resizing Logo source to same dimesnions as gap in IC
        after_reshape = cv2.resize(before_reshape,(vert[1][0]-vert[0][0],vert[1][1]-vert[0][1]))
        plt.imshow(after_reshape)
        # reallocating src
        src = after_reshape 

        # dst now becomes the inpainted output
        dst = output
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        plt.figure()
        plt.imshow(dst)
        plt.title("Inpainted Output")

        hist_full = cv2.calcHist([dst],[0],None,[256],[0,256])
            #plt.hist(dst.ravel(),256,[0,256]); plt.show()
        plt.figure()
        plt.plot(range(0,256),hist_full[:,0])
        plt.title("Histogram")

        for i in range(0,255):
            if hist_full[i]>1e+02:
                text_intensity = i

        print(text_intensity)
        idx_peak =np.argmax(hist_full)

        # OPTION 0 uses alpha composting
        if option >=0:

            #src is already 3 chanl if not usethis
            #src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
            save_src = "./modified_logos/ac_"+files_list[file_idx]
            plt.figure()
            plt.imshow(src)
            plt.title("Option 0: Source to be Implanted")
            ## Alternate code for alpha composting
            logo_mask = (src[:,:,0]>150)*255
            #plt.imshow(logo_mask)
            alpha =text_intensity/255
            print(alpha)
            for i in range(vert[0][1],vert[1][1]):
                for j in range(vert[0][0],vert[1][0]):
                    if logo_mask[i-vert[0][1],j-vert[0][0]]==255:
                        dst[i,j] = alpha* src[i-vert[0][1],j-vert[0][0]][0]
                        #dst[i-vert[0][1],j-vert[0][0]] = alpha* src[i-vert[0][1],j-vert[0][0]][0]
            plt.figure()
            plt.imshow(dst)
            gray_im = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)

            plt.imshow(gray_im)
            src_img = gray_im
            #src_img = np.array(src_img, dtype = 'float32')
            #noise_img_rgb =np.zeros((vert[1][1]-vert[0][1],vert[1][0]- vert[0][0],3))
            #noise_img_rgb = np.array(noise_img_rgb, dtype = 'float32')

            noise_img = random_noise(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]], mode='speckle', var= (np.std(src_img)/255)**2)
            noise_img = np.array(noise_img*255, dtype = 'uint8')
            plt.figure()
            plt.imshow(noise_img)
            plt.title("Speckled Image")
            plt.imshow(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]].astype(np.uint8))
            plt.title("Source Image")
            plt.figure()
            plt.imshow(src_img.astype(np.uint8))
            plt.title("Source Image Full")
            plt.figure()
            plt.imshow(gray_im)
            plt.title("Speckled Image")

            # Blurring the logo region

            src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]] = cv2.GaussianBlur(noise_img, (3,3), 0.5)
            plt.figure()
            plt.imshow(src_img,cmap='gray')
            plt.title("Blurred Image")
            cv2.imwrite(save_src, src_img)
        #
        if option == 1:

            save_src = "./modified_logos/sc_"+files_list[file_idx]
            plt.figure()
            plt.imshow(src)
            plt.title("Option 1: Source to be Implanted")

            thresh_mask = (src<125)*255
            plt.figure()
            plt.imshow(thresh_mask)
            plt.title("Mask for Seamless Cloning")

            mask_sc = ((text_intensity+idx_peak)//2)*np.ones(src.shape, np.uint8)
            plt.figure()
            print(mask_sc[:,:].shape)
            plt.imshow(mask_sc[:,:])
            plt.title("Mask for based on index")
            

            # %%

            #mask = 255 * np.ones(src.shape, src.dtype)
            #mask = np.asarray((src>125)*255).astype(np.uint8)
            center = ( (vert[0][0]+vert[1][0])//2, (vert[0][1]+vert[1][1])//2)
            #output_clone = cv2.seamlessClone(src, dst, mask_sc[:,:,0], center, cv2.MIXED_CLONE)
            output_clone = cv2.seamlessClone(src, dst, mask_sc[:,:,0], center, cv2.NORMAL_CLONE)
            plt.figure()
            plt.imshow(output_clone)
            plt.title("Seamless Cloning")
            dst = output_clone
            #Adding noise to the logo region
            gray_im = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)

            plt.imshow(gray_im)
            src_img = gray_im
            #src_img = np.array(src_img, dtype = 'float32')
            #noise_img_rgb =np.zeros((vert[1][1]-vert[0][1],vert[1][0]- vert[0][0],3))
            #noise_img_rgb = np.array(noise_img_rgb, dtype = 'float32')

            noise_img = random_noise(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]], mode='speckle', var= (np.std(src_img)/255)**2)
            noise_img = np.array(noise_img*255, dtype = 'uint8')
            plt.figure()
            plt.imshow(noise_img)
            plt.title("Speckled Image")
            plt.imshow(src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]].astype(np.uint8))
            plt.title("Source Image")
            plt.figure()
            plt.imshow(src_img.astype(np.uint8))
            plt.title("Source Image Full")
            plt.figure()
            plt.imshow(gray_im)
            plt.title("Speckled Image")

            # Blurring the logo region

            src_img[vert[0][1]:vert[1][1], vert[0][0]:vert[1][0]] = cv2.GaussianBlur(noise_img, (3,3), 0.5)
            plt.figure()
            plt.imshow(src_img,cmap='gray')
            plt.title("Blurred Image")
            cv2.imwrite(save_src, src_img)


'''
# %% Place to change image variace based scaling


# %%





'''
#Read in image and preprocess

    
