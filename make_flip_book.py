
# GAN で生成している画像をパラパラ漫画にして流す

import os

cwd = os.getcwd()
print(cwd)

media_dir = os.path.join(cwd, "single_images")
print(media_dir)

pic_list = os.listdir(media_dir)
pic_list = sorted(pic_list)
print(len(pic_list))

"""
import cv2

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ナニコレ..
video = cv2.VideoWriter('video.mp4', fourcc, len(pic_list), (28, 28))

for i in range(0, len(pic_list), 100):
    target_file = os.path.join(media_dir, "single_image_{}.jpg".format(i) )
    pic = cv2.imread(target_file)
    video.write(pic)

video.release()
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
# make figure object
fig = plt.figure()

# make void list
pics = []

# 枚数分繋げた animation にする
for i in range(0, len(pic_list), 100):
    target_file = os.path.join(media_dir, "single_image_{}.jpg".format(i) )
    pil_obj = Image.open(target_file)
    pic = plt.imshow(pil_obj)
    im_arr = np.array(pil_obj)
    pic = plt.imshow(im_arr)
    pics.append(pic)

#print(pics)

# make animation
anim = animation.ArtistAnimation(fig, pics, interval=200, repeat_delay=1000)
anim.save(os.path.join(cwd, "animation.gif"))
"""

# make void list
pics = []

# 枚数分繋げた animation にする
for i in range(0, len(pic_list)//2):
    target_file = os.path.join(media_dir, "single_image_{}.jpg".format(i*100) )
    pil_obj = Image.open(target_file)
    pics.append(pil_obj)

#print(pics)
pics[0].save( os.path.join(cwd, "animation.gif"),
              save_all=True, append_images=pics[1:],
              optimize=False, duration=40, loop=0)

