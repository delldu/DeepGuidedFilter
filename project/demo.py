# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import image_autops

# image_autops.image_autops_predict("images/autops/*.png", "output/autops")

# image_autops.image_autops_predict("images/FiveK_C/*.jpg", "output/demo/H", horizon=True)
# image_autops.image_autops_predict("images/FiveK_C/*.jpg", "output/demo/V", horizon=False)

image_autops.image_autops_predict("images/hazy/*.jpg", "output/haze")
