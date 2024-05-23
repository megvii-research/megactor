# 这个版本的dit，噪声通过pos embed送入，条件通过另一个pos embed，以及0初始化的1D卷积送入
# 两者add后
# 前12层（down）走unet的，参数冻结不变，后15层走controlnet的，参数可训练，并进行0初始化的1D卷积送入unet