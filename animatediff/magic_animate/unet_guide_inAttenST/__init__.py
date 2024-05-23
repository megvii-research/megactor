# 本方案的合成：
# 除以下内容外，其他基本等同于AnimateAnyone
# 不基于AnimateAnyone那种，将Condition经过PoseGuider后，直接Add在noise latent上
# noise_latent 先过 主干unet的conv_in变成320通道的tensor
# 3 512 512 的Condition先经过2D卷积（或者添加时序），经过几层卷积后变成128 64 64的tensor，
# 再使用out层映射到320 64 64的tensor
# 这里的out层使用零卷积
# 再将处理后的Condition与sample，直接相Add

