# 本方案的合成：
# 除以下内容外，其他基本等同于MagicAnimate
# 也就是说这里不同于AnimateAnyone的所有层融合，只有mid和up层进行了referencenet和unet的融合
# 并且只有key和value进行了修改

# 3 512 512 的Condition先经过2D卷积（或者添加时序），经过几层卷积后变成128 64 64的tensor，再使用out层映射到4 64 64的tensor
# 这里的out层不使用零卷积
# 再将处理后的Condition与sample，在通道维度上进行concat。并送入主干unet的conv_in中
# 其中，conv_in的针对后4通道（Condition对应部分）使用零卷积进行初始化

# 参考图像ref-image和Condition都以原图形式送入PoseGuider
# Condition过多层假3D卷积（由2D扩展到3D的卷积）卷积得到b 128 f 64 64 condition_feature
# ref-image也过多层2D卷积，得到b 256 1 1 1的ref_feature
# 将ref_feature作为scale和shift，并处理condition_feature
# 类似于AdaL系列的方案，更适合风格化