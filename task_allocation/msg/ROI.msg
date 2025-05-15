# ROI.msg
int32 id
geometry_msgs/Point centroid_world  # ROI 的世界坐标系质心 (x, y, 0)
int32 num_cells              # 组成 ROI 的栅格单元数量 (用于覆盖成本估算)
# 你可以根据需要添加其他字段，例如边界框等