# ROIList.msg
std_msgs/Header header       # 包含时间戳和坐标系 (例如 "map")
ROI[] rois     # ROI 对象的列表
nav_msgs/MapMetaData map_info #  同时发布相关的地图元数据