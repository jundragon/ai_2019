import sys

try:
    from django.db import models
except  Exception:
    print("There was an error loading django modules. Do you have django installed?")
    sys.exit()


class Video(models.Model):
    name = models.CharField(max_length=100, default='')

    gt_id = models.BigIntegerField(default=0)                       # video 번호
    p_count = models.BigIntegerField(default=0)                     # people count
    fe_count = models.BigIntegerField(default=0)                    # fire extinguisher count
    fh_count = models.BigIntegerField(default=0)                    # fire hydrant count
    c_count = models.BigIntegerField(default=0)                     # car count
    b_count = models.BigIntegerField(default=0)                     # bicycle count
    m_count = models.BigIntegerField(default=0)                     # motorcycle count


class Frame(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)

    seq = models.BigIntegerField(default=0)
    description = models.TextField(blank=True)

    class Meta:
        order_with_respect_to = 'video'


# class Object(models.Model):
#     # 프레임별 검출 되는 객체
#     rawdata = models.ForeignKey(Rawdata, on_delete=models.CASCADE)
#     video = models.ForeignKey(Video, on_delete=models.CASCADE)
#     frame = models.ForeignKey(Frame, on_delete=models.CASCADE)
#
#     # class 종류는 "person", "fire_extinguisher", "fire_hydrant"
#     # "car", "bus", "truck", "bicycle", "motorcycle"
#     object_class = models.CharField(max_length=50, default='')
#     object_type = models.CharField(max_length=50, default='')       # 검출 방법 "detect", "track", "label"
#
#     # bndbox
#     bndbox_x = models.IntegerField(default=0)
#     bndbox_y = models.IntegerField(default=0)
#     bndbox_w = models.IntegerField(default=0)
#     bndbox_h = models.IntegerField(default=0)
#
#
# class Roi(models.Model):
#     # Region Of Interest 관심 영역, 추적 대상 객체
#     video = models.ForeignKey(Video, on_delete=models.CASCADE)
#
#     # class 종류는 "person", "fire_extinguisher", "fire_hydrant"
#     # "car", "bus", "truck", "bicycle", "motorcycle"
#     roi_class = models.CharField(max_length=50, default='')
#     description = models.TextField(blank=True)
#
#     def __str__(self):
#         return '{}_{}'.format(self.roi_class, self.id)
#
# class RoiFrame(models.Model):
#     roi = models.ForeignKey(Roi, on_delete=models.CASCADE)
#     rawdata = models.ForeignKey(Rawdata, on_delete=models.CASCADE)
#     video = models.ForeignKey(Video, on_delete=models.CASCADE)
#     frame = models.ForeignKey(Frame, on_delete=models.CASCADE)
#
#     # 프레임별 roi 의 position 정보
#     bndbox_x = models.IntegerField(default=0)
#     bndbox_y = models.IntegerField(default=0)
#     bndbox_w = models.IntegerField(default=0)
#     bndbox_h = models.IntegerField(default=0)
#
#
# class RoiObject(models.Model):
#     roi = models.ForeignKey(Roi, on_delete=models.CASCADE)
#     rawdata = models.ForeignKey(Rawdata, on_delete=models.CASCADE)
#     video = models.ForeignKey(Video, on_delete=models.CASCADE)
#     frame = models.ForeignKey(Frame, on_delete=models.CASCADE)
#     object = models.ForeignKey(Object, on_delete=models.CASCADE)
#
#
# class LabelObject(Object):
#     pose = models.CharField(max_length=50, default='Unspecified')   # human labeling,
#     truncated = models.FloatField(default=0.0)
#
#
# class DetectObject(Object):
#     algorithm_code = models.CharField(max_length=100)   # yolo, +
#     accuracy = models.FloatField(default=0.0)
#
#
# class TrackObject(Object):
#     roi_frame = models.ForeignKey(RoiFrame, on_delete=models.CASCADE)
#     algorithm_code = models.CharField(max_length=100)   # siammask, +
#     match_rate = models.FloatField(default=0.0)
#
#
# class AnalysisResult(models.Model):
#     # 전처리, 검출, 추적, 통합 분석 결과물 관리용
#     rawdata = models.ForeignKey(Rawdata, on_delete=models.CASCADE)
#     video = models.ForeignKey(Video, on_delete=models.CASCADE)
#     frame = models.ForeignKey(Frame, on_delete=models.CASCADE)
#
#     def generate_analysis_path(self, file_name):
#         return '{0}/analysis/{2}/{0}_{1:05d}/{3}'.format(
#             self.frame.video.rawdata.name,
#             self.frame.video.gt_id,
#             self.code,
#             file_name
#         )
#
#     code = models.CharField(max_length=50, default='None')
#     seq = models.BigIntegerField(default=0)
#     file = models.ImageField(upload_to=generate_analysis_path, default='no result')
#
#     created = models.DateTimeField(auto_now_add=True)
#     updated = models.DateTimeField(auto_now=True)
#
#     def __str__(self):
#         return "{0}_analysis_result".format(self.frame.file)
#
#
# class CompareObject(models.Model):
#     rawdata = models.ForeignKey(Rawdata, on_delete=models.CASCADE)
#     video = models.ForeignKey(Video, on_delete=models.CASCADE)
#     frame1 = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name='frame1')
#     frame2 = models.ForeignKey(Frame, on_delete=models.CASCADE, related_name='frame2')
#     object1 = models.ForeignKey(Object, on_delete=models.CASCADE, related_name='object1')
#     object2 = models.ForeignKey(Object, on_delete=models.CASCADE, related_name='object2')
#
#     b_intersection = models.BooleanField(default=False)     # intersection 유무
#
#     # intersection 하는 구역의 좌표, 겹치는 구역이 없으면 0
#     interbox_x = models.IntegerField(default=0)
#     interbox_y = models.IntegerField(default=0)
#     interbox_w = models.IntegerField(default=0)
#     interbox_h = models.IntegerField(default=0)
#
#     distance = models.FloatField(default=0.0)       # 중심 좌표 거리
#     overlap_size = models.FloatField(default=0.0)   # 겹치는 영역의 크기
#     union_size = models.FloatField(default=0.0)     # 합친 영역의 크기
#     iou_rate = models.FloatField(default=0.0)       # (겹치는 영역 / 합친 영역): 잘 검출 됐는지 평가 하는 지표