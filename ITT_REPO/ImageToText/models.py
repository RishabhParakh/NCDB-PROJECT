from django.db import models
from django.contrib.auth.models import User
from autoslug import AutoSlugField
from django.utils import timezone

# Model to store the sources of images (renamed to ImageToTextImageSource)
#class ImageSource(models.Model):
#    sourcename = models.CharField(db_column='SourceName', max_length=150, blank=False, null=False)
#    orderid = models.IntegerField(db_column='OrderId', blank=True, null=True)
#    sourcelink = models.CharField(db_column='SourceLink', max_length=500, blank=True, null=True, default='')
#    description = models.TextField(db_column='Description', max_length=700, blank=True, null=True, default='')
#
#    def __str__(self):
#        return str(self.sourcename)
#
#    class Meta:
#        db_table = 'ImageSource'  # Same as class name
#        app_label = 'ImageToText'

# Model to store details about the uploaded images and their extracted text
class ProcessedImageDetail(models.Model):
    #image = models.ImageField(upload_to='uploaded_images/', db_column='Image')
    imageurl = models.CharField(db_column='ImageUrl', max_length=500)
    carid = models.CharField(db_column='CarId', max_length=30, blank=True, null=True)
    #description = models.TextField(db_column='Description', max_length=700, blank=True, null=True, default='')
    uploaded_date = models.DateTimeField(db_column='UploadedDate', default=timezone.now)
    processed_date = models.DateTimeField(db_column='ProcessedDate', blank=True, null=True)
    #imagesourceid = models.ForeignKey(ImageSource, db_column='ImageSource', null=True, on_delete=models.SET_NULL)
    uploader = models.ForeignKey(User, blank=True, null=True, on_delete=models.SET_NULL)
    uploader_username = AutoSlugField(populate_from='uploader')

    # New fields for car model prediction
    car_model = models.CharField(db_column='CarModel', max_length=255, blank=True, null=True)
    model_description = models.TextField(db_column='ModelDescription', max_length=700, blank=True, null=True)

    def __str__(self):
        return str(self.image)

    class Meta:
        db_table = 'ProcessedImageDetail'  # Same as class name
        app_label = 'ImageToText'

