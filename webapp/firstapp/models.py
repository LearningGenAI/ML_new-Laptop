from django.db import models

# Create your models here.

class ehr(models.Model):
    HAEMATOCRIT = models.FloatField()
    HAEMOGLOBIN = models.FloatField()
    ERYTHROCYTE = models.FloatField()
    LEUCOCYTE = models.FloatField()
    THROMBOCYTE = models.IntegerField()
    MCH = models.FloatField()
    MCHC = models.FloatField()
    MCV = models.FloatField()
    AGE = models.IntegerField()
    SEX = models.IntegerField()