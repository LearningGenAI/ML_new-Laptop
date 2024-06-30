from django.contrib import admin
from django.urls import path, include
from firstapp import views

#Stage 6: Modifying URL
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name="index"),#This is how we include first_url   
    path('result/', views.result, name='result'), 
]