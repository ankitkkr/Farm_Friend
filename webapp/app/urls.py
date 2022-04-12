from django.urls import path
from app.views import home, prediction

urlpatterns = [
    path('', home ,name='home'),
    path('prediction', prediction ,name='prediction'),

]
