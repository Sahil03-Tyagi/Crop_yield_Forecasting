from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('predict/', views.predict, name="predict"),
    path('home/',views.home,name='home'),
    path('graph/',views.graph,name='graph'),
    path('about/',views.about,name='about')
]
