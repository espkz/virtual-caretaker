from django.urls import path
from . import views

app_name = 'vip'
urlpatterns = [
    path('', views.home, name='home'),
    path('professor/', views.professor_dashboard, name='professor_dashboard'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
]