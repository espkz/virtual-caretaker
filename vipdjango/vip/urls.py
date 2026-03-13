from django.urls import path
from . import views

app_name = 'vip'
urlpatterns = [
    path('', views.home, name='home'),
    path('professor/', views.professor_dashboard, name='professor_dashboard'),
    path('professor/prompts/<int:prompt_id>/activate/', views.set_active_prompt, name='set_active_prompt'),
    path('professor/prompts/<int:prompt_id>/deactivate/', views.deactivate_prompt, name='deactivate_prompt'),
    path('professor/prompts/new/', views.create_prompt, name='create_prompt'),
    path('professor/prompts/<int:prompt_id>/edit/', views.edit_prompt, name='edit_prompt'),
    path('professor/students/<int:student_id>/logs/', views.professor_student_logs, name='professor_student_logs'),
    path('professor/sessions/<int:session_id>/', views.professor_session_detail, name='professor_session_detail'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
