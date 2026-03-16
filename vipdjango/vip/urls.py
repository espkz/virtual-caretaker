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
    path('professor/sessions/<int:session_id>/delete/', views.professor_delete_session, name='professor_delete_session'),
    path('professor/students/<int:student_id>/logs/delete/', views.professor_delete_student_logs, name='professor_delete_student_logs'),
    path('professor/logs/reset/', views.professor_reset_all_student_logs, name='professor_reset_all_student_logs'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('student/messages/<int:message_id>/tts/', views.student_message_tts, name='student_message_tts'),
    path('student/sessions/<int:session_id>/download/', views.student_download_session, name='student_download_session'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
