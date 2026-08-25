from django.urls import path
from . import views

app_name = 'vip'
urlpatterns = [
    path('', views.home, name='home'),
    path('account/', views.account_settings, name='account_settings'),
    path('professor/', views.professor_dashboard, name='professor_dashboard'),
    path('professor/accounts/', views.professor_manage_accounts, name='professor_manage_accounts'),
    path('professor/prompts/<int:prompt_id>/download/', views.professor_download_prompt_txt, name='professor_download_prompt_txt'),
    path('professor/prompts/template/download/', views.professor_download_prompt_template, name='professor_download_prompt_template'),
    path('professor/prompts/upload/', views.professor_upload_prompt_file, name='professor_upload_prompt_file'),
    path('professor/prompts/<int:prompt_id>/activate/', views.set_active_prompt, name='set_active_prompt'),
    path('professor/prompts/<int:prompt_id>/deactivate/', views.deactivate_prompt, name='deactivate_prompt'),
    path('professor/prompts/<int:prompt_id>/delete/', views.delete_prompt, name='delete_prompt'),
    path('professor/prompts/new/', views.create_prompt, name='create_prompt'),
    path('professor/prompts/<int:prompt_id>/edit/', views.edit_prompt, name='edit_prompt'),
    path('professor/students/<int:student_id>/logs/', views.professor_student_logs, name='professor_student_logs'),
    path('professor/sessions/<int:session_id>/', views.professor_session_detail, name='professor_session_detail'),
    path('professor/sessions/<int:session_id>/delete/', views.professor_delete_session, name='professor_delete_session'),
    path('professor/test-chat/', views.professor_test_chat, name='professor_test_chat'),
    path('professor/students/<int:student_id>/logs/delete/', views.professor_delete_student_logs, name='professor_delete_student_logs'),
    path('professor/logs/reset/', views.professor_reset_all_student_logs, name='professor_reset_all_student_logs'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('student/messages/<int:message_id>/tts/', views.student_message_tts, name='student_message_tts'),
    path('student/stream-tts/', views.student_stream_tts, name='student_stream_tts'),
    path('student/voice-timing/', views.student_voice_timing, name='student_voice_timing'),
    path('student/sessions/<int:session_id>/download/', views.student_download_session, name='student_download_session'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
