from . import views
from django.urls import path,include

urlpatterns = [ 
    path('', views.welcome_page, name="welcome_page"),
    path('accounts/', include('allauth.urls')),
    
    #Restrict user from accessing here to force them use google auth
    path('accounts/login/', views.redirect_to_google_login, name="redirect_to_google_login"),
    
    path('lecturer/login/', views.lecturer_google_login, name='lecturer_google_login'),
    path('student/login/', views.student_google_login, name='student_google_login'),
    path("student/dashboard/", views.student_dashboard, name="student_dashboard"),
    path("lecturer/dashboard/", views.lecturer_dashboard, name="lecturer_dashboard"),
    path('student/profile/', views.student_profile, name="student_profile"),
    path('lecturer/profile/', views.lecturer_profile, name="lecturer_profile"),
    path('student_courses', views.student_courses, name="student_courses"),
    path('lecturer_courses', views.lecturer_courses, name="lecturer_courses"),
    path('course_enrollment', views.course_enrollment, name='course_enrollment'),
    path('lecturer_course_enrollment', views.lecturer_course_enrollment, name='lecturer_course_enrollment'),
    path('lecturer/create-attendance/', views.create_attendance, name='create_attendance'),
    path('student/settings/', views.student_settings, name='student_settings'),
    path('lecturer/settings/', views.lecturer_settings, name="lecturer_settings"),
    path('lecturer_course_unenrollment', views.lecturer_course_unenrollment, name='lecturer_course_unenrollment'),
    path('student/mark-attendance/<str:pk>', views.mark_attendance, name='mark_attendance'), 
    path('student/view-attendance/', views.view_attendance, name='view_attendance'),  
    path('lecturer/session/<int:pk>/', views.lecturer_session_detail, name='lecturer_session_detail'),
    path('lecturer/session/<int:pk>/close/', views.close_session_early, name='close_session_early'),
    path('lecturer/session/<int:pk>/download/', views.download_session_excel, name='download_session_excel'),
]

