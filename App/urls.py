from django.urls import path

from . import views

urlpatterns = [
    path("", views.welcome_page, name="welcome_page"),
    path("contact/", views.handle_contact_form, name="handle_contact_form"),
    # Override default allauth login form BEFORE project-level allauth include.
    # Prevents direct access to username/password login; forces Google OAuth.
    path(
        "accounts/login/",
        views.redirect_to_google_login,
        name="redirect_to_google_login",
    ),
    # Custom logout that redirects to welcome page
    path("logout/", views.custom_logout, name="custom_logout"),
    path("lecturer/login/", views.lecturer_google_login, name="lecturer_google_login"),
    path("student/login/", views.student_google_login, name="student_google_login"),
    path("student/dashboard/", views.student_dashboard, name="student_dashboard"),
    path("lecturer/dashboard/", views.lecturer_dashboard, name="lecturer_dashboard"),
    path("student/profile/", views.student_profile, name="student_profile"),
    path("lecturer/profile/", views.lecturer_profile, name="lecturer_profile"),
    path("student_courses", views.student_courses, name="student_courses"),
    path("lecturer_courses", views.lecturer_courses, name="lecturer_courses"),
    path("course_enrollment", views.course_enrollment, name="course_enrollment"),
    path(
        "lecturer_course_enrollment",
        views.lecturer_course_enrollment,
        name="lecturer_course_enrollment",
    ),
    path("lecturer/create-course/", views.create_course, name="create_course"),
    path(
        "lecturer/create-attendance/", views.create_attendance, name="create_attendance"
    ),
    path("student/settings/", views.student_settings, name="student_settings"),
    path("lecturer/settings/", views.lecturer_settings, name="lecturer_settings"),
    path(
        "lecturer_course_unenrollment",
        views.lecturer_course_unenrollment,
        name="lecturer_course_unenrollment",
    ),
    path(
        "student/mark-attendance/<str:pk>",
        views.mark_attendance,
        name="mark_attendance",
    ),
    path("student/view-attendance/", views.view_attendance, name="view_attendance"),
    path(
        "student/attendance-details/",
        views.student_attendance_detail,
        name="student_attendance_detail",
    ),
    path(
        "lecturer/session/<int:pk>/",
        views.lecturer_session_detail,
        name="lecturer_session_detail",
    ),
    path(
        "lecturer/session/<int:pk>/close/",
        views.close_session_early,
        name="close_session_early",
    ),
    path(
        "lecturer/session/<int:pk>/download/",
        views.download_session_excel,
        name="download_session_excel",
    ),
    path("reports/", views.lecturer_reports, name="lecturer_reports"),
    path(
        "millialms/lecturer/",
        views.millialms_lecturer_dashboard,
        name="millialms_lecturer_dashboard",
    ),
    path(
        "millialms/lecturer/create/",
        views.millialms_create_assessment,
        name="millialms_create_assessment",
    ),
    path(
        "millialms/lecturer/assessment/<int:pk>/",
        views.millialms_lecturer_assessment_detail,
        name="millialms_lecturer_assessment_detail",
    ),
    path(
        "millialms/lecturer/assessment/<int:pk>/edit/",
        views.millialms_edit_assessment,
        name="millialms_edit_assessment",
    ),
    path(
        "millialms/lecturer/assessment/<int:pk>/duplicate/",
        views.millialms_duplicate_assessment,
        name="millialms_duplicate_assessment",
    ),
    path(
        "millialms/lecturer/assessment/<int:pk>/delete/",
        views.millialms_delete_assessment,
        name="millialms_delete_assessment",
    ),
    path(
        "millialms/lecturer/assessment/<int:pk>/toggle/",
        views.millialms_toggle_assessment_publish,
        name="millialms_toggle_assessment_publish",
    ),
    path(
        "millialms/lecturer/assessment/<int:pk>/export/",
        views.millialms_export_results,
        name="millialms_export_results",
    ),
    path(
        "millialms/lecturer/attempt/<int:attempt_id>/grade/",
        views.millialms_grade_attempt,
        name="millialms_grade_attempt",
    ),
    path(
        "millialms/student/",
        views.millialms_student_dashboard,
        name="millialms_student_dashboard",
    ),
    path(
        "millialms/student/assessment/<int:pk>/",
        views.millialms_student_assessment_detail,
        name="millialms_student_assessment_detail",
    ),
    path(
        "millialms/student/assessment/<int:pk>/take/",
        views.millialms_take_assessment,
        name="millialms_take_assessment",
    ),
    path(
        "millialms/student/result/<int:attempt_id>/",
        views.millialms_student_result,
        name="millialms_student_result",
    ),
    path(
        "lecturer/roster/", views.lecturer_course_roster, name="lecturer_course_roster"
    ),
    path(
        "lecturer/student/<int:student_user_id>/",
        views.lecturer_student_detail,
        name="lecturer_student_detail",
    ),
    path(
        "lecturer/student/<int:student_user_id>/course/<int:course_id>/",
        views.lecturer_student_detail,
        name="lecturer_student_detail_course",
    ),
]
