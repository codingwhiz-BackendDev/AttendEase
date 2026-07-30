from django.contrib import admin

from .models import (
    Assessment,
    AssessmentAttempt,
    AssessmentOption,
    AssessmentQuestion,
    AssessmentResponse,
    AttendanceSession,
    Course,
    LecturerProfile,
    StudentProfile,
)

admin.site.register(StudentProfile)
admin.site.register(LecturerProfile)
admin.site.register(Course)
admin.site.register(AttendanceSession)
admin.site.register(Assessment)
admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentOption)
admin.site.register(AssessmentAttempt)
admin.site.register(AssessmentResponse)
