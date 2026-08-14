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
admin.site.register(AssessmentResponse)


@admin.register(AssessmentAttempt)
class AssessmentAttemptAdmin(admin.ModelAdmin):
    list_display  = ("student", "assessment", "attempt_number", "status",
                     "violation_count", "flagged", "submitted_at")
    list_filter   = ("status", "flagged", "assessment")
    search_fields = ("student__username", "student__email", "assessment__title")
    readonly_fields = ("violation_log", "violation_count", "flagged",
                       "auto_score", "manual_score", "total_score",
                       "total_possible_score", "started_at", "submitted_at")
