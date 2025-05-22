from django.contrib import admin
from .models import StudentProfile, LecturerProfile, Course, AttendanceSession
 
admin.site.register(StudentProfile)
admin.site.register(LecturerProfile)
admin.site.register(Course)
admin.site.register(AttendanceSession)