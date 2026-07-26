from django.db import models
from django.contrib.auth import get_user_model
from datetime import datetime 

User = get_user_model()

class Course(models.Model):
    course_code = models.CharField(max_length=20, unique=True) 
    course_title = models.CharField(max_length=255)  
    credit = models.IntegerField()
    
    class Meta:
        indexes = [
            models.Index(fields=['course_code']),
            models.Index(fields=['course_title']),
        ]
    
    def __str__(self):
        return self.course_title
    
    
class StudentProfile(models.Model):
    student_name = models.ForeignKey(User, on_delete=models.CASCADE)
    department = models.CharField(max_length=255, null=True, default='')
    faculty = models.CharField(max_length=255, null=True, default='')
    matric_number = models.CharField(max_length=255, null=True, default='')
    year_of_study = models.IntegerField(default=100, null=True) 
    courses_enrolled  = models.ManyToManyField(Course,  related_name='students')
    face_image = models.ImageField(upload_to='faces/', null=True, blank=True)  
    
    class Meta:
        indexes = [
            models.Index(fields=['matric_number']),
            models.Index(fields=['department']),
        ]
    
    def __str__(self):
        return f"{self.student_name} ({self.matric_number})"

class LecturerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE) 
    staff_id = models.CharField(max_length=255, unique=True)
    department = models.CharField(max_length=100)  
    academic_rank = models.CharField(max_length=50) 
    office_location = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=15)
    courses_taught = models.ManyToManyField('Course', related_name='lecturers')  # Many-to-many relationship

    class Meta:
        indexes = [
            models.Index(fields=['staff_id']),
            models.Index(fields=['department']),
        ]

    def __str__(self):
        return f"{self.user} ({self.staff_id})"
     
class AttendanceSession(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    lecturer = models.ForeignKey(LecturerProfile, on_delete=models.CASCADE)
    lecture_hall = models.CharField(max_length=255)
    start_time = models.DateTimeField(default=datetime.now)
    end_time = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)
    student_marked = models.ManyToManyField(User, related_name='students')
    
    # Geofencing Location Data
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    radius = models.IntegerField(default=100, help_text="Geofence radius in meters")

    class Meta:
        indexes = [
            models.Index(fields=['start_time']),
            models.Index(fields=['course', 'lecturer']),
        ]

    def __str__(self):
        return f"Attendance for {self.course.course_title} by {self.lecturer.user}"


class AttendanceRecord(models.Model):
    session = models.ForeignKey(AttendanceSession, on_delete=models.CASCADE, related_name='records')
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='attendance_records')
    marked_time = models.DateTimeField(auto_now_add=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    distance = models.FloatField(help_text="Distance in meters from center")
    status = models.CharField(max_length=20, default="Present")
    face_similarity = models.FloatField(null=True, blank=True, help_text="Cosine similarity distance")

    class Meta:
        unique_together = ('session', 'student')
        indexes = [
            models.Index(fields=['session', 'student']),
        ]

    def __str__(self):
        return f"{self.student.username} marked present for {self.session.course.course_title}"

