from django.db import models
from django.contrib.auth import get_user_model
from datetime import datetime 

User = get_user_model()

class Course(models.Model):
    course_code = models.CharField(max_length=20, unique=True) 
    course_title = models.CharField(max_length=255)  
    credit = models.IntegerField()
    
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

    def __str__(self):
        return f"Attendance for {self.course.course_title} by {self.lecturer.user}"
