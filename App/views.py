from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import StudentProfile, LecturerProfile, Course,User, AttendanceSession
from django.contrib import messages
from datetime import datetime
import base64
from django.core.files.base import ContentFile 
import csv
import os
import base64
import numpy as np
import cv2 
from django.utils.timezone import now
from datetime import datetime 
from PIL import Image
import math 
import tempfile
import io
from django.conf import settings
import face_recognition
import logging
from functools import wraps
from django.db import transaction

logger = logging.getLogger(__name__)

def get_user_role(user):
    """Helper function to determine user role based on email domain."""
    return "student" if user.email.endswith("@run.edu.ng") else "lecturer"

def student_required(view_func):
    """Decorator to ensure only students can access the view."""
    @wraps(view_func)
    @login_required(login_url='/')
    def wrapped_view(request, *args, **kwargs):
        if get_user_role(request.user) != "student":
            messages.error(request, "Access denied. This page is for students only.")
            return redirect('welcome_page')
        return view_func(request, *args, **kwargs)
    return wrapped_view

def lecturer_required(view_func):
    """Decorator to ensure only lecturers can access the view."""
    @wraps(view_func)
    @login_required(login_url='/')
    def wrapped_view(request, *args, **kwargs):
        if get_user_role(request.user) != "lecturer":
            messages.error(request, "Access denied. This page is for lecturers only.")
            return redirect('welcome_page')
        return view_func(request, *args, **kwargs)
    return wrapped_view

def get_face_encoding(image_path):
    """Extract face encoding using face_recognition library"""
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            logger.warning(f"No face detected in {image_path}")
            return None
        
        # Get face encoding (128-dimensional vector)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            logger.warning(f"Could not generate face encoding for {image_path}")
            return None
        
        logger.info(f"Successfully extracted face encoding from {image_path}")
        return face_encodings[0]  # Return the first face encoding
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def compare_faces(known_encoding, candidate_encoding, tolerance=0.6):
    """Compare two face encodings using face_recognition library"""
    if known_encoding is None or candidate_encoding is None:
        logger.warning("One or both face encodings are None")
        return False
    
    # Use face_recognition's built-in comparison
    try:
        # Compare faces using the face_recognition library
        # tolerance: How much distance between faces to consider it a match. Lower is more strict.
        # 0.6 is typical for face recognition
        matches = face_recognition.compare_faces([known_encoding], candidate_encoding, tolerance=tolerance)
        
        # Calculate distance for debugging
        distance = face_recognition.face_distance([known_encoding], candidate_encoding)[0]
        logger.info(f"Face distance: {distance:.4f} (threshold: {tolerance})")
        
        return matches[0]
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return False

def redirect_to_google_login(request):
    if request.method == 'POST':
        logger.info('Redirecting to welcome page')
        return redirect('welcome_page')

@login_required(login_url='/')
def redirect_after_login(request):
    """Redirect users to the correct dashboard based on their email."""
    user = request.user
    user_object = User.objects.get(username=user)
    user_role = get_user_role(user)
    
    # Check if profile exists, if it doesn't create it
    if user_role == "student":
        if StudentProfile.objects.filter(student_name=user_object).exists():
            pass
        else:
            student_profile = StudentProfile.objects.create(student_name=user_object)
            student_profile.save()
        return redirect("student_dashboard")  # Redirect to student dashboard
    
    # Check if profile exists, if it doesn't create it
    elif user_role == "lecturer":
        if LecturerProfile.objects.filter(user=user_object).exists():
            pass
        else:
            lecturer_profile = LecturerProfile.objects.create(user=user_object)
            lecturer_profile.save()
        return redirect("lecturer_dashboard")  # Redirect to lecturer dashboard

    # Default redirect if email format is unknown
    return redirect("welcome_page")  


"""View for the student dashboard."""
@student_required
def student_dashboard(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    user_role = get_user_role(user)
    
    student_profile = StudentProfile.objects.get(student_name=user_obj)
    enrolled_courses = student_profile.courses_enrolled.all()
    attendances = AttendanceSession.objects.filter(course__in=enrolled_courses).order_by('-start_time')
    
    context = {
        'user_role': user_role,
        'attendances': attendances 
    }  
    return render(request, "student_dashboard.html", context)


"""View for the lecturer dashboard."""
@lecturer_required
def lecturer_dashboard(request): 
    user = request.user
    user_role = get_user_role(user)
      
    user_obj = User.objects.get(username= request.user)
    lecturer = LecturerProfile.objects.get(user=user_obj)
    scheduled_lectures  = AttendanceSession.objects.filter(lecturer = lecturer)
    
    context = {
        'user_role': user_role,
        'scheduled_lectures':scheduled_lectures,
        'lecturer':lecturer,
        }  
    logger.info(f"Lecturer dashboard accessed by {user.username}")
    return render(request, "lecturer_dashboard.html", context) 

# Default page to welcome unauthenticated users
def welcome_page(request):
    return render(request, 'welcome_page.html')

def student_google_login(request):
    # Store the process in the session
    request.session['login_process'] = 'student' 
    return redirect('/accounts/google/login/')

def lecturer_google_login(request):
    # Store the process in the session
    request.session['login_process'] = 'lecturer' 
    return redirect('/accounts/google/login/')

def student_profile(request):
    user = User.objects.get(username= request.user)
    student_profile = StudentProfile.objects.get(student_name=user) 
    
    logged_user = request.user
    user_role = get_user_role(logged_user)
    
    enrolled_courses = student_profile.courses_enrolled.all() 
    context = {
        'enrolled_courses':enrolled_courses,
        'student_profile':student_profile,
        'user_role':user_role
    }
    return render(request, 'student_profile.html', context)



# Search courses & Courses page
@student_required
def student_courses(request):
    user = request.user
    user_role = get_user_role(user)

    if request.method == 'POST':
        course = request.POST['course']
        results = Course.objects.filter(course_title__icontains=course)
        if not results:
            messages.info(request, f" 'No Course Found on {course}'")
        return render(request, 'student_courses.html', {'results': results, 'user_role': user_role})

    student_courses = StudentProfile.objects.all()
    return render(request, 'student_courses.html', {'student_courses': student_courses, 'user_role': user_role})



# View to enroll course
@transaction.atomic
def course_enrollment(request):
    if request.method == "POST":
        student = request.POST.get('student')
        course_title = request.POST.get('course')
        
        # Validation
        if not student or not course_title:
            messages.error(request, "Missing required fields.")
            return redirect('student_courses')
        
        """ Get the user object of the person enrolling & the Course """
        try:
            user = User.objects.get(username= student)
            course = Course.objects.get(course_title=course_title)
        except User.DoesNotExist:
            messages.error(request, "User not found.")
            return redirect('student_courses')
        except Course.DoesNotExist:
            messages.error(request, "Course not found.")
            return redirect('student_courses')
        
        if StudentProfile.objects.filter(student_name = user).exists():
            
            student_profile = StudentProfile.objects.get(student_name=user)
            student_profile.courses_enrolled.add(course)
            messages.success(request, f"You've successfully enrolled for '{course}'")            
        else:
            student_profile  =StudentProfile.objects.create(student_name=user)
            student_profile.courses_enrolled.add(course)
         
        return redirect('student_courses')
    

# Lecturers Profile
def lecturer_profile(request):
    user = User.objects.get(username= request.user)
    lecturer_profile = LecturerProfile.objects.get(user=user) 
    
    courses_taught = lecturer_profile.courses_taught.all() 
    context = {
        'courses_taught':courses_taught,
        'lecturer_profile':lecturer_profile
    }
    return render(request, 'lecturer_profile.html', context)


# Search courses & Courses page
def lecturer_courses(request):
    user = request.user
    user_obj= User.objects.get(username=user)
    # If a form is submitted
    if request.method == 'POST':
        course = request.POST['course']
        # Filter Courses based on search 
        results = Course.objects.filter(course_title__icontains = course)
        # If not found print out an error message
        if not results: 
            messages.error(request, f" 'No Course Found on {course}'")
    else:
        # Else a form is not submitted, display the normal home page
        lecturer_profile = LecturerProfile.objects.get(user=user_obj) 
        courses_taught =  lecturer_profile.courses_taught.all()
        logger.info(f"Lecturer courses: {courses_taught}")
        
        return render(request, 'lecturer_courses.html', {'courses_taught':courses_taught})
    
    return render(request, 'lecturer_courses.html', {'results':results})


#Lecturer enroll for course
@transaction.atomic
def lecturer_course_enrollment(request):
    if request.method == "POST":
        lecturer = request.POST.get('lecturer')
        course_title = request.POST.get('course')
        
        # Validation
        if not lecturer or not course_title:
            messages.error(request, "Missing required fields.")
            return redirect('lecturer_courses')
        
        """ Get the user object of the person enrolling & the Course """
        try:
            user = User.objects.get(username= lecturer)
            course = Course.objects.get(course_title=course_title)
        except User.DoesNotExist:
            messages.error(request, "User not found.")
            return redirect('lecturer_courses')
        except Course.DoesNotExist:
            messages.error(request, "Course not found.")
            return redirect('lecturer_courses')
        
        # Check if lecturer exists
        if LecturerProfile.objects.filter(user = user).exists():
            #Check if a lecturer has enrolled for this course
            if LecturerProfile.objects.filter(courses_taught=course).exists():
                messages.info(request, f"This '{course}' has already been enrolled by a  Lecturer")  
                return redirect('lecturer_courses')
            else:
                lecturer_profile = LecturerProfile.objects.get(user=user)
                lecturer_profile.courses_taught.add(course)
                messages.success(request, f"You've successfully enrolled for '{course}' as a Lecturer")            
        else:
            lecturer_profile  = LecturerProfile.objects.create(user=user)
            lecturer_profile.courses_taught.add(course)
         
        return redirect('lecturer_courses')


#Lecturer Unenroll for course
@transaction.atomic
def lecturer_course_unenrollment(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    if request.method == 'POST':
        course_title = request.POST.get('course')
        
        # Validation
        if not course_title:
            messages.error(request, "Missing required field.")
            return redirect('lecturer_courses')
        
        try:
            lecturer_profile = LecturerProfile.objects.get(user=user_obj)
            course = Course.objects.get(course_title=course_title)
        except LecturerProfile.DoesNotExist:
            messages.error(request, "Lecturer profile not found.")
            return redirect('lecturer_courses')
        except Course.DoesNotExist:
            messages.error(request, "Course not found.")
            return redirect('lecturer_courses')
        
        lecturer_profile.courses_taught.remove(course)
        messages.success(request, f"You've successfully unenrolled for '{course}' as a Lecturer") 
        return redirect('lecturer_courses')

@lecturer_required
@transaction.atomic
def create_attendance(request):
    """Allow lecturers to create an attendance session with geofencing"""
    user_object = User.objects.get(username=request.user)
    lecturer_profile = get_object_or_404(LecturerProfile, user=user_object)
    courses = lecturer_profile.courses_taught.all()  # Get only courses the lecturer teaches

    if request.method == "POST":
        course_id = request.POST['course']
        lecture_hall = request.POST['lecture_hall']
        start_time = request.POST['start_time']
        end_time = request.POST['end_time']
        notes = request.POST.get('notes', '')

        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')

        if not latitude or not longitude:
            messages.error(request, "Location data missing! Please allow location access.")
            return redirect("create_attendance")

        course = get_object_or_404(Course, id=course_id)

        # Check if attendance session already exists
        if AttendanceSession.objects.filter(lecturer=lecturer_profile, course=course).exists():
            messages.warning(request, "Attendance session for this course already exists!")
        else:
            AttendanceSession.objects.create(
                course=course,
                lecturer=lecturer_profile,
                lecture_hall=lecture_hall,
                start_time=datetime.fromisoformat(start_time),
                end_time=datetime.fromisoformat(end_time),
                notes=notes,
                latitude=latitude,  # Store latitude
                longitude=longitude  # Store longitude
            )

            messages.success(request, "Attendance session created successfully!")
        return redirect('lecturer_dashboard')

    context = {'courses': courses}
    return render(request, 'lecturer_create_attendance.html', context)
 


def is_within_geofence(lat1, lon1, lat2, lon2, radius):
    """
    Check if the student is within the specified radius (meters) of the lecture hall.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # Distance in meters
    return distance <= radius


@student_required
@transaction.atomic
def mark_attendance(request, pk):
    user = request.user
    student_profile = get_object_or_404(StudentProfile, student_name=user)
    attendance_session = get_object_or_404(AttendanceSession, id=pk)

    if attendance_session.student_marked.filter(id=user.id).exists():
        messages.info(request, "Attendance already marked.")
        return redirect('view_attendance')

    if request.method == "POST":
        student_lat = request.POST.get("latitude", "").strip()
        student_lon = request.POST.get("longitude", "").strip()

        if not student_lat or not student_lon:
            messages.error(request, "Location data is missing. Please enable GPS and try again.")
            return render(request, 'mark_attendance.html')

        try:
            student_lat = float(student_lat)
            student_lon = float(student_lon)
        except ValueError:
            messages.error(request, "Invalid location data received.")
            return render(request, 'mark_attendance.html')

        hall_lat = attendance_session.latitude
        hall_lon = attendance_session.longitude

        if not is_within_geofence(student_lat, student_lon, hall_lat, hall_lon, 5000):
            messages.error(request, "You are not in the lecture hall! Attendance not marked.")
            return render(request, 'mark_attendance.html')

        captured_image_data = request.FILES.get('captured_image')
        if not captured_image_data:
            messages.error(request, "No captured image provided.")
            return render(request, 'mark_attendance.html')

        # Create a temporary file for the captured image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in captured_image_data.chunks():
                temp_file.write(chunk)
            temp_captured_path = temp_file.name

        try:
            # Check if student has a profile image
            if not student_profile.face_image:
                messages.error(request, "No profile image found! Please upload your profile image.")
                return redirect('student_settings')

            try:
                # Get face encodings
                print(f"DEBUG: Processing profile image: {student_profile.face_image.path}")
                profile_encoding = get_face_encoding(student_profile.face_image.path)
                print(f"DEBUG: Processing captured image: {temp_captured_path}")
                captured_encoding = get_face_encoding(temp_captured_path)
                
                if profile_encoding is None:
                    messages.error(request, "Could not detect face in your profile image. Please upload a clear photo in settings.")
                    return redirect('student_settings')
                
                if captured_encoding is None:
                    messages.error(request, "Could not detect face in captured image. Please ensure your face is clearly visible and well-lit.")
                    return render(request, 'mark_attendance.html')
                
                # Compare faces
                if compare_faces(profile_encoding, captured_encoding):
                    attendance_session.student_marked.add(user)

                    attendance_dir = "Attendance_records"
                    os.makedirs(attendance_dir, exist_ok=True)
                    csv_path = os.path.join(attendance_dir, f"{attendance_session.course.course_title}.csv")

                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([user.username, user.email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

                    messages.success(request, "Attendance marked successfully!")
                    return redirect('view_attendance')
                else:
                    messages.error(request, "Face verification failed. The captured face doesn't match your profile. Please try again with better lighting and angle.")
            except Exception as e:
                print(f"DEBUG: Face verification error: {str(e)}")
                import traceback
                traceback.print_exc()
                messages.error(request, f"Face verification error: {str(e)}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_captured_path):
                os.unlink(temp_captured_path)

        return render(request, 'mark_attendance.html')

    return render(request, 'mark_attendance.html')




# Lecturer Sets his profile
@lecturer_required
def lecturer_settings(request):
    user = request.user
    user_role = get_user_role(user)
    
    # Fetch or create lecturer profile
    lecturer, created = LecturerProfile.objects.get_or_create(user=user)

    if request.method == "POST":
        lecturer.staff_id = request.POST.get('staff_id')
        lecturer.department = request.POST.get('department')
        lecturer.academic_rank = request.POST.get('academic_rank')
        lecturer.office_location = request.POST.get('office_location')
        lecturer.phone_number = request.POST.get('phone_number')

        lecturer.save()
        messages.success(request, "Profile updated successfully!")
        return redirect('lecturer_settings')
    
    context = {
        'user_role': user_role,
        'lecturer': lecturer
    } 
    return render(request, 'lecturer_settings.html', context)


# Students Sets his profile
@student_required
def student_settings(request):
    user = request.user
    user_role = get_user_role(user)
    
    # Fetch or create student profile
    student, created = StudentProfile.objects.get_or_create(student_name=user)

    if request.method == "POST":
        student.matric_number = request.POST.get('matric_number')
        student.department = request.POST.get('department')
        student.faculty = request.POST.get('faculty')
        student.year_of_study = request.POST.get('year_of_study') 
        
        # Check if a face image was uploaded
        if 'face_image' in request.FILES:
            # Handle the face image upload
            face_image = request.FILES['face_image']
            # If the image is being saved as a file, you may want to convert it to an acceptable format
            # Here, we can use PIL to make sure it's in the correct format:
            image = Image.open(face_image)
            image = image.convert('RGB')  # Ensure it is in RGB format
            # Save it back as a new image
            face_image_io = io.BytesIO()
            image.save(face_image_io, format='JPEG')
            face_image_io.seek(0)

            # Save the image in the model
            student.face_image.save('face_image.jpg', ContentFile(face_image_io.read()))

        student.save()
        messages.success(request, "Profile updated successfully!")
        return redirect('student_settings')
    
    context = {
        'user_role': user_role,
        'student': student,
    } 
    return render(request, 'student_settings.html', context)


# Students view his attendance
@student_required
def view_attendance(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    user_role = get_user_role(user)
    
    student_profile = StudentProfile.objects.get(student_name=user_obj)
    enrolled_courses = student_profile.courses_enrolled.all()
    attendances = AttendanceSession.objects.filter(course__in=enrolled_courses).order_by('-start_time')
    
    context = {
        'user_role': user_role,
        'attendances': attendances 
    }  
    return render(request, 'view_attendace.html', context)