from django.shortcuts import render, redirect,get_object_or_404
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import StudentProfile, LecturerProfile, Course, User, AttendanceSession, AttendanceRecord
from django.utils.timezone import is_naive, make_aware, is_aware, localtime, get_default_timezone

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
from datetime import datetime, timezone as dt_timezone
from PIL import Image
import math 
import tempfile
import io
from django.conf import settings
import insightface
import logging
from functools import wraps
from django.db import transaction

logger = logging.getLogger(__name__)

def get_user_role(user):
    """Helper function to determine user role based on email domain."""
    return "student" if user.email.endswith("@run.edu.ng") else "lecturer"

# Initialize InsightFace model (lazy loading)
_face_app = None

def get_face_app():
    """Lazy load InsightFace model to avoid startup delay."""
    global _face_app
    if _face_app is None:
        try:
            _face_app = insightface.app.FaceAnalysis(name='buffalo_l')
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading InsightFace model: {e}")
            raise
    return _face_app

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
    """Extract face encoding using InsightFace library"""
    try:
        # Initialize InsightFace model
        face_app = get_face_app()
        
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image from {image_path}")
            return None
        
        # Detect faces and get embeddings
        faces = face_app.get(image)
        
        if not faces:
            logger.warning(f"No face detected in {image_path}")
            return None
        
        # Get the embedding (face encoding) of the first face
        face_embedding = faces[0].embedding
        
        logger.info(f"Successfully extracted face embedding from {image_path}")
        return face_embedding
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def compare_faces(known_embedding, candidate_embedding, threshold=0.5):
    """Compare two face embeddings using InsightFace library"""
    if known_embedding is None or candidate_embedding is None:
        logger.warning("One or both face embeddings are None")
        return False
    
    # Use cosine similarity for comparison
    try:
        # Calculate cosine similarity
        similarity = np.dot(known_embedding, candidate_embedding) / (
            np.linalg.norm(known_embedding) * np.linalg.norm(candidate_embedding)
        )
        
        # Convert similarity to distance (lower distance = more similar)
        distance = 1 - similarity
        
        logger.info(f"Face similarity: {similarity:.4f}, distance: {distance:.4f} (threshold: {threshold})")
        
        # Return True if distance is below threshold (faces match)
        return distance < threshold
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return False

def redirect_to_google_login(request):
    if request.method == 'POST':
        logger.info('Redirecting to welcome page')
        return redirect('welcome_page')

"""View for the student dashboard."""
@student_required
def student_dashboard(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    user_role = get_user_role(user)
    
    student_profile = StudentProfile.objects.get(student_name=user_obj)
    enrolled_courses = student_profile.courses_enrolled.all()
    attendances = AttendanceSession.objects.filter(course__in=enrolled_courses).order_by('-start_time')
    
    # Annotate attendance session status and check if student has marked attendance
    current_time = now()
    total_eligible = 0  # sessions that are expired (past the end time)
    total_attended = 0
    active_count = 0
    upcoming_count = 0

    for att in attendances:
        att.has_attended = att.records.filter(student=user).exists()
        att.formatted_start = format_lagos_time(att.start_time)
        att.formatted_end = format_lagos_time(att.end_time)
        # Pass raw UTC ISO timestamps to JS for countdowns
        att.start_time_iso = att.start_time.isoformat()
        att.end_time_iso = att.end_time.isoformat()
        att.attendance_count = att.records.count()
        att.hall = att.lecture_hall
        att.latitude = att.latitude
        att.longitude = att.longitude
        att.radius = att.radius

        if current_time < att.start_time:
            att.status_label = 'Upcoming'
            att.status_color = 'indigo'
            upcoming_count += 1
        elif current_time > att.end_time:
            att.status_label = 'Expired'
            att.status_color = 'red'
            total_eligible += 1
            if att.has_attended:
                total_attended += 1
        else:
            att.status_label = 'Active'
            att.status_color = 'green'
            active_count += 1

    # Overall attendance rate (only count sessions that have ended)
    attendance_rate_pct = round((total_attended / total_eligible * 100), 1) if total_eligible > 0 else 100

    # Next upcoming session for the "Next Class" hero widget
    upcoming_sessions = sorted(
        [a for a in attendances if a.status_label in ('Upcoming', 'Active')],
        key=lambda x: x.start_time
    )
    next_session = upcoming_sessions[0] if upcoming_sessions else None

    context = {
        'user_role': user_role,
        'attendances': attendances,
        'server_now_lagos': format_lagos_time(current_time),
        'enrolled_courses_count': enrolled_courses.count(),
        'active_count': active_count,
        'upcoming_count': upcoming_count,
        'attendance_rate_pct': attendance_rate_pct,
        'total_eligible': total_eligible,
        'total_attended': total_attended,
        'next_session': next_session,
        'student_profile': student_profile,
    }  
    return render(request, "student_dashboard.html", context)



"""View for the lecturer dashboard."""
@lecturer_required
def lecturer_dashboard(request): 
    user = request.user
    user_role = get_user_role(user)
      
    user_obj = User.objects.get(username=request.user)
    lecturer = LecturerProfile.objects.get(user=user_obj)
    scheduled_lectures = AttendanceSession.objects.filter(lecturer=lecturer).order_by('-start_time')
    
    # Annotate session status and calculate metrics
    current_time = now()
    active_count = 0
    upcoming_count = 0
    total_marked_all = 0
    for class_obj in scheduled_lectures:
        class_obj.marked_count = class_obj.records.count()
        class_obj.formatted_start = format_lagos_time(class_obj.start_time)
        class_obj.formatted_end = format_lagos_time(class_obj.end_time)
        class_obj.start_time_iso = class_obj.start_time.isoformat()
        class_obj.end_time_iso = class_obj.end_time.isoformat()
        class_obj.latitude = class_obj.latitude
        class_obj.longitude = class_obj.longitude
        class_obj.radius = class_obj.radius
        class_obj.hall = class_obj.lecture_hall
        class_obj.notes = class_obj.notes or ''
        total_marked_all += class_obj.marked_count
        if current_time < class_obj.start_time:
            class_obj.status_label = 'Upcoming'
            class_obj.status_color = 'indigo'
            upcoming_count += 1
        elif current_time > class_obj.end_time:
            class_obj.status_label = 'Expired'
            class_obj.status_color = 'red'
        else:
            class_obj.status_label = 'Active'
            class_obj.status_color = 'green'
            active_count += 1

    # Next active / upcoming session hero widget
    sorted_next = sorted(
        [c for c in scheduled_lectures if c.status_label in ('Upcoming', 'Active')],
        key=lambda x: x.start_time
    )
    next_session = sorted_next[0] if sorted_next else None

    context = {
        'user_role': user_role,
        'scheduled_lectures': scheduled_lectures,
        'lecturer': lecturer,
        'active_count': active_count,
        'upcoming_count': upcoming_count,
        'total_marked_all': total_marked_all,
        'next_session': next_session,
        'server_now_lagos': format_lagos_time(current_time),
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

    # Get current student's enrolled courses
    try:
        student_profile = StudentProfile.objects.get(student_name=request.user)
        student_courses = student_profile.courses_enrolled.all()
    except StudentProfile.DoesNotExist:
        student_courses = []
    
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
            lecturer_profile = LecturerProfile.objects.get(user=user)
            # Check if THIS lecturer has already enrolled for this course
            if lecturer_profile.courses_taught.filter(id=course.id).exists():
                messages.warning(request, f"You are already enrolled in '{course.course_title}'")  
                return redirect('lecturer_courses')
            else:
                lecturer_profile.courses_taught.add(course)
                messages.success(request, f"Successfully enrolled in '{course.course_title}'")            
        else:
            lecturer_profile  = LecturerProfile.objects.create(user=user)
            lecturer_profile.courses_taught.add(course)
            messages.success(request, f"Successfully enrolled in '{course.course_title}'")
         
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

def format_lagos_time(aware_dt):
    """Convert an aware datetime to Africa/Lagos (project TZ) and format as string."""
    if aware_dt is None:
        return "N/A"
    try:
        local_dt = localtime(aware_dt, timezone=get_default_timezone())
        return local_dt.strftime("%Y-%m-%d %H:%M:%S (Lagos/WAT)")
    except Exception:
        return aware_dt.strftime("%Y-%m-%d %H:%M:%S")


def get_aware_datetime(dt_str):
    """
    Parse a datetime string and return a timezone-aware datetime.
    
    Supports two formats from the client:
    1. UTC ISO string with 'Z' suffix (preferred, sent by client-side conversion JS)
       e.g. "2026-07-27T01:18:00.000Z"
    2. Naive datetime-local string (legacy, fallback):
       e.g. "2026-07-27T02:18" - interpreted as project default timezone (Africa/Lagos)
    """
    if not dt_str:
        raise ValueError("Empty datetime string provided")

    dt_str = dt_str.strip()

    # Case 1: Explicit UTC ISO format (ends with Z) or has explicit timezone offset
    if dt_str.endswith('Z'):
        # Replace Z with +00:00 for fromisoformat compatibility (Python <3.11)
        iso_str = dt_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        if is_naive(dt):
            dt = dt.replace(tzinfo=dt_timezone.utc)
        return dt

    # Case 2: Has explicit +HH:MM or -HH:MM offset
    try:
        dt = datetime.fromisoformat(dt_str)
        if is_aware(dt):
            return dt
    except ValueError:
        pass

    # Case 3: Naive datetime-local string from the browser
    # Interpret it as being in the project's default timezone (Africa/Lagos)
    dt = datetime.fromisoformat(dt_str)
    if is_naive(dt):
        dt = make_aware(dt, timezone=get_default_timezone())
    return dt

@lecturer_required
@transaction.atomic
def create_attendance(request):
    """Allow lecturers to create an attendance session with geofencing"""
    user_object = User.objects.get(username=request.user)
    lecturer_profile = get_object_or_404(LecturerProfile, user=user_object)
    courses = lecturer_profile.courses_taught.all()  # Get only courses the lecturer teaches
    
    logger.info(f"[CREATE_ATTENDANCE] Lecturer {request.user.username} has {courses.count()} courses taught")
    for course in courses:
        logger.info(f"[CREATE_ATTENDANCE] Course available: id={course.id} title={course.course_title}")

    if request.method == "POST":
        logger.info("[CREATE_ATTENDANCE] ========== POST received (create session) ==========")
        logger.debug("[CREATE_ATTENDANCE] POST keys: %s", list(request.POST.keys()))

        # Use .get() with graceful error messages (don't raise KeyError silently)
        course_id = request.POST.get('course')
        lecture_hall = request.POST.get('lecture_hall')
        start_time_raw = request.POST.get('start_time')
        end_time_raw = request.POST.get('end_time')
        notes = request.POST.get('notes', '')
        radius = request.POST.get('radius', '100')
        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')

        logger.info(
            "[CREATE_ATTENDANCE] Raw params received: course_id=%s hall=%s "
            "start=%s end=%s radius=%s lat=%s lon=%s",
            course_id, lecture_hall, start_time_raw, end_time_raw, radius, latitude, longitude,
        )

        # ---- Validate each field and show user-friendly specific errors ----
        errors = []

        if not course_id:
            errors.append("Please select a course from the dropdown.")
        if not lecture_hall or not lecture_hall.strip():
            errors.append("Please enter the lecture hall name.")
        if not start_time_raw:
            errors.append("Please select the session start time.")
        if not end_time_raw:
            errors.append("Please select the session end time.")
        if not latitude or not longitude:
            errors.append("Location data missing! Please allow location access (GPS) in your browser and refresh the page.")
        else:
            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                errors.append(f"Invalid GPS coordinates received (lat={latitude!r}, lon={longitude!r}).")

        if errors:
            combined = " ".join(f"• {e}" for e in errors)
            logger.warning("[CREATE_ATTENDANCE] VALIDATION FAILED: %s", combined)
            messages.error(request, "Could not create attendance session. " + combined)
            # Pass courses back so form re-renders correctly with dropdown
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)

        # ---- Fetch course ----
        try:
            course = Course.objects.get(id=course_id)
        except Course.DoesNotExist:
            logger.error("[CREATE_ATTENDANCE] Course id=%s not found", course_id)
            messages.error(request, f"Course with ID {course_id} does not exist.")
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)

        # ---- Parse and validate times ----
        try:
            start_dt = get_aware_datetime(start_time_raw)
            end_dt = get_aware_datetime(end_time_raw)
            logger.info(
                "[CREATE_ATTENDANCE] Times parsed: start(aware)=%s end(aware)=%s | "
                "start(Lagos)=%s end(Lagos)=%s",
                start_dt.isoformat(), end_dt.isoformat(),
                format_lagos_time(start_dt), format_lagos_time(end_dt),
            )
        except ValueError as ve:
            logger.exception("[CREATE_ATTENDANCE] Time parsing error: %s", ve)
            messages.error(
                request,
                f"Invalid date/time values. Start={start_time_raw!r}, End={end_time_raw!r}. "
                "Please try selecting times again."
            )
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)
        except Exception as ex:
            logger.exception("[CREATE_ATTENDANCE] Unexpected time parsing failure.")
            messages.error(request, f"Date/time error: {ex}")
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)

        if start_dt >= end_dt:
            logger.warning(
                "[CREATE_ATTENDANCE] start >= end rejected. start=%s end=%s (Lagos: %s vs %s)",
                start_dt, end_dt, format_lagos_time(start_dt), format_lagos_time(end_dt),
            )
            messages.error(
                request,
                f"Start time ({format_lagos_time(start_dt)}) must be BEFORE end time ({format_lagos_time(end_dt)})."
            )
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)

        # ---- Parse radius safely ----
        try:
            radius_int = int(radius) if radius else 100
            if radius_int < 10 or radius_int > 5000:
                raise ValueError("radius out of range")
        except (ValueError, TypeError):
            logger.warning("[CREATE_ATTENDANCE] Bad radius value: %r", radius)
            messages.error(request, "Geofencing radius must be between 10 and 5000 meters.")
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)

        # ---- Create the session ----
        try:
            session = AttendanceSession.objects.create(
                course=course,
                lecturer=lecturer_profile,
                lecture_hall=lecture_hall.strip(),
                start_time=start_dt,
                end_time=end_dt,
                notes=notes.strip(),
                latitude=latitude,
                longitude=longitude,
                radius=radius_int,
            )
            logger.info(
                "[CREATE_ATTENDANCE] SUCCESS: session id=%s created for course '%s' by lecturer %s. "
                "Lagos window: %s → %s",
                session.id, course.course_title, request.user.username,
                format_lagos_time(session.start_time), format_lagos_time(session.end_time),
            )
        except Exception as ex:
            logger.exception("[CREATE_ATTENDANCE] DB creation failed.")
            messages.error(request, f"Failed to save attendance session to database: {ex}")
            context = {'courses': courses}
            return render(request, 'lecturer_create_attendance.html', context)

        # Pre-generate the CSV spreadsheet template
        try:
            csv_path = generate_session_csv(session)
            logger.info("[CREATE_ATTENDANCE] CSV pre-generated at: %s", csv_path)
        except Exception as ex:
            logger.warning("[CREATE_ATTENDANCE] CSV generation failed (non-fatal): %s", ex)
            # Not fatal — session is created, CSV regenerated on download

        messages.success(
            request,
            f"Attendance session created successfully! 🎉 "
            f"Course: {course.course_title}. Window (Lagos): {format_lagos_time(start_dt)} → {format_lagos_time(end_dt)}"
        )
        return redirect('lecturer_dashboard')

    # GET request — fresh form
    logger.info("[CREATE_ATTENDANCE] Rendering create form for lecturer %s", request.user.username)
    context = {'courses': courses}
    return render(request, 'lecturer_create_attendance.html', context)

 


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def compare_faces_with_distance(known_embedding, candidate_embedding):
    if known_embedding is None or candidate_embedding is None:
        return False, 1.0
    try:
        similarity = np.dot(known_embedding, candidate_embedding) / (
            np.linalg.norm(known_embedding) * np.linalg.norm(candidate_embedding)
        )
        distance = float(1 - similarity)
        return distance < 0.5, distance
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return False, 1.0

def generate_session_csv(session):
    attendance_dir = "Attendance_records"
    os.makedirs(attendance_dir, exist_ok=True)
    # Sanitize course title for safe file name
    safe_title = "".join(c for c in session.course.course_title if c.isalnum() or c in (' ', '_', '-')).strip()
    csv_path = os.path.join(attendance_dir, f"{safe_title}_session_{session.id}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Matric Number", "Student Name", "Email", "Department", 
            "Faculty", "Year of Study", "Marked Time", "Distance (m)", "Status"
        ])
        
        records = session.records.select_related('student').all().order_by('marked_time')
        for rec in records:
            try:
                profile = StudentProfile.objects.get(student_name=rec.student)
                matric = profile.matric_number or "N/A"
                dept = profile.department or "N/A"
                fac = profile.faculty or "N/A"
                year = profile.year_of_study or "N/A"
            except StudentProfile.DoesNotExist:
                matric, dept, fac, year = "N/A", "N/A", "N/A", "N/A"
            
            local_time = rec.marked_time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                matric,
                rec.student.get_full_name() or rec.student.username,
                rec.student.email,
                dept,
                fac,
                year,
                local_time,
                f"{rec.distance:.2f}",
                rec.status
            ])
            
    # Also write a course master csv (backward compatibility)
    master_path = os.path.join(attendance_dir, f"{session.course.course_title}.csv")
    with open(master_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # We append simple logs as it was previously doing
        for rec in records:
            writer.writerow([rec.student.username, rec.student.email, rec.marked_time.strftime("%Y-%m-%d %H:%M:%S")])
            
    return csv_path


@student_required
@transaction.atomic
def mark_attendance(request, pk):
    user = request.user
    student_profile = get_object_or_404(StudentProfile, student_name=user)
    attendance_session = get_object_or_404(AttendanceSession, id=pk)

    # 1. Expiration and time checks
    current_time = now()
    current_time_lagos = format_lagos_time(current_time)
    start_lagos = format_lagos_time(attendance_session.start_time)
    end_lagos = format_lagos_time(attendance_session.end_time)

    logger.info(
        f"[MARK_ATTENDANCE_TIME_CHECK] Session %s: "
        "server_now=%s | session_start=%s | session_end=%s | "
        "raw_now=%s | raw_start=%s | raw_end=%s",
        pk, current_time_lagos, start_lagos, end_lagos,
        current_time.isoformat(),
        attendance_session.start_time.isoformat(),
        attendance_session.end_time.isoformat(),
    )

    if current_time < attendance_session.start_time:
        err = (
            f"This attendance session has not started yet. "
            f"Starts at {start_lagos}. "
            f"Current server time: {current_time_lagos}. "
            f"Session ends at {end_lagos}."
        )
        logger.warning("[MARK_ATTENDANCE_REJECTED_NOT_STARTED] %s", err)
        messages.error(request, err)
        return redirect('view_attendance')
        
    if current_time > attendance_session.end_time:
        err = (
            f"This attendance session has expired. "
            f"It ended at {end_lagos}. "
            f"Current server time: {current_time_lagos}. "
            f"(Started at {start_lagos}."
        )
        logger.warning("[MARK_ATTENDANCE_REJECTED_EXPIRED] %s", err)
        messages.error(request, err)
        return redirect('view_attendance')

    # 2. Check if student has already marked attendance
    if AttendanceRecord.objects.filter(session=attendance_session, student=user).exists():
        messages.info(request, "Attendance already marked.")
        return redirect('view_attendance')

    if request.method == "POST":
        student_lat = request.POST.get("latitude", "").strip()
        student_lon = request.POST.get("longitude", "").strip()

        if not student_lat or not student_lon:
            messages.error(request, "Location data is missing. Please enable GPS and try again.")
            return render(request, 'mark_attendance.html', {
                'attendance_session': attendance_session,
                'session_start_lagos': format_lagos_time(attendance_session.start_time),
                'session_end_lagos': format_lagos_time(attendance_session.end_time),
                'server_now_lagos': current_time_lagos,
            })

        try:
            student_lat = float(student_lat)
            student_lon = float(student_lon)
        except ValueError:
            messages.error(request, "Invalid location data received.")
            return render(request, 'mark_attendance.html', {
                'attendance_session': attendance_session,
                'session_start_lagos': format_lagos_time(attendance_session.start_time),
                'session_end_lagos': format_lagos_time(attendance_session.end_time),
                'server_now_lagos': current_time_lagos,
            })

        hall_lat = attendance_session.latitude
        hall_lon = attendance_session.longitude
        radius = attendance_session.radius

        # Calculate exact distance
        distance = calculate_distance(student_lat, student_lon, hall_lat, hall_lon)

        # 3. Check geofence
        if distance > radius:
            messages.error(request, f"You are not in the lecture hall! You are {distance:.1f} meters away (allowed radius: {radius}m).")
            return render(request, 'mark_attendance.html', {
                'attendance_session': attendance_session,
                'session_start_lagos': format_lagos_time(attendance_session.start_time),
                'session_end_lagos': format_lagos_time(attendance_session.end_time),
                'server_now_lagos': current_time_lagos,
            })

        captured_image_data = request.FILES.get('captured_image')
        if not captured_image_data:
            messages.error(request, "No captured image provided.")
            return render(request, 'mark_attendance.html', {
                'attendance_session': attendance_session,
                'session_start_lagos': format_lagos_time(attendance_session.start_time),
                'session_end_lagos': format_lagos_time(attendance_session.end_time),
                'server_now_lagos': current_time_lagos,
            })

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
                # Get face embeddings
                logger.info(f"Processing profile image: {student_profile.face_image.path}")
                profile_embedding = get_face_encoding(student_profile.face_image.path)
                logger.info(f"Processing captured image: {temp_captured_path}")
                captured_embedding = get_face_encoding(temp_captured_path)
                
                if profile_embedding is None:
                    messages.error(request, "Could not detect face in your profile image. Please upload a clear photo in settings.")
                    return redirect('student_settings')
                
                if captured_embedding is None:
                    messages.error(request, "Could not detect face in captured image. Please ensure your face is clearly visible and well-lit.")
                    return render(request, 'mark_attendance.html', {
                        'attendance_session': attendance_session,
                        'session_start_lagos': format_lagos_time(attendance_session.start_time),
                        'session_end_lagos': format_lagos_time(attendance_session.end_time),
                        'server_now_lagos': current_time_lagos,
                    })
                
                # Compare faces and get similarity score
                match, face_dist = compare_faces_with_distance(profile_embedding, captured_embedding)
                
                if match:
                    # Save AttendanceRecord
                    AttendanceRecord.objects.create(
                        session=attendance_session,
                        student=user,
                        latitude=student_lat,
                        longitude=student_lon,
                        distance=distance,
                        face_similarity=face_dist,
                        status="Present"
                    )
                    # Keep backward compatibility student_marked ManyToMany
                    attendance_session.student_marked.add(user)

                    # Update spreadsheet
                    generate_session_csv(attendance_session)

                    messages.success(request, f"Attendance marked successfully! You are {distance:.1f}m from the center.")
                    return redirect('view_attendance')
                else:
                    messages.error(request, "Face verification failed. The captured face doesn't match your profile. Please try again with better lighting and angle.")
            except Exception as e:
                logger.error(f"Face verification error: {str(e)}")
                import traceback
                traceback.print_exc()
                messages.error(request, f"Face verification error: {str(e)}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_captured_path):
                os.unlink(temp_captured_path)

        return render(request, 'mark_attendance.html', {
            'attendance_session': attendance_session,
            'session_start_lagos': format_lagos_time(attendance_session.start_time),
            'session_end_lagos': format_lagos_time(attendance_session.end_time),
            'server_now_lagos': current_time_lagos,
        })

    return render(request, 'mark_attendance.html', {
        'attendance_session': attendance_session,
        'session_start_lagos': format_lagos_time(attendance_session.start_time),
        'session_end_lagos': format_lagos_time(attendance_session.end_time),
        'server_now_lagos': current_time_lagos,
    })





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
    
    # Annotate attendance session status and check if student has marked attendance
    current_time = now()
    total_eligible = 0
    total_attended = 0
    active_count = 0
    upcoming_count = 0
    missed_count = 0

    for att in attendances:
        att.has_attended = att.records.filter(student=user).exists()
        att.formatted_start = format_lagos_time(att.start_time)
        att.formatted_end = format_lagos_time(att.end_time)
        att.start_time_iso = att.start_time.isoformat()
        att.end_time_iso = att.end_time.isoformat()
        att.attendance_count = att.records.count()
        att.hall = att.lecture_hall
        att.latitude = att.latitude
        att.longitude = att.longitude
        att.radius = att.radius
        att.session_id = att.id
        att.notes = att.notes or ''

        # Record for student: get their AttendanceRecord (if marked) to show distance / time
        if att.has_attended:
            rec = att.records.filter(student=user).first()
            if rec:
                att.student_marked_time = format_lagos_time(rec.marked_time)
                att.student_distance = round(rec.distance, 1)
                att.face_similarity = round(rec.face_similarity, 4) if rec.face_similarity is not None else None

        if current_time < att.start_time:
            att.status_label = 'Upcoming'
            att.status_color = 'indigo'
            upcoming_count += 1
        elif current_time > att.end_time:
            att.status_label = 'Expired'
            att.status_color = 'red'
            total_eligible += 1
            if att.has_attended:
                total_attended += 1
            else:
                missed_count += 1
        else:
            att.status_label = 'Active'
            att.status_color = 'green'
            active_count += 1

    attendance_rate_pct = round((total_attended / total_eligible * 100), 1) if total_eligible > 0 else 100

    context = {
        'user_role': user_role,
        'attendances': attendances,
        'server_now_lagos': format_lagos_time(current_time),
        'total_eligible': total_eligible,
        'total_attended': total_attended,
        'missed_count': missed_count,
        'active_count': active_count,
        'upcoming_count': upcoming_count,
        'attendance_rate_pct': attendance_rate_pct,
    }  
    return render(request, 'view_attendace.html', context)


# Lecturer views the details of a session
@lecturer_required
def lecturer_session_detail(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    session = get_object_or_404(AttendanceSession, id=pk, lecturer=lecturer)
    
    # Annotate session status
    current_time = now()
    session.formatted_start = format_lagos_time(session.start_time)
    session.formatted_end = format_lagos_time(session.end_time)
    session.start_time_iso = session.start_time.isoformat()
    session.end_time_iso = session.end_time.isoformat()
    session.attendance_count = session.records.count()
    if current_time < session.start_time:
        session.status_label = 'Upcoming'
        session.status_color = 'indigo'
    elif current_time > session.end_time:
        session.status_label = 'Expired'
        session.status_color = 'red'
    else:
        session.status_label = 'Active'
        session.status_color = 'green'
        
    records = session.records.select_related('student').all().order_by('marked_time')
    
    # Augment records with student profiles
    for rec in records:
        try:
            profile = StudentProfile.objects.get(student_name=rec.student)
            rec.matric = profile.matric_number
            rec.department = profile.department
            rec.faculty = profile.faculty
            rec.year = profile.year_of_study
        except StudentProfile.DoesNotExist:
            rec.matric = "N/A"
            rec.department = "N/A"
            rec.faculty = "N/A"
            rec.year = "N/A"
            
    context = {
        'session': session,
        'records': records,
        'user_role': 'lecturer',
        'server_now_lagos': format_lagos_time(current_time),
    }
    return render(request, 'lecturer_session_detail.html', context)


# Lecturer closes an active session early
@lecturer_required
@transaction.atomic
def close_session_early(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    session = get_object_or_404(AttendanceSession, id=pk, lecturer=lecturer)
    
    current_time = now()
    if session.start_time <= current_time < session.end_time:
        session.end_time = current_time
        session.save()
        
        # Regenerate CSV report upon expiration
        generate_session_csv(session)
        
        messages.success(request, "Attendance session has been closed early successfully.")
    else:
        messages.warning(request, "This session cannot be closed early because it is not currently active.")
        
    return redirect('lecturer_session_detail', pk=pk)


# Lecturer downloads the spreadsheet (CSV) dynamically
@lecturer_required
def download_session_excel(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    session = get_object_or_404(AttendanceSession, id=pk, lecturer=lecturer)
    
    # Generate/update spreadsheet
    csv_path = generate_session_csv(session)
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            safe_title = "".join(c for c in session.course.course_title if c.isalnum() or c in (' ', '_', '-')).strip()
            response['Content-Disposition'] = f'attachment; filename="{safe_title}_session_{session.id}.csv"'
            return response
    else:
        messages.error(request, "Spreadsheet report could not be generated.")
        return redirect('lecturer_session_detail', pk=pk)