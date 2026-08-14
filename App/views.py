import base64
import csv
import io
import json
import logging
import math
import os
import tempfile
from datetime import datetime
from datetime import timezone as dt_timezone
from App.universal_ai import get_universal_tutor
from functools import wraps
from collections import defaultdict

import numpy as np
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.core.paginator import EmptyPage, Paginator
from django.db import transaction
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils.timezone import (
    get_default_timezone,
    is_aware,
    is_naive,
    localtime,
    make_aware,
    now,
)
from PIL import Image

from .models import (
    Assessment,
    AssessmentAttempt,
    AssessmentOption,
    AssessmentQuestion,
    AssessmentResponse,
    AttendanceRecord,
    AttendanceSession,
    Course,
    LecturerProfile,
    StudentProfile,
    User,
)

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter for anti-cheat endpoint
_violation_rate_limiter = defaultdict(list)
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_REQUESTS = 10  # max requests per window

def _check_rate_limit(identifier):
    """Check if identifier has exceeded rate limit."""
    from django.utils.timezone import now as django_now
    now = django_now()
    # Clean old entries
    _violation_rate_limiter[identifier] = [
        ts for ts in _violation_rate_limiter[identifier] 
        if (now - ts).total_seconds() < _RATE_LIMIT_WINDOW
    ]
    # Check if limit exceeded
    if len(_violation_rate_limiter[identifier]) >= _RATE_LIMIT_MAX_REQUESTS:
        return False
    # Add current request
    _violation_rate_limiter[identifier].append(now)
    return True


MILLIALMS_SUPPORTED_TYPES = {
    AssessmentQuestion.TYPE_MCQ,
    AssessmentQuestion.TYPE_TRUE_FALSE,
    AssessmentQuestion.TYPE_FILL_BLANK,
    AssessmentQuestion.TYPE_SHORT_ANSWER,
    AssessmentQuestion.TYPE_ESSAY,
}


def normalize_answer(value):
    return " ".join((value or "").strip().lower().split())


def question_requires_manual_grading(question):
    return question.question_type in {
        AssessmentQuestion.TYPE_SHORT_ANSWER,
        AssessmentQuestion.TYPE_ESSAY,
    }


def get_assessment_status_label(assessment, current_time=None):
    current_time = current_time or now()
    if assessment.status != Assessment.STATUS_PUBLISHED:
        return (
            "Draft" if assessment.status == Assessment.STATUS_DRAFT else "Unpublished"
        )
    if assessment.start_time and current_time < assessment.start_time:
        return "Upcoming"
    if assessment.end_time and current_time > assessment.end_time:
        return "Closed"
    return "Live"


def get_assessment_type_label(assessment_type):
    return dict(Assessment.TYPE_CHOICES).get(assessment_type, assessment_type)


def auto_grade_response(question, response):
    if question.question_type in {
        AssessmentQuestion.TYPE_SHORT_ANSWER,
        AssessmentQuestion.TYPE_ESSAY,
    }:
        response.is_correct = False
        response.auto_score = 0
        return 0, True

    if question.question_type in {
        AssessmentQuestion.TYPE_MCQ,
        AssessmentQuestion.TYPE_TRUE_FALSE,
    }:
        correct_option = question.options.filter(is_correct=True).first()
        is_correct = bool(
            correct_option and response.selected_option_id == correct_option.id
        )
        response.is_correct = is_correct
        response.auto_score = question.points if is_correct else 0
        return response.auto_score, False

    if question.question_type == AssessmentQuestion.TYPE_FILL_BLANK:
        submitted = normalize_answer(response.text_answer)
        accepted = [
            normalize_answer(ans)
            for ans in (question.accepted_answers or "").splitlines()
            if normalize_answer(ans)
        ]
        is_correct = submitted in accepted if accepted else False
        response.is_correct = is_correct
        response.auto_score = question.points if is_correct else 0
        return response.auto_score, False

    response.is_correct = False
    response.auto_score = 0
    return 0, False


def grade_attempt(attempt):
    auto_total = 0
    manual_total = 0
    total_possible = 0
    manual_pending = False

    for response in attempt.responses.select_related("question", "selected_option"):
        question = response.question
        total_possible += float(question.points)
        auto_score, needs_manual = auto_grade_response(question, response)
        auto_total += float(auto_score)
        manual_total += float(response.manual_score or 0)
        if needs_manual and not response.graded_manually:
            manual_pending = True
        response.save()

    attempt.auto_score = round(auto_total, 2)
    attempt.manual_score = round(manual_total, 2)
    attempt.total_score = round(auto_total + manual_total, 2)
    attempt.total_possible_score = round(total_possible, 2)
    attempt.status = (
        AssessmentAttempt.STATUS_SUBMITTED
        if manual_pending
        else AssessmentAttempt.STATUS_GRADED
    )
    attempt.save()
    return attempt


def serialize_question_for_attempt(question, randomize_options=False):
    options = list(question.options.all())
    if randomize_options:
        import random

        random.shuffle(options)
    return {
        "question": question,
        "options": options,
        "requires_manual": question_requires_manual_grading(question),
    }


def get_user_role(user):
    """Helper function to determine user role based on email domain."""
    return "student" if user.email.endswith("@run.edu.ng") else "lecturer"


def handle_contact_form(request):
    """Handle contact form submission and send email to classmillia@gmail.com"""
    if request.method == "POST":
        institution = request.POST.get("institution", "")
        full_name = request.POST.get("full_name", "")
        role = request.POST.get("role", "")
        work_email = request.POST.get("work_email", "")
        deployment = request.POST.get("deployment", "")
        message = request.POST.get("message", "")

        # Create email body
        email_body = f"""
New Contact Form Submission from ClassMillia Landing Page

INSTITUTION: {institution}
NAME: {full_name}
ROLE: {role}
EMAIL: {work_email}
DEPLOYMENT PREFERENCE: {deployment}

MESSAGE:
{message}

---
Submitted from ClassMillia website
"""

        try:
            # Send email to classmillia@gmail.com
            send_mail(
                subject=f"ClassMillia Contact Form - {institution} - {full_name}",
                message=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL if hasattr(settings, 'DEFAULT_FROM_EMAIL') else work_email,
                recipient_list=["classmillia@gmail.com"],
                fail_silently=False,
            )
            return render(request, "welcome_page.html", {
                "contact_success": True,
                "contact_message": "Thank you! Your message has been sent to classmillia@gmail.com. We'll reply within 1 business day."
            })
        except Exception as e:
            logger.error(f"Error sending contact form email: {e}")
            return render(request, "welcome_page.html", {
                "contact_error": True,
                "contact_message": "Sorry, there was an error sending your message. Please email us directly at classmillia@gmail.com"
            })

    return redirect("welcome_page")


def custom_logout(request):
    """Custom logout view that redirects to welcome page."""
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("welcome_page")


# Initialize InsightFace model (lazy loading)
_face_app = None


def get_face_app():
    """Lazy load InsightFace model to avoid startup delay."""
    global _face_app
    if _face_app is None:
        try:
            import insightface

            # Explicitly pin the CPU provider. Render runs CPU-only, and newer
            # onnxruntime raises (instead of silently falling back) when a GPU
            # provider is requested but unavailable.
            # allowed_modules limits the pack to what we actually use
            # (face detection + recognition embedding), which cuts RAM/startup
            # significantly vs. loading the full buffalo_l pack (genderage,
            # 2d106 & 3d68 landmarks) — important on memory-limited hosts.
            _face_app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            # ctx_id=-1 => CPU mode (ctx_id=0 targets GPU device 0).
            _face_app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully in CPU mode")
        except Exception as e:
            logger.error(f"Error loading InsightFace model: {e}")
            raise
    return _face_app


def student_required(view_func):
    """Decorator to ensure only students can access the view."""

    @wraps(view_func)
    @login_required(login_url="/")
    def wrapped_view(request, *args, **kwargs):
        if get_user_role(request.user) != "student":
            messages.error(request, "Access denied. This page is for students only.")
            return redirect("welcome_page")
        return view_func(request, *args, **kwargs)

    return wrapped_view


def lecturer_required(view_func):
    """Decorator to ensure only lecturers can access the view."""

    @wraps(view_func)
    @login_required(login_url="/")
    def wrapped_view(request, *args, **kwargs):
        if get_user_role(request.user) != "lecturer":
            messages.error(request, "Access denied. This page is for lecturers only.")
            return redirect("welcome_page")
        return view_func(request, *args, **kwargs)

    return wrapped_view


def get_face_encoding(image_path):
    import cv2

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

        logger.info(
            f"Face similarity: {similarity:.4f}, distance: {distance:.4f} (threshold: {threshold})"
        )

        # Return True if distance is below threshold (faces match)
        return distance < threshold
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return False


def redirect_to_google_login(request):
    """Prevent direct access to the default allauth login form (username/password).
    Users must use the role-specific Google OAuth entry points instead.
    """
    logger.info("Intercepted direct access to /accounts/login/; redirecting to welcome page")
    return redirect("welcome_page")


"""View for the student dashboard."""


@student_required
def student_dashboard(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    user_role = get_user_role(user)

    student_profile = StudentProfile.objects.get(student_name=user_obj)
    enrolled_courses = student_profile.courses_enrolled.all()
    attendances = AttendanceSession.objects.filter(
        course__in=enrolled_courses
    ).order_by("-start_time")

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
            att.status_label = "Upcoming"
            att.status_color = "indigo"
            upcoming_count += 1
        elif current_time > att.end_time:
            att.status_label = "Expired"
            att.status_color = "red"
            total_eligible += 1
            if att.has_attended:
                total_attended += 1
        else:
            att.status_label = "Active"
            att.status_color = "green"
            active_count += 1

    # Overall attendance rate (only count sessions that have ended)
    attendance_rate_pct = (
        round((total_attended / total_eligible * 100), 1) if total_eligible > 0 else 100
    )

    # Next upcoming session for the "Next Class" hero widget
    upcoming_sessions = sorted(
        [a for a in attendances if a.status_label in ("Upcoming", "Active")],
        key=lambda x: x.start_time,
    )
    next_session = upcoming_sessions[0] if upcoming_sessions else None

    context = {
        "user_role": user_role,
        "attendances": attendances,
        "server_now_lagos": format_lagos_time(current_time),
        "enrolled_courses_count": enrolled_courses.count(),
        "active_count": active_count,
        "upcoming_count": upcoming_count,
        "attendance_rate_pct": attendance_rate_pct,
        "total_eligible": total_eligible,
        "total_attended": total_attended,
        "next_session": next_session,
        "student_profile": student_profile,
    }
    return render(request, "student_dashboard.html", context)


"""View for the lecturer dashboard."""


@lecturer_required
def lecturer_dashboard(request):
    user = request.user
    user_role = get_user_role(user)

    user_obj = User.objects.get(username=request.user)
    lecturer, created = LecturerProfile.objects.get_or_create(user=user_obj)
    # Show sessions for EVERY course this lecturer teaches, including those
    # created by co-lecturers of the same course (not just this lecturer's own).
    taught_courses = lecturer.courses_taught.all()
    scheduled_lectures = (
        AttendanceSession.objects.filter(course__in=taught_courses)
        .select_related("course", "lecturer", "lecturer__user")
        .order_by("-start_time")
    )

    # Annotate session status and calculate metrics
    current_time = now()
    active_count = 0
    upcoming_count = 0
    total_marked_all = 0
    for class_obj in scheduled_lectures:
        annotate_session_creator(class_obj, lecturer)
        class_obj.marked_count = class_obj.records.count()
        class_obj.formatted_start = format_lagos_time(class_obj.start_time)
        class_obj.formatted_end = format_lagos_time(class_obj.end_time)
        class_obj.start_time_iso = class_obj.start_time.isoformat()
        class_obj.end_time_iso = class_obj.end_time.isoformat()
        class_obj.latitude = class_obj.latitude
        class_obj.longitude = class_obj.longitude
        class_obj.radius = class_obj.radius
        class_obj.hall = class_obj.lecture_hall
        class_obj.notes = class_obj.notes or ""
        total_marked_all += class_obj.marked_count
        if current_time < class_obj.start_time:
            class_obj.status_label = "Upcoming"
            class_obj.status_color = "indigo"
            upcoming_count += 1
        elif current_time > class_obj.end_time:
            class_obj.status_label = "Expired"
            class_obj.status_color = "red"
        else:
            class_obj.status_label = "Active"
            class_obj.status_color = "green"
            active_count += 1

    # Next active / upcoming session hero widget
    sorted_next = sorted(
        [c for c in scheduled_lectures if c.status_label in ("Upcoming", "Active")],
        key=lambda x: x.start_time,
    )
    next_session = sorted_next[0] if sorted_next else None

    context = {
        "user_role": user_role,
        "scheduled_lectures": scheduled_lectures,
        "lecturer": lecturer,
        "active_count": active_count,
        "upcoming_count": upcoming_count,
        "total_marked_all": total_marked_all,
        "next_session": next_session,
        "server_now_lagos": format_lagos_time(current_time),
    }
    logger.info(f"Lecturer dashboard accessed by {user.username}")
    return render(request, "lecturer_dashboard.html", context)


# Default page to welcome unauthenticated users
def welcome_page(request):
    return render(request, "welcome_page.html")


def student_google_login(request):
    # Store the process in the session
    request.session["login_process"] = "student"
    return redirect("/accounts/google/login/")


def lecturer_google_login(request):
    # Store the process in the session
    request.session["login_process"] = "lecturer"
    return redirect("/accounts/google/login/")


@student_required
def student_profile(request):
    user = request.user
    student_profile = get_object_or_404(StudentProfile, student_name=user)

    user_role = get_user_role(user)

    enrolled_courses = student_profile.courses_enrolled.all()
    context = {
        "enrolled_courses": enrolled_courses,
        "student_profile": student_profile,
        "user_role": user_role,
    }
    return render(request, "student_profile.html", context)


# Search courses & Courses page
@student_required
def student_courses(request):
    user = request.user
    user_role = get_user_role(user)

    if request.method == "POST":
        course = request.POST["course"]
        results = Course.objects.filter(course_title__icontains=course)
        if not results:
            messages.info(request, f" 'No Course Found on {course}'")
        return render(
            request,
            "student_courses.html",
            {"results": results, "user_role": user_role},
        )

    # Get current student's enrolled courses
    try:
        student_profile = StudentProfile.objects.get(student_name=request.user)
        student_courses = student_profile.courses_enrolled.all()
    except StudentProfile.DoesNotExist:
        student_courses = []

    return render(
        request,
        "student_courses.html",
        {"student_courses": student_courses, "user_role": user_role},
    )


# View to enroll course
@student_required
@transaction.atomic
def course_enrollment(request):
    if request.method == "POST":
        course_title = request.POST.get("course")

        # Validation
        if not course_title:
            messages.error(request, "Missing required fields.")
            return redirect("student_courses")

        user = request.user
        try:
            course = Course.objects.get(course_title=course_title)
        except Course.DoesNotExist:
            messages.error(request, "Course not found.")
            return redirect("student_courses")

        student_profile, _ = StudentProfile.objects.get_or_create(student_name=user)
        if student_profile.courses_enrolled.filter(id=course.id).exists():
            messages.warning(request, f"You are already enrolled in '{course.course_title}'")
        else:
            student_profile.courses_enrolled.add(course)
            messages.success(request, f"You've successfully enrolled for '{course}'")

        return redirect("student_courses")

    return redirect("student_courses")


# Lecturers Profile
@lecturer_required
def lecturer_profile(request):
    user = request.user
    lecturer_profile = get_object_or_404(LecturerProfile, user=user)

    courses_taught = lecturer_profile.courses_taught.all()
    context = {"courses_taught": courses_taught, "lecturer_profile": lecturer_profile}
    return render(request, "lecturer_profile.html", context)


# Search courses & Courses page
@lecturer_required
def lecturer_courses(request):
    user = request.user
    lecturer_profile = get_object_or_404(LecturerProfile, user=user)
    courses_taught = lecturer_profile.courses_taught.all()

    # If a form is submitted (search)
    if request.method == "POST":
        search_query = request.POST.get("course", "").strip()
        if search_query:
            results = Course.objects.filter(course_title__icontains=search_query)
            if not results:
                messages.error(request, f"No Course Found for '{search_query}'")
        else:
            results = Course.objects.none()
        # Always provide both results AND courses_taught so the template renders fully
        return render(
            request,
            "lecturer_courses.html",
            {"results": results, "courses_taught": courses_taught},
        )

    # GET request — normal home page
    logger.info(f"Lecturer courses: {courses_taught}")
    return render(
        request, "lecturer_courses.html", {"courses_taught": courses_taught}
    )


# Lecturer enroll for course
@lecturer_required
@transaction.atomic
def lecturer_course_enrollment(request):
    if request.method == "POST":
        course_title = request.POST.get("course")

        # Validation
        if not course_title:
            messages.error(request, "Missing required fields.")
            return redirect("lecturer_courses")

        user = request.user
        try:
            course = Course.objects.get(course_title=course_title)
        except Course.DoesNotExist:
            messages.error(request, "Course not found.")
            return redirect("lecturer_courses")

        # Check if lecturer exists
        lecturer_profile, _ = LecturerProfile.objects.get_or_create(user=user)
        if lecturer_profile.courses_taught.filter(id=course.id).exists():
            messages.warning(
                request, f"You are already enrolled in '{course.course_title}'"
            )
        else:
            lecturer_profile.courses_taught.add(course)
            messages.success(
                request, f"Successfully enrolled in '{course.course_title}'"
            )

        return redirect("lecturer_courses")

    return redirect("lecturer_courses")


# Lecturer create new course
@lecturer_required
@transaction.atomic
def create_course(request):
    """Allow lecturers to create new courses"""
    user = request.user
    user_obj = User.objects.get(username=user)

    try:
        lecturer_profile = LecturerProfile.objects.get(user=user_obj)
    except LecturerProfile.DoesNotExist:
        messages.error(
            request, "Lecturer profile not found. Please complete your profile first."
        )
        return redirect("lecturer_settings")

    if request.method == "POST":
        course_code = request.POST.get("course_code", "").strip().upper()
        course_title = request.POST.get("course_title", "").strip()
        credit = request.POST.get("credit", "").strip()

        # Validation
        if not course_code or not course_title or not credit:
            messages.error(request, "All fields are required.")
            return render(request, "create_course.html")

        try:
            credit = int(credit)
            if credit <= 0:
                messages.error(request, "Credit units must be a positive number.")
                return render(request, "create_course.html")
        except ValueError:
            messages.error(request, "Credit units must be a valid number.")
            return render(request, "create_course.html")

        # A course is uniquely identified by its code. If it already exists
        # (created by another lecturer), we DON'T error out — instead we add
        # this lecturer as a co-lecturer, since a course can be taught by
        # multiple lecturers. We match by code first, then fall back to title.
        existing_course = (
            Course.objects.filter(course_code=course_code).first()
            or Course.objects.filter(course_title__iexact=course_title).first()
        )

        if existing_course:
            if lecturer_profile.courses_taught.filter(id=existing_course.id).exists():
                messages.info(
                    request,
                    f"You already teach '{existing_course.course_title}'. "
                    "It's ready in your course list.",
                )
            else:
                lecturer_profile.courses_taught.add(existing_course)
                logger.info(
                    f"Co-lecturer added: {existing_course.course_code} - "
                    f"{existing_course.course_title} now also taught by {user.username}"
                )
                messages.success(
                    request,
                    f"'{existing_course.course_title}' already exists — "
                    "you've been added as a co-lecturer!",
                )
            return redirect("lecturer_courses")

        # Create the course
        course = Course.objects.create(
            course_code=course_code, course_title=course_title, credit=credit
        )

        # Automatically assign the lecturer to the course
        lecturer_profile.courses_taught.add(course)

        logger.info(
            f"Course created: {course_code} - {course_title} by {user.username}"
        )
        messages.success(
            request,
            f"Course '{course_title}' created successfully and assigned to you!",
        )
        return redirect("lecturer_courses")

    return render(request, "create_course.html", {"lecturer_profile": lecturer_profile})


# Lecturer Unenroll for course
@lecturer_required
@transaction.atomic
def lecturer_course_unenrollment(request):
    user = request.user
    if request.method == "POST":
        course_title = request.POST.get("course")

        # Validation
        if not course_title:
            messages.error(request, "Missing required field.")
            return redirect("lecturer_courses")

        try:
            lecturer_profile = LecturerProfile.objects.get(user=user)
            course = Course.objects.get(course_title=course_title)
        except LecturerProfile.DoesNotExist:
            messages.error(request, "Lecturer profile not found.")
            return redirect("lecturer_courses")
        except Course.DoesNotExist:
            messages.error(request, "Course not found.")
            return redirect("lecturer_courses")

        lecturer_profile.courses_taught.remove(course)
        messages.success(
            request, f"You've successfully unenrolled for '{course}' as a Lecturer"
        )
        return redirect("lecturer_courses")

    return redirect("lecturer_courses")


def format_lagos_time(aware_dt):
    """Convert an aware datetime to Africa/Lagos (project TZ) and format as string."""
    if aware_dt is None:
        return "N/A"
    try:
        local_dt = localtime(aware_dt, timezone=get_default_timezone())
        return local_dt.strftime("%Y-%m-%d %H:%M:%S (Lagos/WAT)")
    except Exception:
        return aware_dt.strftime("%Y-%m-%d %H:%M:%S")


def annotate_session_creator(session, current_lecturer):
    """Attach creator info to a session for the multi-lecturer UI.

    A course can be taught by several lecturers, but each AttendanceSession
    records the lecturer who created it in `session.lecturer`. This marks
    whether the viewing lecturer is that creator and provides a display label
    so co-lecturers can tell who created each session.
    """
    creator_profile = session.lecturer
    creator_user = creator_profile.user if creator_profile else None
    session.creator_name = (
        (creator_user.get_full_name() or creator_user.username)
        if creator_user
        else "Unknown"
    )
    session.is_mine = bool(
        current_lecturer
        and creator_profile
        and creator_profile.id == current_lecturer.id
    )
    session.creator_label = "You" if session.is_mine else session.creator_name
    return session


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
    if dt_str.endswith("Z"):
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
    courses = (
        lecturer_profile.courses_taught.all()
    )  # Get only courses the lecturer teaches

    logger.info(
        f"[CREATE_ATTENDANCE] Lecturer {request.user.username} has {courses.count()} courses taught"
    )
    for course in courses:
        logger.info(
            f"[CREATE_ATTENDANCE] Course available: id={course.id} title={course.course_title}"
        )

    if request.method == "POST":
        logger.info(
            "[CREATE_ATTENDANCE] ========== POST received (create session) =========="
        )
        logger.debug("[CREATE_ATTENDANCE] POST keys: %s", list(request.POST.keys()))

        # Use .get() with graceful error messages (don't raise KeyError silently)
        course_id = request.POST.get("course")
        lecture_hall = request.POST.get("lecture_hall")
        start_time_raw = request.POST.get("start_time")
        end_time_raw = request.POST.get("end_time")
        notes = request.POST.get("notes", "")
        radius = request.POST.get("radius", "100")
        latitude = request.POST.get("latitude")
        longitude = request.POST.get("longitude")

        logger.info(
            "[CREATE_ATTENDANCE] Raw params received: course_id=%s hall=%s "
            "start=%s end=%s radius=%s lat=%s lon=%s",
            course_id,
            lecture_hall,
            start_time_raw,
            end_time_raw,
            radius,
            latitude,
            longitude,
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
            errors.append(
                "Location data missing! Please allow location access (GPS) in your browser and refresh the page."
            )
        else:
            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                errors.append(
                    f"Invalid GPS coordinates received (lat={latitude!r}, lon={longitude!r})."
                )

        if errors:
            combined = " ".join(f"• {e}" for e in errors)
            logger.warning("[CREATE_ATTENDANCE] VALIDATION FAILED: %s", combined)
            messages.error(request, "Could not create attendance session. " + combined)
            # Pass courses back so form re-renders correctly with dropdown
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)

        # ---- Fetch course ----
        try:
            course = Course.objects.get(id=course_id)
        except Course.DoesNotExist:
            logger.error("[CREATE_ATTENDANCE] Course id=%s not found", course_id)
            messages.error(request, f"Course with ID {course_id} does not exist.")
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)

        # ---- Parse and validate times ----
        try:
            start_dt = get_aware_datetime(start_time_raw)
            end_dt = get_aware_datetime(end_time_raw)
            logger.info(
                "[CREATE_ATTENDANCE] Times parsed: start(aware)=%s end(aware)=%s | "
                "start(Lagos)=%s end(Lagos)=%s",
                start_dt.isoformat(),
                end_dt.isoformat(),
                format_lagos_time(start_dt),
                format_lagos_time(end_dt),
            )
        except ValueError as ve:
            logger.exception("[CREATE_ATTENDANCE] Time parsing error: %s", ve)
            messages.error(
                request,
                f"Invalid date/time values. Start={start_time_raw!r}, End={end_time_raw!r}. "
                "Please try selecting times again.",
            )
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)
        except Exception as ex:
            logger.exception("[CREATE_ATTENDANCE] Unexpected time parsing failure.")
            messages.error(request, f"Date/time error: {ex}")
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)

        if start_dt >= end_dt:
            logger.warning(
                "[CREATE_ATTENDANCE] start >= end rejected. start=%s end=%s (Lagos: %s vs %s)",
                start_dt,
                end_dt,
                format_lagos_time(start_dt),
                format_lagos_time(end_dt),
            )
            messages.error(
                request,
                f"Start time ({format_lagos_time(start_dt)}) must be BEFORE end time ({format_lagos_time(end_dt)}).",
            )
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)

        # ---- Parse radius safely ----
        try:
            radius_int = int(radius) if radius else 100
            if radius_int < 10 or radius_int > 5000:
                raise ValueError("radius out of range")
        except (ValueError, TypeError):
            logger.warning("[CREATE_ATTENDANCE] Bad radius value: %r", radius)
            messages.error(
                request, "Geofencing radius must be between 10 and 5000 meters."
            )
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)

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
                session.id,
                course.course_title,
                request.user.username,
                format_lagos_time(session.start_time),
                format_lagos_time(session.end_time),
            )
        except Exception as ex:
            logger.exception("[CREATE_ATTENDANCE] DB creation failed.")
            messages.error(
                request, f"Failed to save attendance session to database: {ex}"
            )
            context = {"courses": courses}
            return render(request, "lecturer_create_attendance.html", context)

        # Pre-generate the CSV spreadsheet template
        try:
            csv_path = generate_session_csv(session)
            logger.info("[CREATE_ATTENDANCE] CSV pre-generated at: %s", csv_path)
        except Exception as ex:
            logger.warning(
                "[CREATE_ATTENDANCE] CSV generation failed (non-fatal): %s", ex
            )
            # Not fatal — session is created, CSV regenerated on download

        messages.success(
            request,
            f"Attendance session created successfully! 🎉 "
            f"Course: {course.course_title}. Window (Lagos): {format_lagos_time(start_dt)} → {format_lagos_time(end_dt)}",
        )
        return redirect("lecturer_dashboard")

    # GET request — fresh form
    logger.info(
        "[CREATE_ATTENDANCE] Rendering create form for lecturer %s",
        request.user.username,
    )
    context = {"courses": courses}
    return render(request, "lecturer_create_attendance.html", context)


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
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
    safe_title = "".join(
        c for c in session.course.course_title if c.isalnum() or c in (" ", "_", "-")
    ).strip()
    csv_path = os.path.join(attendance_dir, f"{safe_title}_session_{session.id}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Matric Number",
                "Student Name",
                "Email",
                "Department",
                "Faculty",
                "Year of Study",
                "Marked Time",
                "Distance (m)",
                "Status",
            ]
        )

        records = (
            session.records.select_related("student").all().order_by("marked_time")
        )
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
            writer.writerow(
                [
                    matric,
                    rec.student.get_full_name() or rec.student.username,
                    rec.student.email,
                    dept,
                    fac,
                    year,
                    local_time,
                    f"{rec.distance:.2f}",
                    rec.status,
                ]
            )

    # Also write a course master csv (backward compatibility)
    master_path = os.path.join(attendance_dir, f"{session.course.course_title}.csv")
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # We append simple logs as it was previously doing
        for rec in records:
            writer.writerow(
                [
                    rec.student.username,
                    rec.student.email,
                    rec.marked_time.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

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
        pk,
        current_time_lagos,
        start_lagos,
        end_lagos,
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
        return redirect("view_attendance")

    if current_time > attendance_session.end_time:
        err = (
            f"This attendance session has expired. "
            f"It ended at {end_lagos}. "
            f"Current server time: {current_time_lagos}. "
            f"(Started at {start_lagos}."
        )
        logger.warning("[MARK_ATTENDANCE_REJECTED_EXPIRED] %s", err)
        messages.error(request, err)
        return redirect("view_attendance")

    # 2. Check if student has already marked attendance
    if AttendanceRecord.objects.filter(
        session=attendance_session, student=user
    ).exists():
        messages.info(request, "Attendance already marked.")
        return redirect("view_attendance")

    if request.method == "POST":
        student_lat = request.POST.get("latitude", "").strip()
        student_lon = request.POST.get("longitude", "").strip()

        if not student_lat or not student_lon:
            messages.error(
                request, "Location data is missing. Please enable GPS and try again."
            )
            return render(
                request,
                "mark_attendance.html",
                {
                    "attendance_session": attendance_session,
                    "session_start_lagos": format_lagos_time(
                        attendance_session.start_time
                    ),
                    "session_end_lagos": format_lagos_time(attendance_session.end_time),
                    "server_now_lagos": current_time_lagos,
                },
            )

        try:
            student_lat = float(student_lat)
            student_lon = float(student_lon)
        except ValueError:
            messages.error(request, "Invalid location data received.")
            return render(
                request,
                "mark_attendance.html",
                {
                    "attendance_session": attendance_session,
                    "session_start_lagos": format_lagos_time(
                        attendance_session.start_time
                    ),
                    "session_end_lagos": format_lagos_time(attendance_session.end_time),
                    "server_now_lagos": current_time_lagos,
                },
            )

        hall_lat = attendance_session.latitude
        hall_lon = attendance_session.longitude
        radius = attendance_session.radius

        # Calculate exact distance
        distance = calculate_distance(student_lat, student_lon, hall_lat, hall_lon)

        # 3. Check geofence
        if distance > radius:
            messages.error(
                request,
                f"You are not in the lecture hall! You are {distance:.1f} meters away (allowed radius: {radius}m).",
            )
            return render(
                request,
                "mark_attendance.html",
                {
                    "attendance_session": attendance_session,
                    "session_start_lagos": format_lagos_time(
                        attendance_session.start_time
                    ),
                    "session_end_lagos": format_lagos_time(attendance_session.end_time),
                    "server_now_lagos": current_time_lagos,
                },
            )

        captured_image_data = request.FILES.get("captured_image")
        if not captured_image_data:
            messages.error(request, "No captured image provided.")
            return render(
                request,
                "mark_attendance.html",
                {
                    "attendance_session": attendance_session,
                    "session_start_lagos": format_lagos_time(
                        attendance_session.start_time
                    ),
                    "session_end_lagos": format_lagos_time(attendance_session.end_time),
                    "server_now_lagos": current_time_lagos,
                },
            )

        # Create a temporary file for the captured image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in captured_image_data.chunks():
                temp_file.write(chunk)
            temp_captured_path = temp_file.name

        try:
            # Student must have enrolled a face embedding in settings first.
            if not student_profile.face_embedding:
                messages.error(
                    request,
                    "No face enrolled! Please capture your face in settings first.",
                )
                return redirect("student_settings")

            try:
                # Stored enrollment embedding (from the DB) vs the freshly
                # captured face. We only run the model on the captured image.
                profile_embedding = np.array(
                    student_profile.face_embedding, dtype=np.float32
                )
                logger.info(f"Processing captured image: {temp_captured_path}")
                captured_embedding = get_face_encoding(temp_captured_path)

                if captured_embedding is None:
                    messages.error(
                        request,
                        "Could not detect face in captured image. Please ensure your face is clearly visible and well-lit.",
                    )
                    return render(
                        request,
                        "mark_attendance.html",
                        {
                            "attendance_session": attendance_session,
                            "session_start_lagos": format_lagos_time(
                                attendance_session.start_time
                            ),
                            "session_end_lagos": format_lagos_time(
                                attendance_session.end_time
                            ),
                            "server_now_lagos": current_time_lagos,
                        },
                    )

                # Compare faces and get similarity score
                match, face_dist = compare_faces_with_distance(
                    profile_embedding, captured_embedding
                )

                if match:
                    # Save AttendanceRecord
                    AttendanceRecord.objects.create(
                        session=attendance_session,
                        student=user,
                        latitude=student_lat,
                        longitude=student_lon,
                        distance=distance,
                        face_similarity=face_dist,
                        status="Present",
                    )
                    # Keep backward compatibility student_marked ManyToMany
                    attendance_session.student_marked.add(user)

                    # Update spreadsheet
                    generate_session_csv(attendance_session)

                    messages.success(
                        request,
                        f"Attendance marked successfully! You are {distance:.1f}m from the center.",
                    )
                    return redirect("view_attendance")
                else:
                    messages.error(
                        request,
                        "Face verification failed. The captured face doesn't match your profile. Please try again with better lighting and angle.",
                    )
            except Exception as e:
                logger.error(f"Face verification error: {str(e)}")
                import traceback

                traceback.print_exc()
                messages.error(request, f"Face verification error: {str(e)}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_captured_path):
                os.unlink(temp_captured_path)

        return render(
            request,
            "mark_attendance.html",
            {
                "attendance_session": attendance_session,
                "session_start_lagos": format_lagos_time(attendance_session.start_time),
                "session_end_lagos": format_lagos_time(attendance_session.end_time),
                "server_now_lagos": current_time_lagos,
            },
        )

    return render(
        request,
        "mark_attendance.html",
        {
            "attendance_session": attendance_session,
            "session_start_lagos": format_lagos_time(attendance_session.start_time),
            "session_end_lagos": format_lagos_time(attendance_session.end_time),
            "server_now_lagos": current_time_lagos,
        },
    )


# Lecturer Sets his profile
@lecturer_required
def lecturer_settings(request):
    user = request.user
    user_role = get_user_role(user)

    # Fetch or create lecturer profile
    lecturer, created = LecturerProfile.objects.get_or_create(user=user)

    if request.method == "POST":
        lecturer.staff_id = request.POST.get("staff_id")
        lecturer.department = request.POST.get("department")
        lecturer.academic_rank = request.POST.get("academic_rank")
        lecturer.office_location = request.POST.get("office_location")
        lecturer.phone_number = request.POST.get("phone_number")

        lecturer.save()
        messages.success(request, "Profile updated successfully!")
        return redirect("lecturer_settings")

    context = {"user_role": user_role, "lecturer": lecturer}
    return render(request, "lecturer_settings.html", context)


# Students Sets his profile
@student_required
def student_settings(request):
    user = request.user
    user_role = get_user_role(user)

    # Fetch or create student profile
    student, created = StudentProfile.objects.get_or_create(student_name=user)

    if request.method == "POST":
        student.matric_number = request.POST.get("matric_number")
        student.department = request.POST.get("department")
        student.faculty = request.POST.get("faculty")
        student.year_of_study = request.POST.get("year_of_study")

        # Face enrollment: we compute the embedding from the captured image
        # in memory and store ONLY the numeric vector. The photo is never saved.
        if "captured_face" in request.FILES:
            captured_face = request.FILES["captured_face"]

            # Normalise to RGB JPEG, then write to a temp file so InsightFace
            # (which reads from a path) can extract the embedding.
            image = Image.open(captured_face)
            image = image.convert("RGB")
            face_io = io.BytesIO()
            image.save(face_io, format="JPEG")
            face_io.seek(0)

            temp_face_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as temp_file:
                    temp_file.write(face_io.read())
                    temp_face_path = temp_file.name

                embedding = get_face_encoding(temp_face_path)

                if embedding is None:
                    messages.error(
                        request,
                        "We couldn't detect a clear face in your capture. "
                        "Please retake in good lighting with your face centered.",
                    )
                    return redirect("student_settings")

                # Store the embedding as a plain list of floats (JSON-safe).
                student.face_embedding = [float(x) for x in embedding]
            finally:
                if temp_face_path and os.path.exists(temp_face_path):
                    os.unlink(temp_face_path)

        student.save()
        messages.success(request, "Profile updated successfully!")
        return redirect("student_settings")

    context = {
        "user_role": user_role,
        "student": student,
    }
    return render(request, "student_settings.html", context)


# Students view his attendance
@student_required
def view_attendance(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    user_role = get_user_role(user)

    student_profile = StudentProfile.objects.get(student_name=user_obj)
    enrolled_courses = student_profile.courses_enrolled.all()
    attendances = AttendanceSession.objects.filter(
        course__in=enrolled_courses
    ).order_by("-start_time")

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
        att.notes = att.notes or ""

        # Record for student: get their AttendanceRecord (if marked) to show distance / time
        if att.has_attended:
            rec = att.records.filter(student=user).first()
            if rec:
                att.student_marked_time = format_lagos_time(rec.marked_time)
                att.student_distance = round(rec.distance, 1)
                att.face_similarity = (
                    round(rec.face_similarity, 4)
                    if rec.face_similarity is not None
                    else None
                )

        if current_time < att.start_time:
            att.status_label = "Upcoming"
            att.status_color = "indigo"
            upcoming_count += 1
        elif current_time > att.end_time:
            att.status_label = "Expired"
            att.status_color = "red"
            total_eligible += 1
            if att.has_attended:
                total_attended += 1
            else:
                missed_count += 1
        else:
            att.status_label = "Active"
            att.status_color = "green"
            active_count += 1

    attendance_rate_pct = (
        round((total_attended / total_eligible * 100), 1) if total_eligible > 0 else 100
    )

    context = {
        "user_role": user_role,
        "attendances": attendances,
        "server_now_lagos": format_lagos_time(current_time),
        "total_eligible": total_eligible,
        "total_attended": total_attended,
        "missed_count": missed_count,
        "active_count": active_count,
        "upcoming_count": upcoming_count,
        "attendance_rate_pct": attendance_rate_pct,
    }
    return render(request, "view_attendace.html", context)


@student_required
def student_attendance_detail(request):
    user = request.user
    user_obj = User.objects.get(username=user)
    current_time = now()

    try:
        sp = StudentProfile.objects.get(student_name=user_obj)
    except StudentProfile.DoesNotExist:
        sp = StudentProfile.objects.create(student_name=user_obj)

    course_filter = (request.GET.get("course") or "").strip()

    enrolled_courses_qs = sp.courses_enrolled.all().order_by("course_code")
    sessions_qs = (
        AttendanceSession.objects.filter(course__in=enrolled_courses_qs)
        .select_related("course", "lecturer", "lecturer__user")
        .order_by("-start_time")
    )

    if course_filter:
        try:
            cid = int(course_filter)
            sessions_qs = sessions_qs.filter(course_id=cid)
        except (ValueError, TypeError):
            course_filter = ""

    enrolled_courses = list(enrolled_courses_qs)
    sessions = list(sessions_qs)

    per_course = []
    total_completed = 0
    total_present = 0
    total_absent = 0
    active_count = 0
    upcoming_count = 0

    for c in enrolled_courses:
        if course_filter and str(c.id) != course_filter:
            continue
        course_sessions = [s for s in sessions if s.course_id == c.id]
        completed = [s for s in course_sessions if s.end_time < current_time]
        recs = AttendanceRecord.objects.filter(
            session__in=completed, student=user_obj
        ).select_related("session")
        present = recs.count()
        completed_count = len(completed)
        absent = max(0, completed_count - present)
        pct = round((present / completed_count * 100), 1) if completed_count > 0 else 0
        active = sum(
            1 for s in course_sessions if s.start_time <= current_time <= s.end_time
        )
        upcoming = sum(1 for s in course_sessions if s.start_time > current_time)

        total_completed += completed_count
        total_present += present
        total_absent += absent
        active_count += active
        upcoming_count += upcoming

        per_course.append(
            {
                "course": c,
                "completed_count": completed_count,
                "present": present,
                "absent": absent,
                "active_count": active,
                "upcoming_count": upcoming,
                "pct": pct,
            }
        )

    overall_pct = (
        round((total_present / total_completed * 100), 1)
        if total_completed > 0
        else 100
    )

    history = []
    for s in sessions:
        rec = AttendanceRecord.objects.filter(session=s, student=user_obj).first()
        if s.end_time < current_time:
            if rec:
                status_label = "Present"
                status_color = "green"
            else:
                status_label = "Absent"
                status_color = "red"
        elif s.start_time > current_time:
            status_label = "Upcoming"
            status_color = "indigo"
        else:
            if rec:
                status_label = "Present ✅"
                status_color = "green"
            else:
                status_label = "Pending"
                status_color = "orange"

        history.append(
            {
                "session": s,
                "date": s.start_time.strftime("%b %d, %Y"),
                "window": f"{s.start_time.strftime('%H:%M')}–{s.end_time.strftime('%H:%M')}",
                "hall": s.lecture_hall,
                "lecturer_name": s.lecturer.user.get_full_name()
                or s.lecturer.user.username,
                "status_label": status_label,
                "status_color": status_color,
                "marked_time": format_lagos_time(rec.marked_time) if rec else None,
                "distance": round(rec.distance, 1) if rec else None,
                "face_similarity": round(rec.face_similarity, 4)
                if (rec and rec.face_similarity is not None)
                else None,
                "radius": s.radius,
                "notes": s.notes or "",
            }
        )

    if total_completed == 0:
        trend_label = "No completed sessions yet"
        trend_level = "new"
    elif overall_pct >= 80:
        trend_label = "Excellent attendance"
        trend_level = "good"
    elif overall_pct >= 60:
        trend_label = "Needs attention"
        trend_level = "warn"
    else:
        trend_label = "Critical attendance risk"
        trend_level = "danger"

    context = {
        "user_role": "student",
        "student_profile": sp,
        "student_full_name": user.get_full_name() or user.username,
        "first_initial": (user.get_full_name() or user.username or "?")[0].upper(),
        "email": user.email,
        "has_face_image": bool(sp.face_embedding),
        "matric": sp.matric_number or "N/A",
        "department": sp.department or "N/A",
        "faculty": sp.faculty or "N/A",
        "year_of_study": sp.year_of_study
        if sp.year_of_study and sp.year_of_study != 100
        else "N/A",
        "per_course": per_course,
        "history": history,
        "enrolled_courses": enrolled_courses,
        "selected_course": course_filter,
        "total_completed": total_completed,
        "total_present": total_present,
        "total_absent": total_absent,
        "overall_pct": overall_pct,
        "active_count": active_count,
        "upcoming_count": upcoming_count,
        "trend_label": trend_label,
        "trend_level": trend_level,
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "student_attendance_detail.html", context)


# Lecturer views the details of a session
@lecturer_required
def lecturer_session_detail(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    # Any lecturer who teaches the course can view the session, not just its
    # creator. Ownership is still tracked via session.lecturer for the badge.
    session = get_object_or_404(
        AttendanceSession, id=pk, course__in=lecturer.courses_taught.all()
    )
    annotate_session_creator(session, lecturer)

    # Annotate session status
    current_time = now()
    session.formatted_start = format_lagos_time(session.start_time)
    session.formatted_end = format_lagos_time(session.end_time)
    session.start_time_iso = session.start_time.isoformat()
    session.end_time_iso = session.end_time.isoformat()
    session.attendance_count = session.records.count()
    if current_time < session.start_time:
        session.status_label = "Upcoming"
        session.status_color = "indigo"
    elif current_time > session.end_time:
        session.status_label = "Expired"
        session.status_color = "red"
    else:
        session.status_label = "Active"
        session.status_color = "green"

    records = session.records.select_related("student").all().order_by("marked_time")

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
        "session": session,
        "records": records,
        "user_role": "lecturer",
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "lecturer_session_detail.html", context)


# Lecturer closes an active session early
@lecturer_required
@transaction.atomic
def close_session_early(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    # Any lecturer teaching the course can close the session early.
    session = get_object_or_404(
        AttendanceSession, id=pk, course__in=lecturer.courses_taught.all()
    )

    current_time = now()
    if session.start_time <= current_time < session.end_time:
        session.end_time = current_time
        session.save()

        # Regenerate CSV report upon expiration
        generate_session_csv(session)

        messages.success(
            request, "Attendance session has been closed early successfully."
        )
    else:
        messages.warning(
            request,
            "This session cannot be closed early because it is not currently active.",
        )

    return redirect("lecturer_session_detail", pk=pk)


# Lecturer downloads the spreadsheet (CSV) dynamically
@lecturer_required
def download_session_excel(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    # Any lecturer teaching the course can download the session report.
    session = get_object_or_404(
        AttendanceSession, id=pk, course__in=lecturer.courses_taught.all()
    )

    # Generate/update spreadsheet
    csv_path = generate_session_csv(session)

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            response = HttpResponse(f.read(), content_type="text/csv")
            safe_title = "".join(
                c
                for c in session.course.course_title
                if c.isalnum() or c in (" ", "_", "-")
            ).strip()
            response["Content-Disposition"] = (
                f'attachment; filename="{safe_title}_session_{session.id}.csv"'
            )
            return response
    else:
        messages.error(request, "Spreadsheet report could not be generated.")
        return redirect("lecturer_session_detail", pk=pk)


# ========== COURSE MATERIALS + AI STUDY (UI preview) ==========
# NOTE: The AI model, file storage, text extraction and RAG pipeline are not
# wired yet — these views render the full interface so the workflow can be
# reviewed. Actions show a friendly "preview mode" notice instead of persisting.
MATERIAL_CATEGORIES = [
    {"key": "pdf", "label": "Course PDF", "icon": "fa-file-pdf", "color": "rose"},
    {"key": "notes", "label": "Lecture Notes", "icon": "fa-book-open", "color": "indigo"},
    {"key": "slides", "label": "Slides", "icon": "fa-file-powerpoint", "color": "amber"},
    {
        "key": "past_questions",
        "label": "Past Questions",
        "icon": "fa-clipboard-question",
        "color": "emerald",
    },
    {"key": "study", "label": "Study Material", "icon": "fa-paperclip", "color": "sky"},
]


def _pick_selected_course(courses, raw_id):
    """Resolve the selected course from a list + raw GET value (first as default)."""
    selected_id = None
    if raw_id:
        try:
            selected_id = int(raw_id)
        except (ValueError, TypeError):
            selected_id = None
    if selected_id is None and courses:
        selected_id = courses[0].id
    selected = next((c for c in courses if c.id == selected_id), None)
    return selected, selected_id


@lecturer_required
def lecturer_materials(request):
    """Lecturer hub to upload & manage course materials (UI preview)."""
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    courses = sorted(lecturer.courses_taught.all(), key=lambda c: c.course_code)

    selected_course, selected_course_id = _pick_selected_course(
        courses, request.GET.get("course")
    )

    if request.method == "POST":
        messages.info(
            request,
            "Preview mode — the materials & AI backend isn't connected yet, "
            "so this upload wasn't saved.",
        )
        target = "/lecturer/materials/"
        if selected_course_id:
            target += f"?course={selected_course_id}"
        return redirect(target)

    context = {
        "user_role": "lecturer",
        "lecturer": lecturer,
        "courses": courses,
        "selected_course": selected_course,
        "selected_course_id": selected_course_id,
        "material_categories": MATERIAL_CATEGORIES,
        "materials": [],  # backend not wired yet
    }
    return render(request, "lecturer_materials.html", context)


@student_required
def student_ai_study(request):
    """Student AI Study hub — course materials + AI study tools."""
    user_obj = User.objects.get(username=request.user)
    try:
        student_profile = StudentProfile.objects.get(student_name=user_obj)
    except StudentProfile.DoesNotExist:
        student_profile = StudentProfile.objects.create(student_name=user_obj)

    courses = sorted(
        student_profile.courses_enrolled.all(), key=lambda c: c.course_code
    )
    selected_course, selected_course_id = _pick_selected_course(
        courses, request.GET.get("course")
    )
    
    # Get course materials if available
    materials = []
    if selected_course:
        try:
            from App.models import CourseMaterial
            materials = list(CourseMaterial.objects.filter(course=selected_course).order_by('-uploaded_at'))
        except:
            # CourseMaterial model might not exist yet or migration not run
            materials = []

    context = {
        "user_role": "student",
        "student_profile": student_profile,
        "courses": courses,
        "selected_course": selected_course,
        "selected_course_id": selected_course_id,
        "materials": materials,
    }
    return render(request, "student_ai_study.html", context)


# ========== AI API ENDPOINTS ==========

@student_required
def ai_chat(request):
    """Universal AI Chat endpoint - handles ALL subjects and learning types"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        course_code = data.get('course_code', '')
        
        if not question:
            return JsonResponse({'error': 'Question required'}, status=400)
        
        # Get universal tutor
        student_id = str(request.user.id)
        tutor = get_universal_tutor(course_code, student_id)
        
        # Get universal teaching response
        result = tutor.teach(question)
        
        if 'error' in result:
            return JsonResponse({'error': result['error'], 'success': False}, status=500)
        
        return JsonResponse({
            'response': result['answer'],
            'citations': result.get('citations', []),
            'confidence': result.get('confidence', 0.5),
            'context_used': result.get('context_used', 0),
            'domain': result.get('domain', 'general'),
            'keyword': result.get('keyword', 'general'),
            'learning_style': result.get('learning_style', 'balanced'),
            'bloom_level': result.get('bloom_level', 'understand'),
            'critical_questions': result.get('critical_questions', []),
            'learning_state': result.get('learning_state', {}),
            'success': True
        })
            
    except Exception as e:
        logger.error(f"Universal AI Chat error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_summarize(request):
    """AI Summarize endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        course_code = data.get('course_code', '')
        
        if not topic:
            return JsonResponse({'error': 'Topic required'}, status=400)
        
        # Get materials context
        materials_text = ""
        if course_code:
            from App.models import Course, CourseMaterial
            try:
                course = Course.objects.get(course_code=course_code)
                materials = CourseMaterial.objects.filter(course=course)
                materials_text = get_materials_text(materials)
            except Course.DoesNotExist:
                pass
        
        assistant = AIStudyAssistant()
        summary, success = assistant.summarize(topic, course_code, materials_text)
        
        if success:
            return JsonResponse({'summary': summary, 'success': True})
        else:
            return JsonResponse({'error': summary, 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Summarize error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_explain(request):
    """AI Explain endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        concept = data.get('concept', '').strip()
        course_code = data.get('course_code', '')
        
        if not concept:
            return JsonResponse({'error': 'Concept required'}, status=400)
        
        # Get materials context
        materials_text = ""
        if course_code:
            from App.models import Course, CourseMaterial
            try:
                course = Course.objects.get(course_code=course_code)
                materials = CourseMaterial.objects.filter(course=course)
                materials_text = get_materials_text(materials)
            except Course.DoesNotExist:
                pass
        
        assistant = AIStudyAssistant()
        explanation, success = assistant.explain(concept, course_code, materials_text)
        
        if success:
            return JsonResponse({'explanation': explanation, 'success': True})
        else:
            return JsonResponse({'error': explanation, 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Explain error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_generate_quiz(request):
    """AI Quiz Generator endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        num_questions = int(data.get('num_questions', 10))
        difficulty = data.get('difficulty', 'Medium')
        topics = data.get('topics', '').strip()
        course_code = data.get('course_code', '')
        
        # Get materials context
        materials_text = ""
        if course_code:
            from App.models import Course, CourseMaterial
            try:
                course = Course.objects.get(course_code=course_code)
                materials = CourseMaterial.objects.filter(course=course)
                materials_text = get_materials_text(materials)
            except Course.DoesNotExist:
                pass
        
        assistant = AIStudyAssistant()
        quiz, success = assistant.generate_quiz(num_questions, difficulty, topics, course_code, materials_text)
        
        if success:
            return JsonResponse({'quiz': quiz, 'success': True})
        else:
            return JsonResponse({'error': quiz.get('error', 'Unknown error'), 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Quiz error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_generate_flashcards(request):
    """AI Flashcard Generator endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        num_cards = int(data.get('num_cards', 10))
        topics = data.get('topics', '').strip()
        course_code = data.get('course_code', '')
        
        # Get materials context
        materials_text = ""
        if course_code:
            from App.models import Course, CourseMaterial
            try:
                course = Course.objects.get(course_code=course_code)
                materials = CourseMaterial.objects.filter(course=course)
                materials_text = get_materials_text(materials)
            except Course.DoesNotExist:
                pass
        
        assistant = AIStudyAssistant()
        flashcards, success = assistant.generate_flashcards(num_cards, topics, course_code, materials_text)
        
        if success:
            return JsonResponse({'flashcards': flashcards, 'success': True})
        else:
            return JsonResponse({'error': flashcards.get('error', 'Unknown error'), 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Flashcards error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_practice_question(request):
    """AI Practice Mode - Generate Question endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        course_code = data.get('course_code', '')
        previous_questions = data.get('previous_questions', [])
        
        # Get materials context
        materials_text = ""
        if course_code:
            from App.models import Course, CourseMaterial
            try:
                course = Course.objects.get(course_code=course_code)
                materials = CourseMaterial.objects.filter(course=course)
                materials_text = get_materials_text(materials)
            except Course.DoesNotExist:
                pass
        
        assistant = AIStudyAssistant()
        question, success = assistant.practice_question(course_code, materials_text, previous_questions)
        
        if success:
            return JsonResponse({'question': question, 'success': True})
        else:
            return JsonResponse({'error': question.get('error', 'Unknown error'), 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Practice Question error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_evaluate_answer(request):
    """AI Practice Mode - Evaluate Answer endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        student_answer = data.get('student_answer', '').strip()
        expected_points = data.get('expected_points', [])
        course_code = data.get('course_code', '')
        
        if not question or not student_answer:
            return JsonResponse({'error': 'Question and answer required'}, status=400)
        
        assistant = AIStudyAssistant()
        evaluation, success = assistant.evaluate_answer(question, student_answer, expected_points, course_code)
        
        if success:
            return JsonResponse({'evaluation': evaluation, 'success': True})
        else:
            return JsonResponse({'error': evaluation.get('error', 'Unknown error'), 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Evaluate Answer error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_generate_diagram(request):
    """Generate visual diagrams for concepts"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        concept = data.get('concept', '').strip()
        course_code = data.get('course_code', '')
        
        if not concept:
            return JsonResponse({'error': 'Concept required'}, status=400)
        
        student_id = str(request.user.id)
        tutor = get_ultimate_tutor(course_code, student_id)
        
        diagram = tutor.create_concept_diagram(concept)
        
        if diagram.get('success'):
            return JsonResponse({'diagram': diagram, 'success': True})
        else:
            return JsonResponse({'error': diagram.get('error', 'Unknown error'), 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Diagram error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_interactive_practice(request):
    """Generate interactive practice with guided hints"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        difficulty = data.get('difficulty', 'medium')
        course_code = data.get('course_code', '')
        
        if not topic:
            return JsonResponse({'error': 'Topic required'}, status=400)
        
        student_id = str(request.user.id)
        tutor = get_ultimate_tutor(course_code, student_id)
        
        practice = tutor.generate_interactive_practice(topic, difficulty)
        
        if practice:
            return JsonResponse({'practice': practice, 'success': True})
        else:
            return JsonResponse({'error': 'Failed to generate practice', 'success': False}, status=500)
            
    except Exception as e:
        logger.error(f"AI Interactive Practice error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_research_assistant(request):
    """Research assistant endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        course_code = data.get('course_code', '')
        
        if not topic:
            return JsonResponse({'error': 'Topic required'}, status=400)
        
        student_id = str(request.user.id)
        tutor = get_universal_tutor(course_code, student_id)
        
        research = tutor.suggest_research(topic)
        
        return JsonResponse({'research': research, 'success': True})
            
    except Exception as e:
        logger.error(f"AI Research error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_debate_facilitator(request):
    """Debate facilitator endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        course_code = data.get('course_code', '')
        
        if not topic:
            return JsonResponse({'error': 'Topic required'}, status=400)
        
        student_id = str(request.user.id)
        tutor = get_universal_tutor(course_code, student_id)
        
        debate = tutor.generate_debate(topic)
        
        return JsonResponse({'debate': debate, 'success': True})
            
    except Exception as e:
        logger.error(f"AI Debate error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


@student_required
def ai_project_suggester(request):
    """Project suggestion endpoint"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        concept = data.get('concept', '').strip()
        difficulty = data.get('difficulty', 'medium')
        course_code = data.get('course_code', '')
        
        if not concept:
            return JsonResponse({'error': 'Concept required'}, status=400)
        
        student_id = str(request.user.id)
        tutor = get_universal_tutor(course_code, student_id)
        
        project = tutor.suggest_project(concept, difficulty)
        
        return JsonResponse({'project': project, 'success': True})
            
    except Exception as e:
        logger.error(f"AI Project error: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)


# ========== LECTURER REPORTS HUB (/reports) ==========
@lecturer_required
def lecturer_reports(request):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)

    search_q = (request.GET.get("q") or "").strip()
    status_filter = (request.GET.get("status") or "all").strip()
    course_filter = (request.GET.get("course") or "").strip()
    page_number = request.GET.get("page") or "1"

    # Include sessions for every course this lecturer teaches, so co-lecturers
    # of the same course all see each other's sessions in reports.
    taught_courses = lecturer.courses_taught.all()
    sessions_qs = (
        AttendanceSession.objects.filter(course__in=taught_courses)
        .select_related("course", "lecturer", "lecturer__user")
        .prefetch_related("records__student")
        .order_by("-start_time")
    )

    current_time = now()
    all_sessions = list(sessions_qs)
    session_course_ids = {s.course_id for s in all_sessions}

    course_profiles_map = {}
    if session_course_ids:
        enrolled_profiles = (
            StudentProfile.objects.filter(courses_enrolled__id__in=session_course_ids)
            .select_related("student_name")
            .prefetch_related("courses_enrolled")
            .distinct()
        )
        for sp in enrolled_profiles:
            for course in sp.courses_enrolled.all():
                if course.id in session_course_ids:
                    course_profiles_map.setdefault(course.id, []).append(sp)

    active_count = 0
    upcoming_count = 0
    expired_count = 0
    total_checked_in = 0

    for s in all_sessions:
        total_checked_in += s.records.count()
        if current_time < s.start_time:
            upcoming_count += 1
        elif current_time > s.end_time:
            expired_count += 1
        else:
            active_count += 1

    if search_q:
        sessions_qs = sessions_qs.filter(
            Q(course__course_code__icontains=search_q)
            | Q(course__course_title__icontains=search_q)
            | Q(lecture_hall__icontains=search_q)
            | Q(notes__icontains=search_q)
        )
    if course_filter:
        try:
            cid = int(course_filter)
            sessions_qs = sessions_qs.filter(course_id=cid)
        except (ValueError, TypeError):
            course_filter = ""
    if status_filter == "Active":
        sessions_qs = sessions_qs.filter(
            start_time__lte=current_time, end_time__gte=current_time
        )
    elif status_filter == "Upcoming":
        sessions_qs = sessions_qs.filter(start_time__gt=current_time)
    elif status_filter == "Expired":
        sessions_qs = sessions_qs.filter(end_time__lt=current_time)

    filtered_sessions = list(sessions_qs)
    filtered_count = len(filtered_sessions)

    paginator = Paginator(filtered_sessions, 6)
    try:
        page_obj = paginator.page(page_number)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages if paginator.num_pages else 1)
    sessions = list(page_obj.object_list)

    for s in sessions:
        annotate_session_creator(s, lecturer)
        s.formatted_start = format_lagos_time(s.start_time)
        s.formatted_end = format_lagos_time(s.end_time)
        s.start_time_iso = s.start_time.isoformat()
        s.end_time_iso = s.end_time.isoformat()
        s.date_label = s.start_time.strftime("%b %d, %Y")
        s.attendance_count = s.records.count()
        enrolled_profiles_for_course = course_profiles_map.get(s.course_id, [])
        s.enrolled_count = len(enrolled_profiles_for_course)
        s.absent_count = max(0, s.enrolled_count - s.attendance_count)
        s.attendance_pct = (
            round((s.attendance_count / s.enrolled_count * 100), 1)
            if s.enrolled_count > 0
            else 0
        )
        s.has_location = bool(s.latitude and s.longitude)

        if current_time < s.start_time:
            s.status_label = "Upcoming"
            s.status_color = "indigo"
        elif current_time > s.end_time:
            s.status_label = "Expired"
            s.status_color = "red"
        else:
            s.status_label = "Active"
            s.status_color = "green"

        present_records = list(
            s.records.select_related("student").all().order_by("marked_time")
        )
        present_user_ids = {r.student_id for r in present_records}
        profile_by_user_id = {
            sp.student_name_id: sp for sp in enrolled_profiles_for_course
        }
        s.present_list = []

        for r in present_records:
            sp = profile_by_user_id.get(r.student_id)
            matric = sp.matric_number if sp and sp.matric_number else "N/A"
            dept = sp.department if sp and sp.department else "N/A"
            s.present_list.append(
                {
                    "id": r.student.id,
                    "full_name": r.student.get_full_name() or r.student.username,
                    "username": r.student.username,
                    "email": r.student.email,
                    "matric": matric,
                    "department": dept,
                    "marked_time": format_lagos_time(r.marked_time),
                    "distance": round(r.distance, 1),
                    "face_similarity": round(r.face_similarity, 4)
                    if r.face_similarity is not None
                    else None,
                }
            )

        s.absent_list = []
        if s.status_label == "Expired" and s.enrolled_count > 0:
            for sp in enrolled_profiles_for_course:
                if sp.student_name_id not in present_user_ids:
                    s.absent_list.append(
                        {
                            "id": sp.student_name.id,
                            "full_name": sp.student_name.get_full_name()
                            or sp.student_name.username,
                            "username": sp.student_name.username,
                            "email": sp.student_name.email,
                            "matric": sp.matric_number or "N/A",
                            "department": sp.department or "N/A",
                        }
                    )

    avg_attendance_pct_total = 0
    avg_attendance_pct_count = 0
    for s in all_sessions:
        if current_time > s.end_time:
            enrolled_count = len(course_profiles_map.get(s.course_id, []))
            if enrolled_count > 0:
                avg_attendance_pct_total += s.records.count() / enrolled_count * 100
                avg_attendance_pct_count += 1
    avg_attendance_pct = (
        round(avg_attendance_pct_total / avg_attendance_pct_count, 1)
        if avg_attendance_pct_count
        else 0
    )

    courses_with_sessions = sorted(
        {s.course for s in all_sessions}, key=lambda c: c.course_code
    )
    selected_course_obj = next(
        (c for c in courses_with_sessions if str(c.id) == course_filter), None
    )

    active_filter_count = sum(1 for value in [search_q, course_filter] if value) + (
        0 if status_filter == "all" else 1
    )

    query_params = []
    if search_q:
        query_params.append(f"q={search_q}")
    if course_filter:
        query_params.append(f"course={course_filter}")
    if status_filter and status_filter != "all":
        query_params.append(f"status={status_filter}")
    querystring_without_page = "&".join(query_params)

    context = {
        "user_role": "lecturer",
        "sessions": sessions,
        "page_obj": page_obj,
        "paginator": paginator,
        "filtered_count": filtered_count,
        "total_sessions": len(all_sessions),
        "active_count": active_count,
        "upcoming_count": upcoming_count,
        "expired_count": expired_count,
        "total_checked_in": total_checked_in,
        "avg_attendance_pct": avg_attendance_pct,
        "courses_with_sessions": courses_with_sessions,
        "selected_course": course_filter,
        "selected_course_obj": selected_course_obj,
        "selected_status": status_filter,
        "search_q": search_q,
        "active_filter_count": active_filter_count,
        "querystring_without_page": querystring_without_page,
        "lecturer": lecturer,
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "lecturer_reports.html", context)


@lecturer_required
def millialms_lecturer_dashboard(request):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    assessments = (
        Assessment.objects.filter(lecturer=lecturer)
        .prefetch_related("courses", "questions", "attempts")
        .order_by("-created_at")
    )

    current_time = now()
    for assessment in assessments:
        assessment.status_label = get_assessment_status_label(assessment, current_time)
        assessment.type_label = get_assessment_type_label(assessment.assessment_type)
        assessment.course_count = assessment.courses.count()
        assessment.attempt_count = assessment.attempts.count()
        assessment.pending_manual_count = AssessmentResponse.objects.filter(
            attempt__assessment=assessment,
            question__question_type__in={
                AssessmentQuestion.TYPE_SHORT_ANSWER,
                AssessmentQuestion.TYPE_ESSAY,
            },
            graded_manually=False,
        ).count()

    context = {
        "user_role": "lecturer",
        "lecturer": lecturer,
        "assessments": assessments,
        "draft_count": sum(
            1 for a in assessments if a.status == Assessment.STATUS_DRAFT
        ),
        "published_count": sum(
            1 for a in assessments if a.status == Assessment.STATUS_PUBLISHED
        ),
        "attempt_count": sum(a.attempt_count for a in assessments),
        "pending_manual_count": sum(a.pending_manual_count for a in assessments),
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "millialms_lecturer_dashboard.html", context)


@lecturer_required
@transaction.atomic
def millialms_create_assessment(request):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    lecturer_courses = lecturer.courses_taught.all().order_by("course_code")

    if request.method == "POST":
        title = (request.POST.get("title") or "").strip()
        instructions = (request.POST.get("instructions") or "").strip()
        assessment_type = (
            request.POST.get("assessment_type") or Assessment.TYPE_QUIZ
        ).strip()
        action = (request.POST.get("action") or "draft").strip()
        selected_course_ids = request.POST.getlist("courses")
        start_time_raw = (request.POST.get("start_time") or "").strip()
        end_time_raw = (request.POST.get("end_time") or "").strip()
        time_limit_raw = (request.POST.get("time_limit_minutes") or "").strip()
        max_attempts_raw = (request.POST.get("max_attempts") or "1").strip()
        randomize_question_order = bool(request.POST.get("randomize_question_order"))
        randomize_answer_options = bool(request.POST.get("randomize_answer_options"))

        if not title:
            messages.error(request, "Assessment title is required.")
            return render(
                request,
                "millialms_create_assessment.html",
                {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
            )

        if assessment_type not in dict(Assessment.TYPE_CHOICES):
            messages.error(request, "Invalid assessment type selected.")
            return render(
                request,
                "millialms_create_assessment.html",
                {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
            )

        try:
            max_attempts = max(1, int(max_attempts_raw or "1"))
        except ValueError:
            messages.error(request, "Attempt limit must be a valid number.")
            return render(
                request,
                "millialms_create_assessment.html",
                {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
            )

        time_limit_minutes = None
        if time_limit_raw:
            try:
                time_limit_minutes = max(1, int(time_limit_raw))
            except ValueError:
                messages.error(request, "Time limit must be a valid number of minutes.")
                return render(
                    request,
                    "millialms_create_assessment.html",
                    {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
                )

        start_time = get_aware_datetime(start_time_raw) if start_time_raw else None
        end_time = get_aware_datetime(end_time_raw) if end_time_raw else None
        if start_time and end_time and start_time >= end_time:
            messages.error(request, "Start time must be before end time.")
            return render(
                request,
                "millialms_create_assessment.html",
                {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
            )

        assessment = Assessment.objects.create(
            title=title,
            instructions=instructions,
            assessment_type=assessment_type,
            status=Assessment.STATUS_PUBLISHED
            if action == "publish"
            else Assessment.STATUS_DRAFT,
            lecturer=lecturer,
            start_time=start_time,
            end_time=end_time,
            time_limit_minutes=time_limit_minutes,
            max_attempts=max_attempts,
            randomize_question_order=randomize_question_order,
            randomize_answer_options=randomize_answer_options,
            is_auto_publish=(action == "publish"),
        )
        if selected_course_ids:
            assessment.courses.set(lecturer_courses.filter(id__in=selected_course_ids))

        # --- DYNAMIC QUESTION PARSER ---
        # Accepts arbitrary number of questions + arbitrary MCQ options.
        # Frontend sends consecutive question indices (gaps allowed -> skip empty).
        question_ids = []
        for key in request.POST.keys():
            if key.startswith("q_idx_"):
                try:
                    question_ids.append(int(key.split("_", 2)[2]))
                except (ValueError, IndexError):
                    continue
        question_ids = sorted(set(question_ids))

        created_questions = 0
        for idx in question_ids:
            q_text = (request.POST.get(f"q_text_{idx}") or "").strip()
            q_type = (request.POST.get(f"q_type_{idx}") or "").strip()
            points_raw = (request.POST.get(f"q_points_{idx}") or "1").strip()
            if not q_text or q_type not in MILLIALMS_SUPPORTED_TYPES:
                continue
            try:
                points = float(points_raw or "1")
            except ValueError:
                points = 1
            if points < 0:
                points = 0

            manual_required = q_type in {
                AssessmentQuestion.TYPE_SHORT_ANSWER,
                AssessmentQuestion.TYPE_ESSAY,
            }
            accepted_answers = ""
            if q_type == AssessmentQuestion.TYPE_FILL_BLANK:
                accepted_answers = (
                    request.POST.get(f"q_answers_{idx}") or ""
                ).strip()

            question = AssessmentQuestion.objects.create(
                assessment=assessment,
                question_text=q_text,
                question_type=q_type,
                points=points,
                order=created_questions + 1,
                accepted_answers=accepted_answers,
                manual_grading_required=manual_required,
            )
            created_questions += 1

            if q_type == AssessmentQuestion.TYPE_TRUE_FALSE:
                correct_answer = (
                    (request.POST.get(f"q_tf_correct_{idx}") or "true").strip().lower()
                )
                AssessmentOption.objects.create(
                    question=question,
                    option_text="True",
                    is_correct=(correct_answer == "true"),
                    order=1,
                )
                AssessmentOption.objects.create(
                    question=question,
                    option_text="False",
                    is_correct=(correct_answer == "false"),
                    order=2,
                )
            elif q_type == AssessmentQuestion.TYPE_MCQ:
                # Arbitrary number of MCQ options via consecutive q_opt_N_IDX keys.
                option_keys = []
                for key in request.POST.keys():
                    prefix = f"q_opt_{idx}_"
                    if key.startswith(prefix):
                        tail = key[len(prefix):]
                        # Accepted keys: q_opt_<qidx>_<oidx> (text), q_opt_<qidx>_<oidx>_correct (hidden flag, 1 or 0)
                        if "_correct" not in tail:
                            try:
                                option_keys.append(int(tail))
                            except ValueError:
                                continue
                option_keys = sorted(set(option_keys))
                any_correct_selected = False
                option_objs = []
                oidx_actual = 1
                for oidx in option_keys:
                    opt_text = (
                        request.POST.get(f"q_opt_{idx}_{oidx}") or ""
                    ).strip()
                    if not opt_text:
                        continue
                    is_correct_raw = (
                        request.POST.get(f"q_opt_{idx}_{oidx}_correct") or ""
                    ).strip().lower()
                    is_correct = is_correct_raw in {"1", "on", "yes", "true"}
                    if is_correct:
                        any_correct_selected = True
                    option_objs.append(
                        AssessmentOption(
                            question=question,
                            option_text=opt_text,
                            is_correct=is_correct,
                            order=oidx_actual,
                        )
                    )
                    oidx_actual += 1
                if option_objs:
                    if not any_correct_selected:
                        option_objs[0].is_correct = True
                    AssessmentOption.objects.bulk_create(option_objs)

        if created_questions == 0:
            assessment.delete()
            messages.error(
                request,
                "Add at least one valid question before saving this assessment.",
            )
            return render(
                request,
                "millialms_create_assessment.html",
                {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
            )

        messages.success(
            request, f'MilliaLMS assessment "{assessment.title}" saved successfully.'
        )
        return redirect("millialms_lecturer_assessment_detail", pk=assessment.id)

    return render(
        request,
        "millialms_create_assessment.html",
        {"user_role": "lecturer", "lecturer_courses": lecturer_courses, "edit_mode": False},
    )


# ------- NEW: EDIT Assessment (dynamic question rebuild) -------
@lecturer_required
@transaction.atomic
def millialms_edit_assessment(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    assessment = get_object_or_404(
        Assessment.objects.prefetch_related("courses", "questions__options"),
        id=pk,
        lecturer=lecturer,
    )
    lecturer_courses = lecturer.courses_taught.all().order_by("course_code")

    if request.method == "POST":
        # --- Same parsing as create, but DELETE old questions/options first ---
        title = (request.POST.get("title") or "").strip()
        instructions = (request.POST.get("instructions") or "").strip()
        assessment_type = (
            request.POST.get("assessment_type") or Assessment.TYPE_QUIZ
        ).strip()
        action = (request.POST.get("action") or "draft").strip()
        selected_course_ids = request.POST.getlist("courses")
        start_time_raw = (request.POST.get("start_time") or "").strip()
        end_time_raw = (request.POST.get("end_time") or "").strip()
        time_limit_raw = (request.POST.get("time_limit_minutes") or "").strip()
        max_attempts_raw = (request.POST.get("max_attempts") or "1").strip()
        randomize_question_order = bool(request.POST.get("randomize_question_order"))
        randomize_answer_options = bool(request.POST.get("randomize_answer_options"))

        if not title:
            messages.error(request, "Assessment title is required.")
            return redirect("millialms_edit_assessment", pk=assessment.id)
        if assessment_type not in dict(Assessment.TYPE_CHOICES):
            messages.error(request, "Invalid assessment type selected.")
            return redirect("millialms_edit_assessment", pk=assessment.id)
        try:
            max_attempts = max(1, int(max_attempts_raw or "1"))
        except ValueError:
            messages.error(request, "Attempt limit must be a valid number.")
            return redirect("millialms_edit_assessment", pk=assessment.id)

        time_limit_minutes = None
        if time_limit_raw:
            try:
                time_limit_minutes = max(1, int(time_limit_raw))
            except ValueError:
                messages.error(request, "Time limit must be a valid number of minutes.")
                return redirect("millialms_edit_assessment", pk=assessment.id)

        start_time = get_aware_datetime(start_time_raw) if start_time_raw else None
        end_time = get_aware_datetime(end_time_raw) if end_time_raw else None
        if start_time and end_time and start_time >= end_time:
            messages.error(request, "Start time must be before end time.")
            return redirect("millialms_edit_assessment", pk=assessment.id)

        assessment.title = title
        assessment.instructions = instructions
        assessment.assessment_type = assessment_type
        assessment.status = (
            Assessment.STATUS_PUBLISHED if action == "publish" else Assessment.STATUS_DRAFT
        )
        assessment.start_time = start_time
        assessment.end_time = end_time
        assessment.time_limit_minutes = time_limit_minutes
        assessment.max_attempts = max_attempts
        assessment.randomize_question_order = randomize_question_order
        assessment.randomize_answer_options = randomize_answer_options
        assessment.is_auto_publish = (action == "publish")
        assessment.save()

        if selected_course_ids:
            assessment.courses.set(lecturer_courses.filter(id__in=selected_course_ids))
        else:
            assessment.courses.clear()

        # Drop old questions/options — they're re-created from POST.
        # (Note: Historical attempts still reference the QUESTIONS they were
        # created with. To preserve them fully, this would need a more complex
        # "update vs replace" diff. For MVP, we keep it simple: editing only
        # allowed BEFORE any attempt has been made.)
        has_attempts = assessment.attempts.exists()
        if not has_attempts:
            AssessmentOption.objects.filter(question__assessment=assessment).delete()
            assessment.questions.all().delete()
        else:
            messages.warning(
                request,
                "Questions were NOT replaced because attempts already exist for this assessment. If you need to change questions, please duplicate the assessment.",
            )
            return redirect("millialms_lecturer_assessment_detail", pk=assessment.id)

        # --- Parse dynamic questions identically to create_assessment ---
        question_ids = []
        for key in request.POST.keys():
            if key.startswith("q_idx_"):
                try:
                    question_ids.append(int(key.split("_", 2)[2]))
                except (ValueError, IndexError):
                    continue
        question_ids = sorted(set(question_ids))

        created_questions = 0
        for idx in question_ids:
            q_text = (request.POST.get(f"q_text_{idx}") or "").strip()
            q_type = (request.POST.get(f"q_type_{idx}") or "").strip()
            points_raw = (request.POST.get(f"q_points_{idx}") or "1").strip()
            if not q_text or q_type not in MILLIALMS_SUPPORTED_TYPES:
                continue
            try:
                points = float(points_raw or "1")
            except ValueError:
                points = 1
            if points < 0:
                points = 0

            manual_required = q_type in {
                AssessmentQuestion.TYPE_SHORT_ANSWER,
                AssessmentQuestion.TYPE_ESSAY,
            }
            accepted_answers = ""
            if q_type == AssessmentQuestion.TYPE_FILL_BLANK:
                accepted_answers = (request.POST.get(f"q_answers_{idx}") or "").strip()

            question = AssessmentQuestion.objects.create(
                assessment=assessment,
                question_text=q_text,
                question_type=q_type,
                points=points,
                order=created_questions + 1,
                accepted_answers=accepted_answers,
                manual_grading_required=manual_required,
            )
            created_questions += 1

            if q_type == AssessmentQuestion.TYPE_TRUE_FALSE:
                correct_answer = (
                    (request.POST.get(f"q_tf_correct_{idx}") or "true").strip().lower()
                )
                AssessmentOption.objects.bulk_create([
                    AssessmentOption(question=question, option_text="True",
                                     is_correct=(correct_answer == "true"), order=1),
                    AssessmentOption(question=question, option_text="False",
                                     is_correct=(correct_answer == "false"), order=2),
                ])
            elif q_type == AssessmentQuestion.TYPE_MCQ:
                option_keys = []
                for key in request.POST.keys():
                    prefix = f"q_opt_{idx}_"
                    if key.startswith(prefix) and "_correct" not in key[len(prefix):]:
                        tail = key[len(prefix):]
                        try:
                            option_keys.append(int(tail))
                        except ValueError:
                            continue
                option_keys = sorted(set(option_keys))
                any_correct_selected = False
                option_objs = []
                oidx_actual = 1
                for oidx in option_keys:
                    opt_text = (request.POST.get(f"q_opt_{idx}_{oidx}") or "").strip()
                    if not opt_text:
                        continue
                    is_correct_raw = (
                        request.POST.get(f"q_opt_{idx}_{oidx}_correct") or ""
                    ).strip().lower()
                    is_correct = is_correct_raw in {"1", "on", "yes", "true"}
                    if is_correct:
                        any_correct_selected = True
                    option_objs.append(AssessmentOption(
                        question=question,
                        option_text=opt_text,
                        is_correct=is_correct,
                        order=oidx_actual,
                    ))
                    oidx_actual += 1
                if option_objs:
                    if not any_correct_selected:
                        option_objs[0].is_correct = True
                    AssessmentOption.objects.bulk_create(option_objs)

        if created_questions == 0:
            # Recreate with a default question so the assessment isn't empty.
            AssessmentQuestion.objects.create(
                assessment=assessment, question_text=assessment.title,
                question_type=AssessmentQuestion.TYPE_MCQ, points=1, order=1,
                manual_grading_required=False,
            )
            messages.warning(
                request, "Saved but no valid questions were detected.",
            )

        messages.success(request, f'Updated "{assessment.title}" successfully.')
        return redirect("millialms_lecturer_assessment_detail", pk=assessment.id)

    # GET: serialize existing questions for the JS editor
    q_serialized = []
    for q in assessment.questions.all():
        payload = {
            "index": q.order,
            "type": q.question_type,
            "text": q.question_text,
            "points": str(q.points),
            "accepted_answers": q.accepted_answers or "",
        }
        if q.question_type == AssessmentQuestion.TYPE_TRUE_FALSE:
            correct_opt = q.options.filter(is_correct=True).first()
            payload["tf_correct"] = (
                correct_opt.option_text.lower() if correct_opt else "true"
            )
        elif q.question_type == AssessmentQuestion.TYPE_MCQ:
            opts = []
            for idx, o in enumerate(q.options.all(), start=1):
                opts.append({
                    "slot": idx,
                    "text": o.option_text,
                    "correct": "1" if o.is_correct else "0",
                })
            payload["options"] = opts
        q_serialized.append(payload)

    return render(
        request,
        "millialms_create_assessment.html",
        {
            "user_role": "lecturer",
            "lecturer_courses": lecturer_courses,
            "edit_mode": True,
            "assessment": assessment,
            "existing_questions_json": json.dumps(q_serialized),
            "selected_course_ids": [c.id for c in assessment.courses.all()],
        },
    )


# ------- NEW: DUPLICATE Assessment (clone questions + options + settings) -------
@lecturer_required
@transaction.atomic
def millialms_duplicate_assessment(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    original = get_object_or_404(
        Assessment.objects.prefetch_related("courses", "questions__options"),
        id=pk,
        lecturer=lecturer,
    )
    course_ids = list(original.courses.values_list("id", flat=True))
    orig_questions = list(original.questions.prefetch_related("options").all())

    dup = Assessment(
        title=f"{original.title} (Copy)",
        instructions=original.instructions,
        assessment_type=original.assessment_type,
        status=Assessment.STATUS_DRAFT,
        lecturer=lecturer,
        start_time=original.start_time,
        end_time=original.end_time,
        time_limit_minutes=original.time_limit_minutes,
        max_attempts=original.max_attempts,
        randomize_question_order=original.randomize_question_order,
        randomize_answer_options=original.randomize_answer_options,
        is_auto_publish=False,
    )
    dup.save()
    if course_ids:
        dup.courses.set(Course.objects.filter(id__in=course_ids))

    for q in orig_questions:
        opts = list(q.options.all())
        q.pk = None
        q.id = None
        q.assessment = dup
        q.save()
        for o in opts:
            o.pk = None
            o.id = None
            o.question = q
        if opts:
            AssessmentOption.objects.bulk_create(opts)

    messages.success(request, f'Created a copy of "{dup.title}". Edit what you need.')
    return redirect("millialms_edit_assessment", pk=dup.id)


# ------- NEW: DELETE Assessment (cascades to questions/attempts/responses) -------
@lecturer_required
@transaction.atomic
def millialms_delete_assessment(request, pk):
    if request.method != "POST":
        messages.error(request, "Please confirm deletion from the assessment detail page.")
        return redirect("millialms_lecturer_dashboard")
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    assessment = get_object_or_404(Assessment, id=pk, lecturer=lecturer)
    title = assessment.title
    assessment.delete()
    messages.success(request, f'Deleted "{title}" permanently.')
    return redirect("millialms_lecturer_dashboard")


@lecturer_required
def millialms_lecturer_assessment_detail(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    assessment = get_object_or_404(
        Assessment.objects.prefetch_related(
            "courses", "questions__options", "attempts__student"
        ),
        id=pk,
        lecturer=lecturer,
    )
    current_time = now()
    assessment.status_label = get_assessment_status_label(assessment, current_time)
    assessment.type_label = get_assessment_type_label(assessment.assessment_type)

    # Force-evaluate to a list so we can annotate each object with
    # violation_log_json without the queryset being re-evaluated from DB.
    # Exclude bare in-progress attempts (anti-cheat pre-creates with no answers)
    # — only show submitted/graded attempts in the lecturer table.
    attempts = list(
        assessment.attempts
        .select_related("student")
        .exclude(status=AssessmentAttempt.STATUS_IN_PROGRESS)
        .order_by("-started_at")
    )
    # Include BOTH short answer AND essay responses in pending review.
    pending_manual = AssessmentResponse.objects.filter(
        attempt__assessment=assessment,
        question__question_type__in={
            AssessmentQuestion.TYPE_SHORT_ANSWER,
            AssessmentQuestion.TYPE_ESSAY,
        },
        graded_manually=False,
    ).select_related("attempt", "attempt__student", "question")

    # Aggregate stats for overview cards
    total_possible = sum(
        float(q.points) for q in assessment.questions.all()
    )
    graded_attempts = [a for a in attempts if a.status == AssessmentAttempt.STATUS_GRADED]
    if graded_attempts:
        avg_score = round(
            sum(float(a.total_score) for a in graded_attempts) / len(graded_attempts), 2
        )
        if total_possible > 0:
            avg_pct = round((avg_score / total_possible) * 100, 1)
        else:
            avg_pct = 0
        highest = max(float(a.total_score) for a in graded_attempts)
        lowest = min(float(a.total_score) for a in graded_attempts)
    else:
        avg_score = 0
        avg_pct = 0
        highest = 0
        lowest = 0
    submission_rate = 0
    enrolled = StudentProfile.objects.filter(
        courses_enrolled__in=assessment.courses.all()
    ).distinct().count()
    if enrolled:
        unique_subs = {a.student_id for a in attempts}
        submission_rate = round(len(unique_subs) / enrolled * 100, 1)

    # Per-question breakdown
    per_question = []
    for q in assessment.questions.all():
        q_responses = q.responses.filter(attempt__assessment=assessment).all()
        q_total = len(q_responses)
        q_correct = sum(
            1 for r in q_responses
            if r.is_correct and r.question.question_type in {
                AssessmentQuestion.TYPE_MCQ,
                AssessmentQuestion.TYPE_TRUE_FALSE,
                AssessmentQuestion.TYPE_FILL_BLANK,
            }
        )
        q_avg_score = 0
        if q_total:
            scored = sum(
                (float(r.auto_score) + float(r.manual_score or 0))
                for r in q_responses
            )
            q_avg_score = round(scored / q_total, 2)
        per_question.append({
            "question": q,
            "answered_count": q_total,
            "correct_count": q_correct,
            "avg_score": q_avg_score,
            "correct_pct": round(q_correct / q_total * 100, 1) if q_total else 0,
        })

    # Pre-serialize violation_log on each attempt so the template can embed
    # it as safe JSON (JSONField stores a Python list; we need a JSON string).
    for a in attempts:
        raw_log = a.violation_log if isinstance(a.violation_log, list) else []
        a.violation_log_json = json.dumps(raw_log)

    context = {
        "user_role": "lecturer",
        "assessment": assessment,
        "attempts": attempts,
        "flagged_count": sum(1 for a in attempts if a.flagged),
        "pending_manual": pending_manual,
        "pending_manual_count": pending_manual.count(),
        "server_now_lagos": format_lagos_time(current_time),
        "essay_reminder": "Essay and short-answer submissions appear as Pending until you grade them manually below.",
        "stats": {
            "enrolled": enrolled,
            "submission_count": len({a.student_id for a in attempts}),
            "submission_rate": submission_rate,
            "avg_score": avg_score,
            "avg_pct": avg_pct,
            "highest": highest,
            "lowest": lowest,
            "total_possible": total_possible,
            "attempt_count": len(attempts),
            "graded_count": len(graded_attempts),
            "pending_grading_count": sum(1 for a in attempts if a.status == AssessmentAttempt.STATUS_SUBMITTED),
        },
        "per_question": per_question,
    }
    return render(request, "millialms_lecturer_assessment_detail.html", context)


@lecturer_required
@transaction.atomic
def millialms_toggle_assessment_publish(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    assessment = get_object_or_404(Assessment, id=pk, lecturer=lecturer)
    if assessment.status == Assessment.STATUS_PUBLISHED:
        assessment.status = Assessment.STATUS_UNPUBLISHED
        messages.success(request, "Assessment unpublished successfully.")
    else:
        assessment.status = Assessment.STATUS_PUBLISHED
        messages.success(request, "Assessment published successfully.")
    assessment.save(update_fields=["status", "updated_at"])
    return redirect("millialms_lecturer_assessment_detail", pk=assessment.id)


@lecturer_required
@transaction.atomic
def millialms_grade_attempt(request, attempt_id):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    attempt = get_object_or_404(
        AssessmentAttempt.objects.select_related("assessment"),
        id=attempt_id,
        assessment__lecturer=lecturer,
    )

    if request.method == "POST":
        for response in attempt.responses.select_related("question"):
            if response.question.question_type in {
                AssessmentQuestion.TYPE_SHORT_ANSWER,
                AssessmentQuestion.TYPE_ESSAY,
            }:
                score_raw = (request.POST.get(f"score_{response.id}") or "0").strip()
                feedback = (request.POST.get(f"feedback_{response.id}") or "").strip()
                try:
                    score = float(score_raw or "0")
                except ValueError:
                    score = 0
                response.manual_score = max(
                    0, min(score, float(response.question.points))
                )
                response.graded_manually = True
                response.lecturer_feedback = feedback
                response.save()
        grade_attempt(attempt)
        messages.success(request, "Attempt graded successfully.")

    return redirect("millialms_lecturer_assessment_detail", pk=attempt.assessment.id)


@lecturer_required
def millialms_export_results(request, pk):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    assessment = get_object_or_404(Assessment, id=pk, lecturer=lecturer)

    export_format = (request.GET.get("format") or "csv").lower()
    attempts = assessment.attempts.select_related("student").order_by("-started_at")

    if export_format == "pdf":
        lines = [
            f"MilliaLMS Results - {assessment.title}",
            f"Type: {get_assessment_type_label(assessment.assessment_type)}",
            "",
        ]
        for attempt in attempts:
            lines.append(
                f"{attempt.student.get_full_name() or attempt.student.username} | Attempt {attempt.attempt_number} | Score {attempt.total_score}/{attempt.total_possible_score} | Status {attempt.status}"
            )
        response = HttpResponse("\n".join(lines), content_type="application/pdf")
        response["Content-Disposition"] = (
            f'attachment; filename="millialms_{assessment.id}_results.pdf"'
        )
        return response

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Student",
            "Email",
            "Attempt",
            "Status",
            "Started At",
            "Submitted At",
            "Auto Score",
            "Manual Score",
            "Total Score",
            "Possible Score",
        ]
    )
    for attempt in attempts:
        writer.writerow(
            [
                attempt.student.get_full_name() or attempt.student.username,
                attempt.student.email,
                attempt.attempt_number,
                attempt.status,
                format_lagos_time(attempt.started_at),
                format_lagos_time(attempt.submitted_at) if attempt.submitted_at else "",
                attempt.auto_score,
                attempt.manual_score,
                attempt.total_score,
                attempt.total_possible_score,
            ]
        )
    response = HttpResponse(output.getvalue(), content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="millialms_{assessment.id}_results.csv"'
    )
    return response


@student_required
def millialms_student_dashboard(request):
    user_obj = User.objects.get(username=request.user)
    student_profile = get_object_or_404(StudentProfile, student_name=user_obj)
    enrolled_courses = student_profile.courses_enrolled.all()
    current_time = now()

    assessments = (
        Assessment.objects.filter(courses__in=enrolled_courses)
        .select_related("lecturer", "lecturer__user")
        .prefetch_related("courses", "questions")
        .distinct()
        .order_by("-created_at")
    )

    visible_assessments = []
    for assessment in assessments:
        assessment.status_label = get_assessment_status_label(assessment, current_time)
        assessment.type_label = get_assessment_type_label(assessment.assessment_type)
        # Count only completed attempts — in-progress records are anti-cheat
        # pre-creates and must not reduce the student's remaining attempts.
        assessment.my_attempts = assessment.attempts.filter(
            student=user_obj,
            status__in=[AssessmentAttempt.STATUS_SUBMITTED, AssessmentAttempt.STATUS_GRADED],
        ).count()
        assessment.remaining_attempts = max(
            assessment.max_attempts - assessment.my_attempts, 0
        )
        assessment.can_take = (
            assessment.status == Assessment.STATUS_PUBLISHED
            and assessment.status_label == "Live"
            and assessment.remaining_attempts > 0
        )
        visible_assessments.append(assessment)

    context = {
        "user_role": "student",
        "assessments": visible_assessments,
        "live_count": sum(1 for a in visible_assessments if a.status_label == "Live"),
        "upcoming_count": sum(
            1 for a in visible_assessments if a.status_label == "Upcoming"
        ),
        "submitted_count": AssessmentAttempt.objects.filter(student=user_obj).count(),
        "server_now_lagos": format_lagos_time(current_time),
        "essay_reminder": "Essay and short-answer questions are graded manually by your lecturer. Your final score updates once they review.",
    }
    return render(request, "millialms_student_dashboard.html", context)


@student_required
def millialms_student_assessment_detail(request, pk):
    user_obj = User.objects.get(username=request.user)
    student_profile = get_object_or_404(StudentProfile, student_name=user_obj)
    assessment = get_object_or_404(
        Assessment.objects.select_related(
            "lecturer", "lecturer__user"
        ).prefetch_related("courses", "questions__options"),
        id=pk,
        courses__in=student_profile.courses_enrolled.all(),
    )
    current_time = now()
    assessment.status_label = get_assessment_status_label(assessment, current_time)
    assessment.type_label = get_assessment_type_label(assessment.assessment_type)
    attempts = assessment.attempts.filter(
        student=user_obj,
        status__in=[AssessmentAttempt.STATUS_SUBMITTED, AssessmentAttempt.STATUS_GRADED],
    ).order_by("-attempt_number")
    # Count only completed attempts against max_attempts — in-progress attempts
    # are pre-created records for the anti-cheat system and must not count.
    completed_count = attempts.count()
    can_take = (
        assessment.status == Assessment.STATUS_PUBLISHED
        and assessment.status_label == "Live"
        and completed_count < assessment.max_attempts
    )

    context = {
        "user_role": "student",
        "assessment": assessment,
        "attempts": attempts,
        "can_take": can_take,
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "millialms_student_assessment_detail.html", context)


@student_required
def millialms_take_assessment(request, pk):
    user_obj = User.objects.get(username=request.user)
    student_profile = get_object_or_404(StudentProfile, student_name=user_obj)
    assessment = get_object_or_404(
        Assessment.objects.prefetch_related("questions__options"),
        id=pk,
        courses__in=student_profile.courses_enrolled.all(),
    )
    current_time = now()
    status_label = get_assessment_status_label(assessment, current_time)

    if assessment.status != Assessment.STATUS_PUBLISHED or status_label != "Live":
        messages.error(request, "This assessment is not currently available to take.")
        return redirect("millialms_student_assessment_detail", pk=assessment.id)

    # Count only completed (submitted / graded) attempts against the limit.
    # In-progress attempts are orphans from previous page loads and must not
    # count, otherwise a page refresh would lock the student out.
    completed_attempts = assessment.attempts.filter(
        student=user_obj,
        status__in=[AssessmentAttempt.STATUS_SUBMITTED, AssessmentAttempt.STATUS_GRADED],
    ).count()

    if completed_attempts >= assessment.max_attempts:
        messages.error(
            request,
            "You have reached the maximum number of attempts allowed for this assessment.",
        )
        return redirect("millialms_student_assessment_detail", pk=assessment.id)

    questions = list(assessment.questions.prefetch_related("options").all())
    if assessment.randomize_question_order:
        import random
        random.shuffle(questions)

    if request.method == "POST":
        # Retrieve the in-progress attempt created on GET, or create a new one
        # (fallback for JS-disabled browsers or direct POST).
        ac_attempt_id = (request.POST.get("ac_attempt_id") or "").strip()
        attempt = None
        if ac_attempt_id:
            try:
                attempt = AssessmentAttempt.objects.get(
                    id=int(ac_attempt_id),
                    assessment=assessment,
                    student=user_obj,
                    status=AssessmentAttempt.STATUS_IN_PROGRESS,
                )
            except (AssessmentAttempt.DoesNotExist, ValueError):
                attempt = None

        with transaction.atomic():
            if attempt is None:
                attempt = AssessmentAttempt.objects.create(
                    assessment=assessment,
                    student=user_obj,
                    attempt_number=completed_attempts + 1,
                    status=AssessmentAttempt.STATUS_IN_PROGRESS,
                )

            for question in questions:
                response, _ = AssessmentResponse.objects.get_or_create(
                    attempt=attempt, question=question
                )
                if question.question_type in {
                    AssessmentQuestion.TYPE_MCQ,
                    AssessmentQuestion.TYPE_TRUE_FALSE,
                }:
                    option_id = (request.POST.get(f"question_{question.id}") or "").strip()
                    if option_id:
                        try:
                            response.selected_option = question.options.get(id=option_id)
                        except AssessmentOption.DoesNotExist:
                            response.selected_option = None
                    else:
                        response.selected_option = None
                else:
                    response.text_answer = (
                        request.POST.get(f"question_{question.id}") or ""
                    ).strip()
                response.save()

            attempt.submitted_at = now()
            grade_attempt(attempt)
            attempt.save(update_fields=["submitted_at"])

        messages.success(request, "Your assessment has been submitted successfully.")
        return redirect("millialms_student_result", attempt_id=attempt.id)

    # GET — reuse any existing in-progress attempt for this student + assessment
    # (handles page refresh without creating orphaned records), or create a
    # fresh one so the anti-cheat JS can report violations in real time.
    attempt_obj = AssessmentAttempt.objects.filter(
        assessment=assessment,
        student=user_obj,
        status=AssessmentAttempt.STATUS_IN_PROGRESS,
    ).first()

    if attempt_obj is None:
        attempt_obj = AssessmentAttempt.objects.create(
            assessment=assessment,
            student=user_obj,
            attempt_number=completed_attempts + 1,
            status=AssessmentAttempt.STATUS_IN_PROGRESS,
        )

    serialized_questions = [
        serialize_question_for_attempt(q, assessment.randomize_answer_options)
        for q in questions
    ]
    context = {
        "user_role": "student",
        "assessment": assessment,
        "questions": serialized_questions,
        "attempt_id": attempt_obj.id,
        "server_now_lagos": format_lagos_time(current_time),
        "essay_reminder": "Essay and short-answer responses will be reviewed and graded by your lecturer after submission.",
    }
    return render(request, "millialms_take_assessment.html", context)


@student_required
def millialms_student_result(request, attempt_id):
    user_obj = User.objects.get(username=request.user)
    attempt = get_object_or_404(
        AssessmentAttempt.objects.select_related("assessment").prefetch_related(
            "responses__question", "responses__selected_option"
        ),
        id=attempt_id,
        student=user_obj,
    )
    context = {
        "user_role": "student",
        "attempt": attempt,
        "responses": attempt.responses.all(),
        "server_now_lagos": format_lagos_time(now()),
    }
    return render(request, "millialms_student_result.html", context)


# ========== ANTI-CHEAT VIOLATION REPORTING ==========

# Maximum violations before the attempt is auto-flagged as suspicious.
VIOLATION_FLAG_THRESHOLD = 3

# Valid violation types for server-side validation
VALID_VIOLATION_TYPES = {
    'tab_switch', 'blur', 'copy', 'cut', 'paste', 'devtools',
    'fullscreen_exit', 'right_click', 'print_screen', 'print_attempt',
    'fingerprint-change', 'function-tampered', 'dom-tampered',
    'debugger-detected', 'console-tampered', 'variable-tampered'
}


@csrf_exempt
@require_POST
@student_required
def millialms_report_violation(request, attempt_id):
    """
    AJAX endpoint called by the anti-cheat JS to log a single violation event.

    Expected JSON body:
        { "type": "tab_switch" | "copy" | "paste" | "devtools" | "fullscreen_exit"
                  | "right_click" | "context_menu" | "print_screen" | "blur",
          "detail": "<optional extra info string>" }

    Returns JSON:
        { "ok": true, "violation_count": N, "flagged": bool, "auto_submit": bool }

    auto_submit is True when the threshold is exceeded for the first time,
    telling the JS to force-submit the form immediately.
    """
    user_obj = User.objects.get(username=request.user)

    # Rate limiting check (prevent abuse)
    rate_limit_key = f"{user_obj.id}_{attempt_id}"
    if not _check_rate_limit(rate_limit_key):
        logger.warning("Rate limit exceeded for violation reporting: user=%s attempt=%s", 
                      user_obj.username, attempt_id)
        return JsonResponse({"ok": False, "error": "Rate limit exceeded."}, status=429)

    # Only allow reporting on an in-progress attempt that belongs to this student.
    attempt = get_object_or_404(
        AssessmentAttempt,
        id=attempt_id,
        student=user_obj,
        status=AssessmentAttempt.STATUS_IN_PROGRESS,
    )

    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)

    violation_type = str(payload.get("type", "unknown"))[:64]
    detail = str(payload.get("detail", ""))[:256]

    # Server-side validation of violation type
    if violation_type not in VALID_VIOLATION_TYPES:
        logger.warning("Invalid violation type received: user=%s type=%s", 
                      user_obj.username, violation_type)
        return JsonResponse({"ok": False, "error": "Invalid violation type."}, status=400)

    # Build the new log entry
    entry = {
        "type": violation_type,
        "detail": detail,
        "ts": now().isoformat(),
    }

    # Atomically update the attempt so concurrent requests don't lose events.
    with transaction.atomic():
        attempt = AssessmentAttempt.objects.select_for_update().get(id=attempt_id)

        log = attempt.violation_log if isinstance(attempt.violation_log, list) else []
        log.append(entry)
        attempt.violation_log = log
        attempt.violation_count = len(log)

        was_flagged_before = attempt.flagged
        if attempt.violation_count >= VIOLATION_FLAG_THRESHOLD:
            attempt.flagged = True

        attempt.save(update_fields=["violation_count", "violation_log", "flagged"])

    # Signal auto-submit only on the exact crossing of the threshold
    # (not on every subsequent violation) to avoid double-submitting.
    auto_submit = (
        attempt.flagged and not was_flagged_before
        and attempt.violation_count >= VIOLATION_FLAG_THRESHOLD
    )

    # Enhanced logging with more context
    logger.warning(
        "Anti-cheat violation: attempt=%d student=%s assessment=%d type=%s count=%d flagged=%s detail=%s",
        attempt.id,
        user_obj.username,
        attempt.assessment.id,
        violation_type,
        attempt.violation_count,
        attempt.flagged,
        detail,
    )
    
    # Additional logging for auto-submit events
    if auto_submit:
        logger.error(
            "AUTO-SUBMIT TRIGGERED: attempt=%d student=%s assessment=%d violations=%d",
            attempt.id,
            user_obj.username,
            attempt.assessment.id,
            attempt.violation_count,
        )

    return JsonResponse({
        "ok": True,
        "violation_count": attempt.violation_count,
        "flagged": attempt.flagged,
        "auto_submit": auto_submit,
    })


# ========== LECTURER COURSE ROSTER ==========
@lecturer_required
def lecturer_course_roster(request):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    current_time = now()

    courses = list(lecturer.courses_taught.all())
    courses.sort(key=lambda c: c.course_code)

    selected_course_id_raw = request.GET.get("course") or (
        str(courses[0].id) if courses else ""
    )
    search_q = (request.GET.get("q") or "").strip()
    try:
        selected_course_id = int(selected_course_id_raw)
    except (ValueError, TypeError):
        selected_course_id = courses[0].id if courses else None

    selected_course = None
    students = []
    course_sessions_all = []
    course_expired_sessions = 0
    total_present_all = 0

    for c in courses:
        if c.id == selected_course_id:
            selected_course = c
            break

    if selected_course:
        # All attendance sessions for THIS course, regardless of which
        # co-lecturer created them (a course may be taught by several lecturers).
        course_sessions = list(
            AttendanceSession.objects.filter(course=selected_course)
            .select_related("lecturer", "lecturer__user")
            .order_by("-start_time")
        )
        for s in course_sessions:
            annotate_session_creator(s, lecturer)
            # Display fields for the sessions panel (times shown in Lagos TZ)
            start_local = localtime(s.start_time, timezone=get_default_timezone())
            end_local = localtime(s.end_time, timezone=get_default_timezone())
            s.date_label = start_local.strftime("%b %d, %Y")
            s.time_window = (
                f"{start_local.strftime('%H:%M')} – {end_local.strftime('%H:%M')}"
            )
            s.attendance_count = s.records.count()
            if s.end_time < current_time:
                s.status_label = "Held"
                s.status_color = "gray"
            elif s.start_time > current_time:
                s.status_label = "Upcoming"
                s.status_color = "indigo"
            else:
                s.status_label = "Active"
                s.status_color = "emerald"
        course_sessions_all = course_sessions
        course_expired_sessions = sum(
            1 for s in course_sessions if s.end_time < current_time
        )

        enrolled_students_qs = (
            StudentProfile.objects.filter(courses_enrolled=selected_course)
            .select_related("student_name")
            .order_by("student_name__first_name", "student_name__last_name")
        )

        if search_q:
            enrolled_students_qs = enrolled_students_qs.filter(
                Q(student_name__first_name__icontains=search_q)
                | Q(student_name__last_name__icontains=search_q)
                | Q(student_name__username__icontains=search_q)
                | Q(student_name__email__icontains=search_q)
                | Q(matric_number__icontains=search_q)
                | Q(department__icontains=search_q)
            )

        enrolled_students = list(enrolled_students_qs)

        # Pre-fetch all records per session for this course once
        all_records_qs = AttendanceRecord.objects.filter(
            session__in=course_sessions
        ).select_related("student")
        records_by_student = {}
        for r in all_records_qs:
            records_by_student.setdefault(r.student_id, []).append(r)

        # Build per-student analytics
        for sp in enrolled_students:
            sid = sp.student_name_id
            recs = records_by_student.get(sid, [])
            present_count = len(recs)
            absent_count = max(0, course_expired_sessions - present_count)
            attendance_pct = (
                round((present_count / course_expired_sessions * 100), 1)
                if course_expired_sessions > 0
                else 0
            )
            total_present_all += present_count

            # Avg distance/face similarity
            if recs:
                avg_dist = round(sum(r.distance for r in recs) / len(recs), 1)
                sims = [
                    r.face_similarity for r in recs if r.face_similarity is not None
                ]
                avg_sim = round(sum(sims) / len(sims), 4) if sims else None
            else:
                avg_dist = None
                avg_sim = None

            # Last seen
            last_marked_time = max((r.marked_time for r in recs), default=None)
            if last_marked_time:
                last_seen = format_lagos_time(last_marked_time)
            else:
                last_seen = "Never"

            # Attendance rating / risk flag
            if course_expired_sessions == 0:
                risk_level = "new"
                risk_label = "No sessions held yet"
            elif attendance_pct >= 80:
                risk_level = "good"
                risk_label = "On Track 🌟"
            elif attendance_pct >= 60:
                risk_level = "warn"
                risk_label = "At Risk ⚠️"
            else:
                risk_level = "danger"
                risk_label = "Failing Attendance 🚨"

            students.append(
                {
                    "student_user_id": sid,
                    "sp": sp,
                    "full_name": sp.student_name.get_full_name()
                    or sp.student_name.username,
                    "first_initial": (
                        sp.student_name.get_full_name()
                        or sp.student_name.username
                        or "?"
                    )[0].upper(),
                    "email": sp.student_name.email,
                    "matric": sp.matric_number or "N/A",
                    "department": sp.department or "N/A",
                    "faculty": sp.faculty or "N/A",
                    "year_of_study": sp.year_of_study
                    if sp.year_of_study and sp.year_of_study != 100
                    else "N/A",
                    "has_face_image": bool(sp.face_embedding),
                    "present_count": present_count,
                    "absent_count": absent_count,
                    "attendance_pct": attendance_pct,
                    "avg_distance": avg_dist,
                    "avg_face_similarity": avg_sim,
                    "last_seen": last_seen,
                    "risk_level": risk_level,
                    "risk_label": risk_label,
                }
            )

    # Overall course stats
    total_enrolled = len(students)
    overall_pct = 0
    if total_enrolled > 0 and course_expired_sessions > 0:
        overall_pct = round(
            (total_present_all / (total_enrolled * course_expired_sessions) * 100), 1
        )
    # Risk breakdown
    good_count = sum(1 for s in students if s["risk_level"] == "good")
    warn_count = sum(1 for s in students if s["risk_level"] == "warn")
    danger_count = sum(1 for s in students if s["risk_level"] == "danger")
    new_count = sum(1 for s in students if s["risk_level"] == "new")

    context = {
        "user_role": "lecturer",
        "lecturer": lecturer,
        "courses": courses,
        "selected_course": selected_course,
        "selected_course_id": selected_course_id,
        "search_q": search_q,
        "students": students,
        "total_enrolled": total_enrolled,
        "course_sessions_held": course_expired_sessions,
        "course_sessions_total": len(course_sessions_all),
        "course_sessions": course_sessions_all,
        "course_total_present": total_present_all,
        "course_overall_pct": overall_pct,
        "good_count": good_count,
        "warn_count": warn_count,
        "danger_count": danger_count,
        "new_count": new_count,
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "lecturer_course_roster.html", context)


# ========== LECTURER STUDENT DETAIL (per-student deep dive) ==========
@lecturer_required
def lecturer_student_detail(request, student_user_id, course_id=None):
    user_obj = User.objects.get(username=request.user)
    lecturer = get_object_or_404(LecturerProfile, user=user_obj)
    current_time = now()

    target_user = get_object_or_404(User, id=student_user_id)
    try:
        sp = StudentProfile.objects.get(student_name=target_user)
    except StudentProfile.DoesNotExist:
        sp = StudentProfile.objects.create(student_name=target_user)

    # Filter scope: either 1 course (if course_id given and lecturer teaches it) OR all lecturer's courses
    scope_courses = []
    if course_id:
        c = get_object_or_404(Course, id=course_id)
        if c in lecturer.courses_taught.all():
            scope_courses = [c]
            scope_course = c
        else:
            scope_courses = list(lecturer.courses_taught.all())
            scope_course = None
    else:
        scope_courses = list(lecturer.courses_taught.all())
        scope_course = None

    # All sessions in scope courses, regardless of which co-lecturer created
    # them — a course may be taught by several lecturers, and every session
    # counts toward the student's expected attendance.
    scope_sessions = list(
        AttendanceSession.objects.filter(
            course__in=scope_courses
        ).order_by("-start_time")
    )

    # Which of those scope courses is the student actually enrolled in?
    enrolled_in_scope = [
        c for c in scope_courses if sp.courses_enrolled.filter(id=c.id).exists()
    ]

    # Per-course breakdown
    per_course = []
    for c in enrolled_in_scope:
        course_sessions = [s for s in scope_sessions if s.course_id == c.id]
        expired = [s for s in course_sessions if s.end_time < current_time]
        recs = AttendanceRecord.objects.filter(
            session__in=expired, student=target_user
        ).select_related("session")
        present = recs.count()
        expired_count = len(expired)
        absent = max(0, expired_count - present)
        pct = round((present / expired_count * 100), 1) if expired_count > 0 else 0
        per_course.append(
            {
                "course": c,
                "expired_count": expired_count,
                "upcoming_count": sum(
                    1 for s in course_sessions if s.start_time > current_time
                ),
                "active_count": sum(
                    1
                    for s in course_sessions
                    if s.start_time <= current_time <= s.end_time
                ),
                "present": present,
                "absent": absent,
                "pct": pct,
            }
        )

    # Aggregate totals (across all scope)
    scope_expired = [
        s
        for s in scope_sessions
        if s.course in enrolled_in_scope and s.end_time < current_time
    ]
    all_records = (
        AttendanceRecord.objects.filter(session__in=scope_expired, student=target_user)
        .select_related("session", "session__course")
        .order_by("-session__start_time")
    )

    total_present = all_records.count()
    total_expired_scope_enrolled = len(scope_expired)
    overall_pct = (
        round((total_present / total_expired_scope_enrolled * 100), 1)
        if total_expired_scope_enrolled > 0
        else 0
    )
    total_absent = max(0, total_expired_scope_enrolled - total_present)

    # Session-by-session history for this student
    history = []
    for s in scope_sessions:
        # Only include history if student is enrolled in that course
        if not sp.courses_enrolled.filter(id=s.course_id).exists():
            continue
        rec = all_records.filter(session_id=s.id).first() if all_records else None
        if not rec:
            # Check all_records set:
            try:
                rec = AttendanceRecord.objects.get(session=s, student=target_user)
            except AttendanceRecord.DoesNotExist:
                rec = None
        if s.end_time < current_time:
            if rec:
                status_label = "Present"
                status_color = "green"
            else:
                status_label = "Absent"
                status_color = "red"
        elif s.start_time > current_time:
            status_label = "Upcoming"
            status_color = "indigo"
        else:
            if rec:
                status_label = "Present ✅"
                status_color = "green"
            else:
                status_label = "Pending"
                status_color = "orange"

        history.append(
            {
                "session": s,
                "date": s.start_time.strftime("%b %d, %Y"),
                "window": f"{s.start_time.strftime('%H:%M')}–{s.end_time.strftime('%H:%M')}",
                "hall": s.lecture_hall,
                "status_label": status_label,
                "status_color": status_color,
                "marked_time": format_lagos_time(rec.marked_time) if rec else None,
                "distance": round(rec.distance, 1) if rec else None,
                "face_similarity": round(rec.face_similarity, 4)
                if (rec and rec.face_similarity is not None)
                else None,
            }
        )

    # Risk flag
    if total_expired_scope_enrolled == 0:
        risk_level = "new"
        risk_label = "No completed sessions"
    elif overall_pct >= 80:
        risk_level = "good"
        risk_label = "On Track 🌟"
    elif overall_pct >= 60:
        risk_level = "warn"
        risk_label = "At Risk ⚠️"
    else:
        risk_level = "danger"
        risk_label = "Failing Attendance 🚨"

    context = {
        "user_role": "lecturer",
        "lecturer": lecturer,
        "sp": sp,
        "target_user": target_user,
        "student_full_name": target_user.get_full_name() or target_user.username,
        "first_initial": (target_user.get_full_name() or target_user.username or "?")[
            0
        ].upper(),
        "matric": sp.matric_number or "N/A",
        "department": sp.department or "N/A",
        "faculty": sp.faculty or "N/A",
        "year_of_study": sp.year_of_study
        if sp.year_of_study and sp.year_of_study != 100
        else "N/A",
        "email": target_user.email,
        "has_face_image": bool(sp.face_embedding),
        "scope_course": scope_course,
        "per_course": per_course,
        "total_expired_scope_enrolled": total_expired_scope_enrolled,
        "total_present": total_present,
        "total_absent": total_absent,
        "overall_pct": overall_pct,
        "history": history,
        "risk_level": risk_level,
        "risk_label": risk_label,
        "server_now_lagos": format_lagos_time(current_time),
    }
    return render(request, "lecturer_student_detail.html", context)
