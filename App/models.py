from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator
from django.db import models
from django.utils import timezone

User = get_user_model()


class Course(models.Model):
    course_code = models.CharField(max_length=20, unique=True)
    course_title = models.CharField(max_length=255)
    credit = models.IntegerField()

    class Meta:
        indexes = [
            models.Index(fields=["course_code"]),
            models.Index(fields=["course_title"]),
        ]

    def __str__(self):
        return self.course_title


class StudentProfile(models.Model):
    student_name = models.ForeignKey(User, on_delete=models.CASCADE)
    department = models.CharField(max_length=255, null=True, default="")
    faculty = models.CharField(max_length=255, null=True, default="")
    matric_number = models.CharField(max_length=255, null=True, default="")
    year_of_study = models.IntegerField(default=100, null=True)
    courses_enrolled = models.ManyToManyField(Course, related_name="students")
    # Privacy-first: we store ONLY the face embedding (a numeric vector), never
    # the raw photo. The captured image is processed in memory and discarded.
    face_embedding = models.JSONField(null=True, blank=True)

    @property
    def has_face_data(self):
        """True once the student has enrolled a face embedding."""
        return bool(self.face_embedding)

    class Meta:
        indexes = [
            models.Index(fields=["matric_number"]),
            models.Index(fields=["department"]),
        ]

    def __str__(self):
        return f"{self.student_name} ({self.matric_number})"


class LecturerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    staff_id = models.CharField(max_length=255, unique=True, null=True, blank=True)
    department = models.CharField(max_length=100, null=True, blank=True)
    academic_rank = models.CharField(max_length=50, null=True, blank=True)
    office_location = models.CharField(max_length=255, null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    courses_taught = models.ManyToManyField(
        "Course", related_name="lecturers"
    )  # Many-to-many relationship

    class Meta:
        indexes = [
            models.Index(fields=["staff_id"]),
            models.Index(fields=["department"]),
        ]

    def __str__(self):
        return f"{self.user} ({self.staff_id})"


class AttendanceSession(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    lecturer = models.ForeignKey(LecturerProfile, on_delete=models.CASCADE)
    lecture_hall = models.CharField(max_length=255)
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)
    student_marked = models.ManyToManyField(User, related_name="students")

    # Geofencing Location Data
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    radius = models.IntegerField(default=100, help_text="Geofence radius in meters")

    class Meta:
        indexes = [
            models.Index(fields=["start_time"]),
            models.Index(fields=["course", "lecturer"]),
        ]

    def __str__(self):
        return f"Attendance for {self.course.course_title} by {self.lecturer.user}"


class AttendanceRecord(models.Model):
    session = models.ForeignKey(
        AttendanceSession, on_delete=models.CASCADE, related_name="records"
    )
    student = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="attendance_records"
    )
    marked_time = models.DateTimeField(auto_now_add=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    distance = models.FloatField(help_text="Distance in meters from center")
    status = models.CharField(max_length=20, default="Present")
    face_similarity = models.FloatField(
        null=True, blank=True, help_text="Cosine similarity distance"
    )

    class Meta:
        unique_together = ("session", "student")
        indexes = [
            models.Index(fields=["session", "student"]),
        ]

    def __str__(self):
        return f"{self.student.username} marked present for {self.session.course.course_title}"


class Assessment(models.Model):
    TYPE_QUIZ = "quiz"
    TYPE_TIMED_TEST = "timed_test"
    TYPE_PRACTICE = "practice"
    TYPE_CHOICES = [
        (TYPE_QUIZ, "Quiz"),
        (TYPE_TIMED_TEST, "Timed Test"),
        (TYPE_PRACTICE, "Practice Test"),
    ]

    STATUS_DRAFT = "draft"
    STATUS_PUBLISHED = "published"
    STATUS_UNPUBLISHED = "unpublished"
    STATUS_CHOICES = [
        (STATUS_DRAFT, "Draft"),
        (STATUS_PUBLISHED, "Published"),
        (STATUS_UNPUBLISHED, "Unpublished"),
    ]

    title = models.CharField(max_length=255)
    instructions = models.TextField(blank=True)
    assessment_type = models.CharField(
        max_length=20, choices=TYPE_CHOICES, default=TYPE_QUIZ
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_DRAFT
    )
    lecturer = models.ForeignKey(
        LecturerProfile, on_delete=models.CASCADE, related_name="assessments"
    )
    courses = models.ManyToManyField(Course, related_name="assessments", blank=True)
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    time_limit_minutes = models.PositiveIntegerField(
        null=True, blank=True, validators=[MinValueValidator(1)]
    )
    max_attempts = models.PositiveIntegerField(
        default=1, validators=[MinValueValidator(1)]
    )
    randomize_question_order = models.BooleanField(default=False)
    randomize_answer_options = models.BooleanField(default=False)
    is_auto_publish = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["lecturer", "status"]),
            models.Index(fields=["start_time", "end_time"]),
        ]

    def __str__(self):
        return self.title

    @property
    def total_questions(self):
        return self.questions.count()

    @property
    def is_published(self):
        return self.status == self.STATUS_PUBLISHED


class AssessmentQuestion(models.Model):
    TYPE_MCQ = "mcq"
    TYPE_TRUE_FALSE = "true_false"
    TYPE_FILL_BLANK = "fill_blank"
    TYPE_SHORT_ANSWER = "short_answer"
    TYPE_ESSAY = "essay"
    QUESTION_TYPE_CHOICES = [
        (TYPE_MCQ, "Multiple Choice"),
        (TYPE_TRUE_FALSE, "True / False"),
        (TYPE_FILL_BLANK, "Fill in the Blank"),
        (TYPE_SHORT_ANSWER, "Short Answer"),
        (TYPE_ESSAY, "Essay"),
    ]

    assessment = models.ForeignKey(
        Assessment, on_delete=models.CASCADE, related_name="questions"
    )
    question_text = models.TextField()
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPE_CHOICES)
    points = models.DecimalField(max_digits=6, decimal_places=2, default=1)
    order = models.PositiveIntegerField(default=1)
    accepted_answers = models.TextField(
        blank=True,
        help_text="For fill-in-the-blank, separate accepted answers with new lines.",
    )
    manual_grading_required = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "id"]
        indexes = [
            models.Index(fields=["assessment", "order"]),
        ]

    def __str__(self):
        return f"{self.assessment.title} - Q{self.order}"


class AssessmentOption(models.Model):
    question = models.ForeignKey(
        AssessmentQuestion, on_delete=models.CASCADE, related_name="options"
    )
    option_text = models.CharField(max_length=500)
    is_correct = models.BooleanField(default=False)
    order = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ["order", "id"]
        indexes = [
            models.Index(fields=["question", "order"]),
        ]

    def __str__(self):
        return self.option_text


class AssessmentAttempt(models.Model):
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_SUBMITTED = "submitted"
    STATUS_GRADED = "graded"
    STATUS_CHOICES = [
        (STATUS_IN_PROGRESS, "In Progress"),
        (STATUS_SUBMITTED, "Submitted"),
        (STATUS_GRADED, "Graded"),
    ]

    assessment = models.ForeignKey(
        Assessment, on_delete=models.CASCADE, related_name="attempts"
    )
    student = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="assessment_attempts"
    )
    attempt_number = models.PositiveIntegerField(default=1)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_IN_PROGRESS
    )
    started_at = models.DateTimeField(auto_now_add=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    auto_score = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    manual_score = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    total_score = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    total_possible_score = models.DecimalField(
        max_digits=8, decimal_places=2, default=0
    )

    class Meta:
        unique_together = ("assessment", "student", "attempt_number")
        ordering = ["-started_at"]
        indexes = [
            models.Index(fields=["assessment", "student"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"{self.student.username} - {self.assessment.title} (Attempt {self.attempt_number})"


class AssessmentResponse(models.Model):
    attempt = models.ForeignKey(
        AssessmentAttempt, on_delete=models.CASCADE, related_name="responses"
    )
    question = models.ForeignKey(
        AssessmentQuestion, on_delete=models.CASCADE, related_name="responses"
    )
    selected_option = models.ForeignKey(
        AssessmentOption,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="selected_responses",
    )
    text_answer = models.TextField(blank=True)
    is_correct = models.BooleanField(default=False)
    auto_score = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    manual_score = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    graded_manually = models.BooleanField(default=False)
    lecturer_feedback = models.TextField(blank=True)

    class Meta:
        unique_together = ("attempt", "question")
        indexes = [
            models.Index(fields=["attempt", "question"]),
        ]

    def __str__(self):
        return f"Response: {self.attempt} / {self.question}"
