from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.core.exceptions import ValidationError
from django.contrib import messages
from django.shortcuts import redirect
from allauth.exceptions import ImmediateHttpResponse
from django.urls import reverse
from django.db import transaction
import logging

logger = logging.getLogger(__name__)

class CustomAccountAdapter(DefaultAccountAdapter):
    def clean_email(self, email):
        """Restrict login to student or lecturer emails."""
        allowed_domains = ["@run.edu.ng", "@gmail.com"]
        if not any(email.endswith(domain) for domain in allowed_domains):
            raise ValidationError("Only student (@run.edu.ng) and lecturer (@gmail.com) emails are allowed.")
        return email

    def _ensure_profile_and_get_url(self, user):
        """Ensure the correct profile exists, then return the dashboard URL.

        Uses get_or_create so a duplicate/failed insert can never raise and
        bounce the user back to the welcome page. Role is derived from the
        email domain (@run.edu.ng = student, otherwise lecturer).
        """
        from .models import StudentProfile, LecturerProfile

        if user.email.endswith("@run.edu.ng"):
            StudentProfile.objects.get_or_create(student_name=user)
            logger.info(f"Ensured StudentProfile for user {user.email}")
            return reverse("student_dashboard")

        LecturerProfile.objects.get_or_create(user=user)
        logger.info(f"Ensured LecturerProfile for user {user.email}")
        return reverse("lecturer_dashboard")

    def get_login_redirect_url(self, request):
        """Runs on every returning login."""
        return self._ensure_profile_and_get_url(request.user)

    def get_signup_redirect_url(self, request):
        """Runs on FIRST-TIME signup. allauth uses a separate redirect for
        signup (ACCOUNT_SIGNUP_REDIRECT_URL, default '/'), which is why new
        lecturers were landing on the welcome page instead of the dashboard.
        """
        return self._ensure_profile_and_get_url(request.user)


class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):
        """Ensure login is for students or lecturers and validate emails accordingly."""
        from .models import StudentProfile, LecturerProfile
        from django.contrib.auth.models import User

        process = request.session.get("login_process", "")
        email = sociallogin.account.extra_data.get("email", "")

        logger.info(f"Login process: {process}, Email: {email}")

        if process == "student" and not email.endswith("@run.edu.ng"):
            messages.error(
                request,
                f"🚫 Student portal requires a school email (@run.edu.ng).\n"
                f"You tried logging in with '{email}'.\n"
                f"Please use your official Redeemer's University student email, or click the Lecturer portal instead.",
            )
            logger.warning(f"Blocked non-student email for student portal: {email}")
            # Redirect to welcome_page (RENDERED, not another redirect) so messages display
            raise ImmediateHttpResponse(redirect("welcome_page"))
        elif process == "lecturer" and not email.endswith("@gmail.com"):
            messages.error(
                request,
                f"🚫 Lecturer portal requires a Gmail account (@gmail.com).\n"
                f"You tried logging in with '{email}'.\n"
                f"Please sign in with your Gmail address, or click the Student portal instead.",
            )
            logger.warning(f"Blocked non-lecturer email for lecturer portal: {email}")
            # Redirect to welcome_page (RENDERED, not another redirect) so messages display
            raise ImmediateHttpResponse(redirect("welcome_page"))

        # Auto-link social account to existing user by email
        if not sociallogin.is_existing:
            try:
                user = User.objects.get(email=email)
                sociallogin.user = user
                logger.info(f"Auto-linked Google account to existing user: {email}")
            except User.DoesNotExist:
                logger.info(f"No existing user found for {email}, will create new account")

        # Auto-create profile if doesn't exist
        with transaction.atomic():
            if process == "student":
                if not StudentProfile.objects.filter(student_name__email=email).exists():
                    # User will be created by allauth, so we need to handle this after user creation
                    logger.info(f"Will create StudentProfile for {email}")
            elif process == "lecturer":
                if not LecturerProfile.objects.filter(user__email=email).exists():
                    logger.info(f"Will create LecturerProfile for {email}")

        # NOTE: Do NOT pop login_process here. It is consumed AFTER login in
        # get_login_redirect_url(). Popping early would break role-based dashboard redirect.

    def get_login_redirect_url(self, request):
        """Redirect based on stored session login type (with email-domain fallback)."""
        from .models import StudentProfile, LecturerProfile

        process = request.session.get("login_process", "")
        user = request.user
        logger.info(f"Social redirect: login_process={process!r}, email={user.email!r}")

        # Determine role: prefer explicit session process, fall back to email domain
        if not process:
            process = "student" if user.email.endswith("@run.edu.ng") else "lecturer"
            logger.info(f"No login_process in session; inferred role '{process}' from email domain")

        # Consume the session flag now that we've read it (prevents stale state)
        request.session.pop("login_process", None)

        if process == "student":
            with transaction.atomic():
                StudentProfile.objects.get_or_create(student_name=user)
            logger.info(f"Social login redirect: user {user.email} → student_dashboard")
            return reverse("student_dashboard")
        elif process == "lecturer":
            with transaction.atomic():
                LecturerProfile.objects.get_or_create(user=user)
            logger.info(f"Social login redirect: user {user.email} → lecturer_dashboard")
            return reverse("lecturer_dashboard")

        # Safety fallback (should never reach here due to email inference above)
        logger.warning(f"Social redirect fallback triggered for user {user.email}")
        return reverse("welcome_page")
