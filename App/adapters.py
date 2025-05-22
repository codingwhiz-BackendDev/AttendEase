from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.core.exceptions import ValidationError
from django.contrib import messages
from django.shortcuts import redirect
from allauth.exceptions import ImmediateHttpResponse
from django.urls import reverse

class CustomAccountAdapter(DefaultAccountAdapter):
    def clean_email(self, email):
        """Restrict login to student or lecturer emails."""
        allowed_domains = ["@run.edu.ng", "@lecturer.edu.ng"]
        if not any(email.endswith(domain) for domain in allowed_domains):
            raise ValidationError("Only student (@run.edu.ng) and lecturer (@lecturer.edu.ng) emails are allowed.")
        return email

    def get_login_redirect_url(self, request):
        """Redirect students and lecturers to their respective dashboards after login."""
        user = request.user
        if user.email.endswith("@run.edu.ng"):
            return reverse("student_dashboard")  # Ensure this URL exists in `urls.py`
        elif user.email.endswith("@lecturer.edu.ng"):
            return reverse("lecturer_dashboard")  # Ensure this URL exists in `urls.py`
        return reverse("welcome_page")  # Fallback


class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):
        """Ensure login is for students or lecturers and validate emails accordingly."""
        process = request.session.get("login_process", "")
        email = sociallogin.account.extra_data.get("email", "")

        print(f"Process: {process}")  # Debugging
        print(f"Email: {email}")  # Debugging

        if process == "student" and not email.endswith("@run.edu.ng"):
            messages.error(request, "Only student emails (@run.edu.ng) are allowed.")
            print("Blocking non-student email")  # Debugging
            raise ImmediateHttpResponse(redirect("/student/login/"))
        elif process == "lecturer" and not email.endswith("@lecturer.edu.ng"):
            messages.error(request, "Only lecturer emails (@lecturer.edu.ng) are allowed.")
            print("Blocking non-lecturer email")  # Debugging
            raise ImmediateHttpResponse(redirect("/lecturer/login/"))

        # Clear session data after login
        request.session.pop("login_process", None)

    def get_login_redirect_url(self, request):
        """Redirect based on stored session login type."""
        process = request.session.get("login_process", "")
        print(f"Redirecting based on process: {process}")  # Debugging

        if process == "student":
            return reverse("student_dashboard")
        elif process == "lecturer":
            return reverse("lecturer_dashboard")

        return super().get_login_redirect_url(request)  # Default
