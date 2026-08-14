from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from App.models import Assessment, AssessmentAttempt, AssessmentQuestion, AssessmentOption, StudentProfile, Course, LecturerProfile
import json

User = get_user_model()


class AntiCheatTestCase(TestCase):
    """Test cases for anti-cheat functionality"""

    def setUp(self):
        """Set up test data"""
        # Create users
        self.lecturer_user = User.objects.create_user(
            username='lecturer@test.com',
            email='lecturer@test.com',
            password='testpass123'
        )
        self.student_user = User.objects.create_user(
            username='student@test.com',
            email='student@test.com',
            password='testpass123'
        )

        # Create profiles
        self.lecturer = LecturerProfile.objects.create(
            user=self.lecturer_user,
            staff_id='LEC001',
            department='Computer Science'
        )
        self.student = StudentProfile.objects.create(
            student_name=self.student_user,
            matric_number='MAT001',
            department='Computer Science'
        )

        # Create course
        self.course = Course.objects.create(
            course_code='CS101',
            course_title='Introduction to Computer Science',
            credit=3
        )
        self.student.courses_enrolled.add(self.course)

        # Create assessment
        self.assessment = Assessment.objects.create(
            title='Test Assessment',
            lecturer=self.lecturer,
            status=Assessment.STATUS_PUBLISHED,
            time_limit_minutes=30
        )
        self.assessment.courses.add(self.course)

        # Create question
        self.question = AssessmentQuestion.objects.create(
            assessment=self.assessment,
            question_text='What is 2+2?',
            question_type=AssessmentQuestion.TYPE_MCQ,
            points=1
        )

        # Create options
        self.option1 = AssessmentOption.objects.create(
            question=self.question,
            option_text='3',
            is_correct=False
        )
        self.option2 = AssessmentOption.objects.create(
            question=self.question,
            option_text='4',
            is_correct=True
        )

        # Create assessment attempt
        self.attempt = AssessmentAttempt.objects.create(
            assessment=self.assessment,
            student=self.student_user,
            status=AssessmentAttempt.STATUS_IN_PROGRESS
        )

        self.client = Client()
        self.client.login(username='student@test.com', password='testpass123')

    def test_violation_endpoint_valid_type(self):
        """Test violation reporting with valid type"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        data = {
            'type': 'tab_switch',
            'detail': 'Test violation'
        }
        
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data['ok'])
        self.assertEqual(response_data['violation_count'], 1)
        self.assertFalse(response_data['flagged'])

    def test_violation_endpoint_invalid_type(self):
        """Test violation reporting with invalid type"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        data = {
            'type': 'invalid_type',
            'detail': 'Test violation'
        }
        
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['ok'])

    def test_violation_endpoint_rate_limiting(self):
        """Test rate limiting on violation endpoint"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        data = {
            'type': 'tab_switch',
            'detail': 'Test violation'
        }
        
        # Make 10 requests (should be allowed)
        for i in range(10):
            response = self.client.post(
                url,
                data=json.dumps(data),
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200)
        
        # 11th request should be rate limited
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 429)

    def test_violation_auto_submit_trigger(self):
        """Test auto-submit trigger when threshold is reached"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        
        # Report 2 violations (should not trigger auto-submit)
        for i in range(2):
            data = {'type': 'tab_switch', 'detail': f'Violation {i+1}'}
            response = self.client.post(
                url,
                data=json.dumps(data),
                content_type='application/json'
            )
            response_data = json.loads(response.content)
            self.assertFalse(response_data['auto_submit'])
        
        # 3rd violation should trigger auto-submit
        data = {'type': 'tab_switch', 'detail': 'Violation 3'}
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type='application/json'
        )
        response_data = json.loads(response.content)
        self.assertTrue(response_data['auto_submit'])
        
        # Refresh attempt from database
        self.attempt.refresh_from_db()
        self.assertTrue(self.attempt.flagged)
        self.assertEqual(self.attempt.violation_count, 3)

    def test_violation_log_storage(self):
        """Test that violations are properly logged"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        
        violation_types = ['tab_switch', 'copy', 'devtools']
        for i, vtype in enumerate(violation_types):
            data = {'type': vtype, 'detail': f'Violation {i+1}'}
            self.client.post(
                url,
                data=json.dumps(data),
                content_type='application/json'
            )
        
        # Refresh attempt from database
        self.attempt.refresh_from_db()
        
        self.assertEqual(self.attempt.violation_count, 3)
        self.assertEqual(len(self.attempt.violation_log), 3)
        
        # Check log entries
        logged_types = [entry['type'] for entry in self.attempt.violation_log]
        self.assertEqual(logged_types, violation_types)

    def test_unauthorized_access_to_violation_endpoint(self):
        """Test that only the attempt owner can report violations"""
        # Create another student
        other_user = User.objects.create_user(
            username='other@test.com',
            email='other@test.com',
            password='testpass123'
        )
        
        # Login as other student
        self.client.logout()
        self.client.login(username='other@test.com', password='testpass123')
        
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        data = {'type': 'tab_switch', 'detail': 'Test'}
        
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        # Should get 404 or 403 (attempt doesn't belong to this user)
        self.assertIn(response.status_code, [403, 404])

    def test_violation_endpoint_invalid_json(self):
        """Test violation reporting with invalid JSON"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        
        response = self.client.post(
            url,
            data='invalid json',
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)

    def test_multiple_violation_types(self):
        """Test that different violation types are accepted"""
        url = reverse('millialms_report_violation', kwargs={'attempt_id': self.attempt.id})
        
        valid_types = [
            'tab_switch', 'blur', 'copy', 'cut', 'paste', 'devtools',
            'fullscreen_exit', 'right_click', 'print_screen', 'print_attempt'
        ]
        
        for vtype in valid_types:
            data = {'type': vtype, 'detail': 'Test'}
            response = self.client.post(
                url,
                data=json.dumps(data),
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200, f"Failed for type: {vtype}")
        
        # Refresh and check count
        self.attempt.refresh_from_db()
        self.assertEqual(self.attempt.violation_count, len(valid_types))
