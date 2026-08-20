"""
AI Study Hub - Gemini AI Integration
Handles all AI-powered study features using Google's Gemini API
"""

import os
import json
import re
from django.conf import settings
import google.generativeai as genai


class AIStudyAssistant:
    """Main AI assistant for study hub features"""
    
    def __init__(self):
        """Initialize Gemini AI client"""
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured in settings")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
    def _build_context(self, course_code, materials_text=""):
        """Build context from course materials"""
        context = f"""You are an AI study assistant for a university course: {course_code}.
        
Your role is to help students learn from their course materials. Always:
- Be accurate and educational
- Explain concepts clearly with examples
- When asked about topics not in materials, clearly state this
- Use the provided course materials as your primary source
- Be encouraging and supportive
"""
        if materials_text:
            context += f"\n\nCourse Materials Context:\n{materials_text}\n\n"
        return context
    
    def chat(self, question, course_code, materials_text="", chat_history=[]):
        """Handle chat Q&A with course materials context"""
        context = self._build_context(course_code, materials_text)
        
        # Build conversation history
        conversation = context + "\n\nConversation:\n"
        for msg in chat_history[-5:]:  # Keep last 5 messages for context
            role = "Student" if msg['role'] == 'user' else "Assistant"
            conversation += f"{role}: {msg['content']}\n"
        conversation += f"Student: {question}\n\nAssistant:"
        
        try:
            response = self.model.generate_content(conversation)
            return response.text, True
        except Exception as e:
            return f"Error generating response: {str(e)}", False
    
    def summarize(self, topic, course_code, materials_text=""):
        """Generate summary of a topic from materials"""
        context = self._build_context(course_code, materials_text)
        prompt = f"""{context}

Task: Create a comprehensive summary of the topic: "{topic}"

Format your response as:
1. Key Concepts (bullet points)
2. Important Definitions & Formulas
3. What Students Should Focus On for Revision
4. Common Misconceptions to Avoid

Be thorough but concise. Use the course materials as your source."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text, True
        except Exception as e:
            return f"Error generating summary: {str(e)}", False
    
    def explain(self, concept, course_code, materials_text=""):
        """Explain a concept in simple terms with examples"""
        context = self._build_context(course_code, materials_text)
        prompt = f"""{context}

Task: Explain the concept "{concept}" in simple, easy-to-understand terms.

Your explanation should:
- Start with a one-sentence summary
- Use analogies or real-world examples
- Break down complex ideas into simple parts
- Include why this concept matters
- Use the course materials as your source

Be conversational and encouraging, like a patient tutor."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text, True
        except Exception as e:
            return f"Error generating explanation: {str(e)}", False
    
    def generate_quiz(self, num_questions, difficulty, topics, course_code, materials_text=""):
        """Generate MCQ quiz from materials with difficulty levels"""
        context = self._build_context(course_code, materials_text)
        
        # Enhanced prompt with difficulty-specific instructions
        difficulty_instructions = {
            'easy': "Make questions straightforward with clear, obvious answers. Focus on basic concepts and definitions.",
            'medium': "Make questions moderately challenging. Include some that require applying concepts.",
            'hard': "Make questions challenging that require deep understanding, synthesis, or application of multiple concepts."
        }
        
        difficulty_instruction = difficulty_instructions.get(difficulty.lower(), difficulty_instructions['medium'])
        
        prompt = f"""{context}

Task: Generate {num_questions} multiple-choice questions about: {topics}

Difficulty level: {difficulty}
Specific requirement: {difficulty_instruction}

Format each question as JSON:
{{
    "question": "Question text",
    "options": ["A", "B", "C", "D"],
    "correct_answer": 0,
    "explanation": "Why this is correct",
    "difficulty": "{difficulty}",
    "topic": "specific topic covered"
}}

Return the entire quiz as a JSON array of questions.
Make questions realistic and based on the course materials.
Ensure questions match the requested difficulty level."""
        
        try:
            response = self.model.generate_content(prompt)
            # Parse JSON from response
            text = response.text
            # Extract JSON array
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                quiz_data = json.loads(json_match.group())
                return quiz_data, True
            else:
                return {"error": "Could not parse quiz JSON"}, False
        except Exception as e:
            return {"error": f"Error generating quiz: {str(e)}"}, False
    
    def generate_flashcards(self, num_cards, topics, course_code, materials_text=""):
        """Generate flashcards from materials with categorization"""
        context = self._build_context(course_code, materials_text)
        prompt = f"""{context}

Task: Generate {num_cards} flashcards about: {topics}

Format each flashcard as JSON:
{{
    "question": "Front of card (question/term)",
    "answer": "Back of card (answer/definition)",
    "category": "definition/formula/concept/example",
    "difficulty": "easy/medium/hard"
}}

Return the entire set as a JSON array of flashcards.
Focus on key terms, definitions, formulas, and important concepts from the materials.
Include a mix of easy and challenging cards for comprehensive study."""
        
        try:
            response = self.model.generate_content(prompt)
            # Parse JSON from response
            text = response.text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                flashcards = json.loads(json_match.group())
                return flashcards, True
            else:
                return {"error": "Could not parse flashcards JSON"}, False
        except Exception as e:
            return {"error": f"Error generating flashcards: {str(e)}"}, False
    
    def practice_question(self, course_code, materials_text="", previous_questions=[]):
        """Generate one practice question for interactive mode"""
        context = self._build_context(course_code, materials_text)
        prev_q_text = "\n".join([f"- {q}" for q in previous_questions[-3:]])
        
        prompt = f"""{context}

Task: Generate ONE practice question for the student to answer in their own words.

Previous questions asked (avoid repeating):
{prev_q_text}

Format as JSON:
{{
    "question": "Open-ended question requiring explanation",
    "expected_points": ["key point 1", "key point 2", "key point 3"],
    "difficulty": "easy/medium/hard"
}}

Make the question thought-provoking and based on course materials."""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                question = json.loads(json_match.group())
                return question, True
            else:
                return {"error": "Could not parse question JSON"}, False
        except Exception as e:
            return {"error": f"Error generating question: {str(e)}"}, False
    
    def evaluate_answer(self, question, student_answer, expected_points, course_code):
        """Evaluate student's practice answer"""
        context = self._build_context(course_code)
        points_text = "\n".join([f"- {p}" for p in expected_points])
        
        prompt = f"""{context}

Question: {question}
Student's Answer: {student_answer}

Expected key points to cover:
{points_text}

Evaluate the student's answer and provide:
1. Score (0-100)
2. What they did well
3. What they missed
4. How to improve

Format as JSON:
{{
    "score": 85,
    "feedback": "Your answer...",
    "strong_points": ["point 1", "point 2"],
    "missed_points": ["point 1", "point 2"],
    "improvement_tips": ["tip 1", "tip 2"]
}}

Be encouraging and constructive."""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return evaluation, True
            else:
                return {"error": "Could not parse evaluation JSON"}, False
        except Exception as e:
            return {"error": f"Error evaluating answer: {str(e)}"}, False


def get_materials_text(materials):
    """Extract text from course materials for AI context"""
    if not materials:
        return ""
    
    text_parts = []
    for material in materials:
        if hasattr(material, 'content') and material.content:
            text_parts.append(f"=== {material.title} ===\n{material.content}")
        elif hasattr(material, 'file_path'):
            # For file-based materials, we'd need to extract text
            # This is a placeholder - implement file text extraction as needed
            text_parts.append(f"=== {material.title} ===\n[File content - text extraction needed]")
    
    return "\n\n".join(text_parts)
