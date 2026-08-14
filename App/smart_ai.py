"""
Smart AI Teaching System - Advanced RAG-based Virtual Tutor
This is NOT a generic chatbot - it's an intelligent teaching system with:
- Vector embeddings for semantic search
- Retrieval-Augmented Generation (RAG)
- Learning memory and adaptation
- Citation system
- Multi-step reasoning
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from django.conf import settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Intelligent document chunking for better retrieval"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into intelligent chunks"""
        if not text or not text.strip():
            return []
        
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_metadata = metadata.copy() if metadata else {}
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is small, add to current chunk
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if it exists
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': current_metadata.copy()
                    })
                
                # Start new chunk with overlap
                if len(para) > self.chunk_size:
                    # Very long paragraph - split by sentences
                    sentences = re.split(r'[.!?]+', para)
                    for sent in sentences:
                        if sent.strip():
                            chunks.append({
                                'text': sent.strip(),
                                'metadata': current_metadata.copy()
                            })
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': current_metadata.copy()
            })
        
        return chunks


class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []
        self.metadata = []
    
    def add_chunks(self, chunks: List[Dict]):
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend([chunk['text'] for chunk in chunks])
        self.metadata.extend([chunk.get('metadata', {}) for chunk in chunks])
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Search for relevant chunks"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], self.metadata[idx], float(score)))
        
        return results


class LearningMemory:
    """Track student learning progress and understanding"""
    
    def __init__(self, student_id: str):
        self.student_id = student_id
        self.conversation_history = []
        self.topics_mastered = set()
        self.topics_struggling = set()
        self.questions_asked = []
        self.last_study_session = None
    
    def add_interaction(self, question: str, answer: str, topic: str, rating: float):
        """Record a learning interaction"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'topic': topic,
            'rating': rating  # 0-1, how well the student understood
        })
        
        if rating >= 0.8:
            self.topics_mastered.add(topic)
        elif rating <= 0.4:
            self.topics_struggling.add(topic)
        
        self.questions_asked.append(question)
        self.last_study_session = datetime.now()
    
    def get_learning_context(self) -> Dict:
        """Get context about student's learning state"""
        return {
            'mastered_topics': list(self.topics_mastered),
            'struggling_topics': list(self.topics_struggling),
            'recent_questions': self.questions_asked[-10:],
            'study_frequency': len(self.conversation_history)
        }


class SmartAITutor:
    """
    Advanced AI Tutor with RAG, learning memory, and adaptive teaching
    This is NOT a generic chatbot - it's an intelligent teaching system
    """
    
    def __init__(self, course_code: str, student_id: str):
        self.course_code = course_code
        self.student_id = student_id
        
        # Initialize components
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()
        self.learning_memory = LearningMemory(student_id)
        
        # Initialize Gemini
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Load course materials
        self._load_course_materials()
    
    def _load_course_materials(self):
        """Load and index course materials"""
        try:
            from App.models import Course, CourseMaterial
            course = Course.objects.get(course_code=self.course_code)
            materials = CourseMaterial.objects.filter(course=course)
            
            all_chunks = []
            for material in materials:
                if material.content:
                    chunks = self.chunker.chunk_text(
                        material.content,
                        metadata={
                            'material_id': material.id,
                            'title': material.title,
                            'type': material.material_type
                        }
                    )
                    all_chunks.extend(chunks)
            
            if all_chunks:
                self.vector_store.add_chunks(all_chunks)
                logger.info(f"Loaded {len(all_chunks)} chunks for {self.course_code}")
            
        except Exception as e:
            logger.error(f"Error loading materials: {str(e)}")
    
    def _retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant document chunks"""
        results = self.vector_store.search(query, top_k)
        
        context = []
        for text, metadata, score in results:
            context.append({
                'text': text,
                'metadata': metadata,
                'relevance_score': score
            })
        
        return context
    
    def _build_teaching_prompt(self, query: str, context: List[Dict], learning_context: Dict) -> str:
        """Build sophisticated teaching prompt"""
        
        # Format retrieved context
        context_text = "\n\n---\n\n".join([
            f"[Source: {c['metadata'].get('title', 'Unknown')} - Relevance: {c['relevance_score']:.2f}]\n{c['text']}"
            for c in context[:3]  # Use top 3 most relevant
        ])
        
        # Format learning context
        learning_text = ""
        if learning_context['mastered_topics']:
            learning_text += f"\nTopics student has mastered: {', '.join(learning_context['mastered_topics'])}\n"
        if learning_context['struggling_topics']:
            learning_text += f"Topics student is struggling with: {', '.join(learning_context['struggling_topics'])}\n"
        
        prompt = f"""You are an expert virtual tutor for the course {self.course_code}. Your role is to teach, not just answer questions.

TEACHING PRINCIPLES:
1. Use the retrieved course materials as your PRIMARY source of truth
2. Cite specific materials when explaining concepts
3. Adapt your explanation complexity based on the student's learning level
4. If the student is struggling with a topic, provide more examples and simpler explanations
5. If the student has mastered related topics, make connections to build deeper understanding
6. Always be encouraging and supportive
7. Use analogies and real-world examples when helpful
8. Break down complex ideas into step-by-step explanations
9. Check for understanding by asking follow-up questions
10. If information is not in the materials, clearly state this

COURSE MATERIALS (most relevant):
{context_text if context_text else "No materials available yet"}

STUDENT LEARNING CONTEXT:
{learning_text if learning_text else "No learning history yet"}

STUDENT'S QUESTION:
{query}

Provide a comprehensive, well-structured response that:
- Directly answers the question using course materials
- Explains the "why" and "how", not just the "what"
- Provides examples or analogies when helpful
- Connects to related concepts if the student has mastered them
- Asks a follow-up question to check understanding
- Cites which materials you used (e.g., "[From: Week 3 Notes]")"""
        
        return prompt
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract material citations from response"""
        citations = re.findall(r'\[From: ([^\]]+)\]', response)
        return citations
    
    def _rate_student_understanding(self, question: str, response: str) -> float:
        """Estimate how well the student understood based on their follow-up"""
        # This is a simplified version - could use AI to evaluate
        if len(response) < 50:
            return 0.3  # Likely didn't understand well
        elif "?" in response:
            return 0.6  # Asking follow-up - partially understood
        else:
            return 0.8  # Good response
    
    def teach(self, question: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Main teaching method - NOT a generic chat response
        Returns intelligent, context-aware teaching response
        """
        # Retrieve relevant context
        context = self._retrieve_relevant_context(question)
        
        # Get learning context
        learning_context = self.learning_memory.get_learning_context()
        
        # Build teaching prompt
        prompt = self._build_teaching_prompt(question, context, learning_context)
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Extract citations
            citations = self._extract_citations(answer)
            
            # Detect topic from question (simplified)
            topic = self._detect_topic(question)
            
            # Record interaction
            self.learning_memory.add_interaction(
                question=question,
                answer=answer,
                topic=topic,
                rating=0.7  # Default rating, would be updated based on student's response
            )
            
            return {
                'answer': answer,
                'citations': citations,
                'context_used': len(context),
                'topic': topic,
                'confidence': self._calculate_confidence(context),
                'learning_state': learning_context
            }
            
        except Exception as e:
            logger.error(f"Teaching error: {str(e)}")
            return {
                'error': str(e),
                'answer': "I'm having trouble accessing my knowledge base. Please try again."
            }
    
    def _detect_topic(self, question: str) -> str:
        """Detect the main topic from a question"""
        # Simplified topic detection - could use NLP
        words = question.lower().split()
        # Remove common words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'explain', 'describe'}
        topic_words = [w for w in words if w not in stop_words and len(w) > 3]
        return topic_words[0] if topic_words else "general"
    
    def _calculate_confidence(self, context: List[Dict]) -> float:
        """Calculate confidence based on context relevance"""
        if not context:
            return 0.3  # Low confidence without context
        
        avg_score = sum(c['relevance_score'] for c in context) / len(context)
        return min(avg_score, 1.0)
    
    def generate_adaptive_quiz(self, num_questions: int, difficulty: str, focus_topics: List[str] = None) -> List[Dict]:
        """Generate quiz adapted to student's learning state"""
        learning_context = self.learning_memory.get_learning_context()
        
        # Focus on struggling topics if specified
        if not focus_topics and learning_context['struggling_topics']:
            focus_topics = learning_context['struggling_topics']
        
        # Build quiz generation prompt
        prompt = f"""Generate {num_questions} multiple-choice questions for {self.course_code}.

STUDENT LEARNING STATE:
- Mastered topics: {', '.join(learning_context['mastered_topics'])}
- Struggling topics: {', '.join(learning_context['struggling_topics'])}
- Recent questions: {', '.join(learning_context['recent_questions'][-5:])}

DIFFICULTY: {difficulty}
FOCUS TOPICS: {', '.join(focus_topics) if focus_topics else 'General course content'}

TEACHING APPROACH:
- If student is struggling with a topic, make questions easier and more foundational
- If student has mastered topics, create more challenging application questions
- Each question should test understanding, not just memorization
- Include clear explanations for why the correct answer is right

Format as JSON array:
{{
    "question": "Question text",
    "options": ["A", "B", "C", "D"],
    "correct_answer": 0,
    "explanation": "Why this is correct",
    "topic": "topic_name",
    "difficulty": "easy/medium/hard"
}}"""
        
        try:
            response = self.model.generate_content(prompt)
            quiz_json = self._extract_json(response.text)
            return quiz_json if quiz_json else []
        except Exception as e:
            logger.error(f"Quiz generation error: {str(e)}")
            return []
    
    def _extract_json(self, text: str) -> List:
        """Extract JSON array from response"""
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return []
    
    def evaluate_student_answer(self, question: str, student_answer: str, expected_points: List[str]) -> Dict:
        """Evaluate student's answer with detailed feedback"""
        learning_context = self.learning_memory.get_learning_context()
        
        prompt = f"""Evaluate this student's answer:

QUESTION: {question}
STUDENT'S ANSWER: {student_answer}

EXPECTED KEY POINTS:
{chr(10).join(f'- {p}' for p in expected_points)}

STUDENT LEARNING CONTEXT:
- Mastered topics: {', '.join(learning_context['mastered_topics'])}
- Struggling topics: {', '.join(learning_context['struggling_topics'])}

Provide detailed evaluation as JSON:
{{
    "score": 85,
    "feedback": "Overall feedback...",
    "strong_points": ["point 1", "point 2"],
    "missed_points": ["point 1", "point 2"],
    "improvement_tips": ["tip 1", "tip 2"],
    "next_steps": ["What to study next"],
    "understanding_level": "excellent/good/needs_improvement"
}}

Be constructive and encouraging. If student is struggling, provide more specific guidance."""
        
        try:
            response = self.model.generate_content(prompt)
            evaluation = self._extract_json_object(response.text)
            
            # Update learning memory
            topic = self._detect_topic(question)
            rating = evaluation.get('score', 0) / 100
            self.learning_memory.add_interaction(question, student_answer, topic, rating)
            
            return evaluation if evaluation else {}
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return {}
    
    def _extract_json_object(self, text: str) -> Dict:
        """Extract JSON object from response"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {}
    
    def generate_personalized_study_path(self, time_available: int) -> Dict:
        """Generate personalized study plan based on learning state"""
        learning_context = self.learning_memory.get_learning_context()
        
        prompt = f"""Create a personalized study plan for {self.course_code}.

STUDENT LEARNING STATE:
- Mastered topics: {', '.join(learning_context['mastered_topics'])}
- Struggling topics: {', '.join(learning_context['struggling_topics'])}
- Study frequency: {learning_context['study_frequency']} sessions
- Recent questions: {', '.join(learning_context['recent_questions'][-5:])}

TIME AVAILABLE: {time_available} minutes

STUDY PLAN PRINCIPLES:
1. Prioritize struggling topics
2. Build on mastered topics to teach related concepts
3. Include variety: reading, practice, and review
4. Be realistic about time constraints
5. Include specific activities and time allocations

Format as JSON:
{{
    "total_time": {time_available},
    "sessions": [
        {{
            "topic": "topic_name",
            "duration": 15,
            "activity": "Read chapter X and take notes",
            "focus": "struggling/review/enrichment",
            "priority": "high/medium/low"
        }}
    ],
    "goals": ["goal 1", "goal 2"],
    "resources_needed": ["resource 1", "resource 2"]
}}"""
        
        try:
            response = self.model.generate_content(prompt)
            plan = self._extract_json_object(response.text)
            return plan if plan else {}
        except Exception as e:
            logger.error(f"Study plan error: {str(e)}")
            return {}


# Singleton tutor instances per student-course combination
_tutor_cache = {}

def get_smart_tutor(course_code: str, student_id: str) -> SmartAITutor:
    """Get or create a smart tutor instance"""
    cache_key = f"{course_code}_{student_id}"
    if cache_key not in _tutor_cache:
        _tutor_cache[cache_key] = SmartAITutor(course_code, student_id)
    return _tutor_cache[cache_key]
