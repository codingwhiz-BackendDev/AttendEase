"""
Ultimate Smart AI Teaching System - The Most Advanced Educational AI
Features:
- Advanced Math with SymPy (LaTeX rendering, step-by-step solutions)
- Visual/Diagram Generation (Mermaid, flowcharts)
- Multi-Strategy Teaching (Socratic, analogies, real-world)
- Interactive Problem Solving (guided walkthroughs)
- Deep Conceptual Understanding (first principles)
- Prerequisite Mapping
- Personalized Learning Paths
"""

import os
import json
import re
import logging
import sympy as sp
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from django.conf import settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Intelligent document chunking with semantic awareness"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into intelligent chunks preserving semantic meaning"""
        if not text or not text.strip():
            return []
        
        chunks = []
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_metadata = metadata.copy() if metadata else {}
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect mathematical content (preserve these as separate chunks)
            if self._is_math_content(para):
                chunks.append({
                    'text': para,
                    'metadata': {**current_metadata, 'type': 'math'}
                })
                continue
            
            # Detect code or technical content
            if self._is_code_content(para):
                chunks.append({
                    'text': para,
                    'metadata': {**current_metadata, 'type': 'code'}
                })
                continue
            
            # Regular text chunking
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': current_metadata.copy()
                    })
                
                if len(para) > self.chunk_size:
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
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': current_metadata.copy()
            })
        
        return chunks
    
    def _is_math_content(self, text: str) -> bool:
        """Detect if text contains mathematical content"""
        math_indicators = ['=', '≤', '≥', '≠', '∫', '∑', '∏', '√', '∞', '∂', '∇', '×', '÷', '²', '³']
        return any(indicator in text for indicator in math_indicators)
    
    def _is_code_content(self, text: str) -> bool:
        """Detect if text contains code"""
        code_indicators = ['def ', 'function', 'class ', 'import ', 'for(', 'while(', 'if(']
        return any(indicator in text for indicator in code_indicators)


class VectorStore:
    """FAISS-based vector store with hybrid search"""
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        # Fix meta tensor issue by explicitly setting device
        import torch
        self.device = 'cpu'  # Force CPU to avoid CUDA issues
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []
        self.metadata = []
        self.chunk_types = []  # Track math, code, regular text
    
    def add_chunks(self, chunks: List[Dict]):
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend([chunk['text'] for chunk in chunks])
        self.metadata.extend([chunk.get('metadata', {}) for chunk in chunks])
        self.chunk_types.extend([chunk.get('metadata', {}).get('type', 'text') for chunk in chunks])
    
    def search(self, query: str, top_k: int = 5, content_type: str = None) -> List[Tuple[str, Dict, float]]:
        """Search for relevant chunks with optional type filtering"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k * 2, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                # Filter by content type if specified
                if content_type and self.chunk_types[idx] != content_type:
                    continue
                results.append((self.chunks[idx], self.metadata[idx], float(score)))
        
        return results[:top_k]


class MathSolver:
    """Advanced mathematical problem solving with SymPy"""
    
    @staticmethod
    def solve_equation(equation_str: str) -> Dict:
        """Solve mathematical equations with step-by-step explanation"""
        try:
            # Clean the equation string
            equation_str = equation_str.replace('^', '**')
            
            # Try to parse and solve
            x = sp.symbols('x')
            
            # Handle different equation formats
            if '=' in equation_str:
                left, right = equation_str.split('=')
                eq = sp.Eq(sp.sympify(left), sp.sympify(right))
                solution = sp.solve(eq, x)
            else:
                # Assume it's an expression to simplify
                expr = sp.sympify(equation_str)
                solution = sp.simplify(expr)
            
            # Generate LaTeX
            latex_eq = sp.latex(eq if '=' in equation_str else expr)
            latex_sol = sp.latex(solution)
            
            # Generate step-by-step explanation
            steps = MathSolver._generate_steps(equation_str, solution)
            
            return {
                'success': True,
                'solution': str(solution),
                'latex_equation': latex_eq,
                'latex_solution': latex_sol,
                'steps': steps
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _generate_steps(equation_str: str, solution) -> List[str]:
        """Generate step-by-step solution explanation"""
        steps = []
        steps.append(f"Step 1: Identify the equation: {equation_str}")
        steps.append("Step 2: Isolate the variable")
        steps.append("Step 3: Apply inverse operations")
        steps.append(f"Step 4: The solution is: {solution}")
        return steps
    
    @staticmethod
    def explain_concept(concept: str) -> Dict:
        """Explain mathematical concept with visual aids"""
        concept_explanations = {
            'derivative': {
                'definition': 'The derivative measures the rate of change of a function',
                'formula': "f'(x) = lim(h→0) [f(x+h) - f(x)]/h",
                'latex': r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
                'real_world': 'Speed is the derivative of position with respect to time',
                'visual_aid': 'graph with tangent line showing slope'
            },
            'integral': {
                'definition': 'The integral represents the area under a curve',
                'formula': '∫f(x)dx = F(x) + C',
                'latex': r'\int f(x)dx = F(x) + C',
                'real_world': 'Distance traveled is the integral of velocity',
                'visual_aid': 'area under curve visualization'
            }
        }
        
        concept_lower = concept.lower()
        for key, value in concept_explanations.items():
            if key in concept_lower:
                return value
        
        return {
            'definition': f'Explanation for {concept} would be generated here',
            'formula': 'Formula would be generated here',
            'latex': '',
            'real_world': 'Real-world application would be here',
            'visual_aid': 'Visual aid description'
        }


class DiagramGenerator:
    """Generate visual diagrams and flowcharts"""
    
    @staticmethod
    def generate_flowchart(steps: List[str]) -> str:
        """Generate Mermaid flowchart from steps"""
        mermaid = "graph TD\n"
        for i, step in enumerate(steps):
            step_id = f"step{i}"
            mermaid += f"    {step_id}[\"{step}\"]\n"
            if i > 0:
                mermaid += f"    step{i-1} --> {step_id}\n"
        return mermaid
    
    @staticmethod
    def generate_concept_map(concept: str, related: List[str]) -> str:
        """Generate concept map diagram"""
        mermaid = f"graph LR\n"
        mermaid += f"    {concept}[\"{concept}\"]\n"
        for related_concept in related:
            mermaid += f"    {concept} --> {related_concept}\n"
        return mermaid
    
    @staticmethod
    def generate_process_diagram(process_name: str, stages: List[str]) -> str:
        """Generate process flow diagram"""
        mermaid = f"flowchart TD\n"
        mermaid += f"    Start([Start]) --> {stages[0]}\n"
        for i, stage in enumerate(stages):
            stage_id = f"stage{i}"
            mermaid += f"    {stage_id}[\"{stage}\"]\n"
            if i < len(stages) - 1:
                mermaid += f"    {stage_id} --> stage{i+1}\n"
        mermaid += f"    stage{len(stages)-1} --> End([End])\n"
        return mermaid


class PrerequisiteMapper:
    """Map and track prerequisite knowledge"""
    
    def __init__(self):
        self.prerequisite_graph = {
            'calculus': ['algebra', 'trigonometry'],
            'linear algebra': ['algebra', 'vectors'],
            'statistics': ['probability', 'algebra'],
            'differential equations': ['calculus', 'linear algebra'],
            'machine learning': ['statistics', 'linear algebra', 'calculus'],
            'data structures': ['programming', 'algorithms'],
            'algorithms': ['mathematics', 'logic']
        }
    
    def get_prerequisites(self, topic: str) -> List[str]:
        """Get prerequisites for a topic"""
        topic_lower = topic.lower()
        for key, prereqs in self.prerequisite_graph.items():
            if key in topic_lower:
                return prereqs
        return []
    
    def check_readiness(self, topic: str, mastered_topics: set) -> Dict:
        """Check if student is ready for a topic"""
        prereqs = self.get_prerequisites(topic)
        missing = [p for p in prereqs if p not in mastered_topics]
        
        return {
            'ready': len(missing) == 0,
            'prerequisites': prereqs,
            'missing': missing,
            'suggestion': f"Study {', '.join(missing)} first" if missing else "You're ready!"
        }


class TeachingStrategy:
    """Multiple teaching strategies for different learning styles"""
    
    @staticmethod
    def socratic_method(concept: str, context: str) -> str:
        """Use Socratic method - ask guiding questions"""
        return f"""Let's explore {concept} together through questions:

1. What do you think {concept} means based on what you've learned so far?
2. Can you think of a real-world example where {concept} might apply?
3. How does {concept} relate to what we discussed previously?
4. What would happen if we changed one aspect of {concept}?

Answer these step by step, and I'll guide you to the full understanding."""

    @staticmethod
    def analogy_based(concept: str, context: str) -> str:
        """Use analogies to explain complex concepts"""
        return f"""Let me explain {concept} using an analogy:

{context}

Think of it like this: [Detailed analogy here]

This analogy helps because:
- It captures the key property of {concept}
- It's something you're already familiar with
- It shows how {concept} works in practice

Now, how would you apply this to the actual {concept}?"""

    @staticmethod
    def first_principles(concept: str, context: str) -> str:
        """Break down to first principles"""
        return f"""Let's understand {concept} from first principles:

1. **Fundamental Truth**: What is the basic, undeniable fact about {concept}?
2. **Building Blocks**: What are the essential components?
3. **Logical Derivation**: How do we build up from the basics to the full concept?
4. **Applications**: How does this fundamental understanding apply?

{context}

This approach ensures you understand WHY {concept} works, not just WHAT it is."""

    @staticmethod
    def real_world_application(concept: str, context: str) -> str:
        """Focus on real-world applications"""
        return f"""Let's see how {concept} is used in the real world:

**Industry Applications:**
- [Specific industry examples]

**Everyday Examples:**
- [Daily life examples]

**Professional Use:**
- [How professionals use it]

{context}

Understanding these applications will help you see why {concept} matters beyond the classroom."""


class UltimateSmartTutor:
    """
    The Ultimate AI Tutor - Most Advanced Educational AI
    Combines all advanced features for exceptional teaching
    """
    
    def __init__(self, course_code: str, student_id: str):
        self.course_code = course_code
        self.student_id = student_id
        
        # Initialize all components
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()
        self.math_solver = MathSolver()
        self.diagram_generator = DiagramGenerator()
        self.prerequisite_mapper = PrerequisiteMapper()
        self.teaching_strategy = TeachingStrategy()
        
        # Learning memory
        self.topics_mastered = set()
        self.topics_struggling = set()
        self.conversation_history = []
        self.learning_style = "balanced"  # visual, auditory, kinesthetic, balanced
        
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
    
    def _detect_request_type(self, query: str) -> str:
        """Detect what type of request this is"""
        query_lower = query.lower()
        
        # Math detection
        if any(char in query for char in ['=', '+', '-', '*', '/', '^', '∫', '∑']):
            return 'math'
        
        # Diagram/visual request
        if any(word in query_lower for word in ['diagram', 'chart', 'graph', 'visual', 'flowchart', 'map']):
            return 'diagram'
        
        # Definition/explanation
        if any(word in query_lower for word in ['what is', 'define', 'explain', 'meaning']):
            return 'explanation'
        
        # Problem solving
        if any(word in query_lower for word in ['solve', 'how to', 'calculate', 'find']):
            return 'problem'
        
        # Conceptual understanding
        if any(word in query_lower for word in ['why', 'how does', 'relationship', 'connection']):
            return 'conceptual'
        
        return 'general'
    
    def _select_teaching_strategy(self, query: str, context: List[Dict]) -> str:
        """Select the best teaching strategy based on query and context"""
        query_lower = query.lower()
        
        # Use Socratic method for conceptual questions
        if any(word in query_lower for word in ['why', 'how does', 'understand']):
            return 'socratic'
        
        # Use analogies for complex/abstract concepts
        if any(word in query_lower for word in ['complex', 'abstract', 'difficult', 'confusing']):
            return 'analogy'
        
        # Use first principles for fundamental questions
        if any(word in query_lower for word in ['fundamental', 'basic', 'foundation', 'core']):
            return 'first_principles'
        
        # Use real-world applications for practical questions
        if any(word in query_lower for word in ['application', 'use', 'practice', 'real world']):
            return 'real_world'
        
        # Default to balanced approach
        return 'balanced'
    
    def _retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant document chunks"""
        request_type = self._detect_request_type(query)
        
        # Prioritize math chunks for math questions
        content_type = 'math' if request_type == 'math' else None
        
        results = self.vector_store.search(query, top_k, content_type)
        
        context = []
        for text, metadata, score in results:
            context.append({
                'text': text,
                'metadata': metadata,
                'relevance_score': score
            })
        
        return context
    
    def _build_ultimate_teaching_prompt(self, query: str, context: List[Dict], request_type: str, strategy: str) -> str:
        """Build the most sophisticated teaching prompt"""
        
        # Format retrieved context
        context_text = "\n\n---\n\n".join([
            f"[Source: {c['metadata'].get('title', 'Unknown')} - Type: {c['metadata'].get('type', 'text')} - Relevance: {c['relevance_score']:.2f}]\n{c['text']}"
            for c in context[:5]
        ])
        
        # Check prerequisites
        topic = self._detect_topic(query)
        readiness = self.prerequisite_mapper.check_readiness(topic, self.topics_mastered)
        
        # Learning state
        learning_state = f"""
STUDENT LEARNING STATE:
- Mastered topics: {', '.join(self.topics_mastered)}
- Struggling topics: {', '.join(self.topics_struggling)}
- Current topic readiness: {readiness['suggestion']}
- Learning style: {self.learning_style}
"""
        
        # Strategy-specific prompts
        strategy_prompts = {
            'socratic': "Use the Socratic method - ask guiding questions to help the student discover the answer themselves.",
            'analogy': "Use analogies and real-world comparisons to make complex concepts understandable.",
            'first_principles': "Break down the concept to first principles - start from fundamental truths and build up.",
            'real_world': "Focus on real-world applications and practical uses of this concept.",
            'balanced': "Use a balanced approach combining explanation, examples, and applications."
        }
        
        strategy_instruction = strategy_prompts.get(strategy, strategy_prompts['balanced'])
        
        prompt = f"""You are the world's most advanced AI tutor for {self.course_code}. Your goal is to ensure the student understands the material so deeply they don't need a lecturer.

TEACHING PHILOSOPHY:
1. Teach for DEEP UNDERSTANDING, not just memorization
2. Break down complex concepts into simple, digestible parts
3. Use multiple representations: text, math, diagrams, examples
4. Connect new concepts to what the student already knows
5. Check for understanding at each step
6. Adapt to the student's learning level
7. Be encouraging and supportive
8. Never say "I don't know" - always guide to resources

COURSE MATERIALS (most relevant):
{context_text if context_text else "No materials available yet"}

{learning_state}

PREREQUISITE CHECK: {readiness['suggestion']}

TEACHING STRATEGY: {strategy_instruction}

STUDENT'S QUESTION ({request_type.upper()}):
{query}

PROVIDE A COMPREHENSIVE RESPONSE THAT INCLUDES:

1. **Direct Answer**: Clear, concise answer to the question
2. **Mathematical Treatment** (if applicable):
   - Show formulas in LaTeX format: $\\[formula\\]$
   - Provide step-by-step derivation
   - Include worked examples
3. **Visual Representation** (if helpful):
   - Describe diagrams that would help understanding
   - Suggest flowcharts or concept maps
   - Use visual language (imagine, picture, visualize)
4. **Real-World Examples**: At least 2-3 concrete examples
5. **Connections**: Link to related concepts the student has mastered
6. **Practice Problem**: Create a practice problem to test understanding
7. **Follow-up Question**: Ask a question to check understanding
8. **Citations**: Reference which materials you used [From: Material Name]

ADDITIONAL REQUIREMENTS:
- If the topic is mathematical, show the derivation step-by-step
- If the concept is abstract, use multiple analogies
- If the student is struggling, provide more examples and simpler explanations
- If the student has mastered prerequisites, make connections to build deeper understanding
- Always explain the "WHY" not just the "WHAT"
- Use clear headings and structure

The response should be so comprehensive that the student truly understands and can apply the concept independently."""

        return prompt
    
    def _detect_topic(self, query: str) -> str:
        """Detect the main topic from a question"""
        words = query.lower().split()
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'explain', 'describe', 'calculate', 'solve', 'find'}
        topic_words = [w for w in words if w not in stop_words and len(w) > 3]
        return topic_words[0] if topic_words else "general"
    
    def teach(self, query: str) -> Dict:
        """
        Ultimate teaching method - the most sophisticated AI tutoring
        """
        # Detect request type
        request_type = self._detect_request_type(query)
        
        # Handle math problems specially
        if request_type == 'math':
            math_result = self.math_solver.solve_equation(query)
            if math_result['success']:
                return self._format_math_response(math_result, query)
        
        # Retrieve relevant context
        context = self._retrieve_relevant_context(query)
        
        # Select teaching strategy
        strategy = self._select_teaching_strategy(query, context)
        
        # Build ultimate teaching prompt
        prompt = self._build_ultimate_teaching_prompt(query, context, request_type, strategy)
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Post-process response
            answer = self._enhance_response(answer, request_type)
            
            # Extract citations
            citations = self._extract_citations(answer)
            
            # Detect topic
            topic = self._detect_topic(query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(context)
            
            # Update learning memory
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': answer,
                'topic': topic,
                'strategy': strategy,
                'request_type': request_type
            })
            
            return {
                'answer': answer,
                'citations': citations,
                'context_used': len(context),
                'confidence': confidence,
                'topic': topic,
                'strategy': strategy,
                'request_type': request_type,
                'learning_state': {
                    'mastered': list(self.topics_mastered),
                    'struggling': list(self.topics_struggling)
                }
            }
            
        except Exception as e:
            logger.error(f"Ultimate teaching error: {str(e)}")
            return {
                'error': str(e),
                'answer': "I'm experiencing technical difficulties. Please try again."
            }
    
    def _format_math_response(self, math_result: Dict, original_query: str) -> Dict:
        """Format mathematical solution response"""
        answer = f"""## Mathematical Solution

**Equation:** {original_query}

### Step-by-Step Solution:
"""
        for step in math_result['steps']:
            answer += f"{step}\n\n"
        
        answer += f"""
### LaTeX Representation:
**Equation:** ${math_result['latex_equation']}$
**Solution:** ${math_result['latex_solution']}$

### Final Answer:
{math_result['solution']}

### Practice:
Try solving a similar problem to reinforce your understanding. Would you like me to generate a practice problem?"""
        
        return {
            'answer': answer,
            'citations': [],
            'context_used': 0,
            'confidence': 0.95,
            'topic': 'mathematics',
            'strategy': 'analytical',
            'request_type': 'math',
            'math_data': math_result
        }
    
    def _enhance_response(self, answer: str, request_type: str) -> str:
        """Enhance the response with additional formatting"""
        # Ensure LaTeX is properly formatted
        answer = re.sub(r'\$(.*?)\$', r'$\1$', answer)
        
        # Add visual cues for different sections
        if request_type == 'explanation':
            answer = answer.replace('**Definition:**', '\n📚 **Definition:**')
            answer = answer.replace('**Example:**', '\n💡 **Example:**')
            answer = answer.replace('**Real-world:**', '\n🌍 **Real-world Application:**')
        
        return answer
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract material citations from response"""
        citations = re.findall(r'\[From: ([^\]]+)\]', response)
        return citations
    
    def _calculate_confidence(self, context: List[Dict]) -> float:
        """Calculate confidence based on context relevance"""
        if not context:
            return 0.4  # Moderate confidence without context
        
        avg_score = sum(c['relevance_score'] for c in context) / len(context)
        return min(avg_score, 1.0)
    
    def generate_interactive_practice(self, topic: str, difficulty: str = 'medium') -> Dict:
        """Generate interactive practice with guided hints"""
        prompt = f"""Create an interactive practice problem for {topic} in {self.course_code}.

DIFFICULTY: {difficulty}

Create a problem that:
1. Tests deep understanding, not just memorization
2. Requires multiple steps to solve
3. Has clear learning objectives
4. Can be solved with hints

Format as JSON:
{{
    "problem": "Problem statement",
    "learning_objectives": ["objective 1", "objective 2"],
    "hints": [
        "Hint 1 (gentle nudge)",
        "Hint 2 (more specific)",
        "Hint 3 (almost giving answer)"
    ],
    "solution": "Full solution with explanation",
    "difficulty": "easy/medium/hard",
    "time_estimate": "5-10 minutes"
}}"""
        
        try:
            response = self.model.generate_content(prompt)
            practice = self._extract_json_object(response.text)
            return practice if practice else {}
        except Exception as e:
            logger.error(f"Practice generation error: {str(e)}")
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
    
    def create_concept_diagram(self, concept: str) -> Dict:
        """Generate visual diagram for a concept"""
        try:
            from App.models import CourseMaterial
            materials = CourseMaterial.objects.filter(
                course__course_code=self.course_code
            )
            
            # Extract related concepts from materials
            related_concepts = []
            for material in materials:
                if material.content:
                    # Simple keyword extraction (could be improved with NLP)
                    words = re.findall(r'\b[A-Z][a-z]+\b', material.content)
                    related_concepts.extend([w for w in words if len(w) > 5 and w != concept])
            
            related_concepts = list(set(related_concepts))[:5]  # Top 5 related
            
            # Generate concept map
            mermaid = self.diagram_generator.generate_concept_map(concept, related_concepts)
            
            return {
                'success': True,
                'concept': concept,
                'related': related_concepts,
                'mermaid': mermaid,
                'type': 'concept_map'
            }
        except Exception as e:
            logger.error(f"Diagram generation error: {str(e)}")
            return {'success': False, 'error': str(e)}


# Singleton tutor instances
_tutor_cache = {}
_current_model = None

def get_ultimate_tutor(course_code: str, student_id: str) -> UltimateSmartTutor:
    """Get or create an ultimate smart tutor instance"""
    # Clear cache if model has changed
    global _current_model
    if _current_model != settings.GEMINI_MODEL:
        _tutor_cache.clear()
        _current_model = settings.GEMINI_MODEL
    
    cache_key = f"{course_code}_{student_id}"
    if cache_key not in _tutor_cache:
        _tutor_cache[cache_key] = UltimateSmartTutor(course_code, student_id)
    
    return _tutor_cache[cache_key]
