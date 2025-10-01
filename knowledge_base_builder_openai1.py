"""
GLLAVA-Based Geometric Knowledge Base Builder - Part 1 Implementation
Uses image-to-text + LLM to solve GLLAVA geometric problems and extract theorems for knowledge base
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import base64
from io import BytesIO
import os
import time

# Core dependencies
import neo4j
from neo4j import GraphDatabase
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Image processing
from PIL import Image

# Data handling
import pandas as pd
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometricKnowledgeBaseManager:
    """High-level manager for generic geometric knowledge base construction"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.kb = GeometricKnowledgeBase(
            neo4j_uri=config['neo4j_uri'],
            neo4j_user=config['neo4j_user'],
            neo4j_password=config['neo4j_password'],
            openai_api_key=config['openai_api_key']
        )
    
    def build_knowledge_base_from_dataset(self, dataset_path: str, max_problems: Optional[int] = None,
                                        continuous_improvement: bool = True) -> Dict[str, Any]:
        """Complete pipeline to build generic knowledge base from dataset"""
        
        logger.info("Starting generic geometric knowledge base construction")
        
        # Process dataset
        processing_results = self.kb.build_knowledge_from_dataset(dataset_path, max_problems)
        
        # Continuous improvement
        # if continuous_improvement and processing_results['successful_extractions'] > 0:
        #     logger.info("Running continuous improvement process")
        #     self.kb.continuously_improve_theorems()
        
        # Generate final report
        final_stats = self.kb._generate_knowledge_base_stats()
        
        results = {
            'processing_results': processing_results,
            'final_statistics': final_stats,
            'knowledge_base_ready': True
        }
        
        logger.info(f"Generic knowledge base construction completed:")
        logger.info(f"- Problems processed: {processing_results['total_problems_processed']}")
        logger.info(f"- Success rate: {processing_results['successful_extractions']}/{processing_results['total_problems_processed']}")
        logger.info(f"- Theorems discovered: {final_stats['total_theorems']}")
        logger.info(f"- Shapes identified: {final_stats['total_shapes']}")
        logger.info(f"- Relationships created: {final_stats['shape_theorem_relationships']}")
        
        return results
    
    def validate_knowledge_base(self) -> Dict[str, Any]:
        """Validate the built knowledge base"""
        
        validation_results = {
            'theorem_validation': [],
            'shape_validation': [],
            'coverage_analysis': {},
            'quality_metrics': {}
        }
        
        with self.kb.driver.session() as session:
            # Validate theorems have sufficient information
            theorems = session.run("""
                MATCH (t:Theorem)
                RETURN t.name as name, t.description as description,
                       t.usage_count as usage_count, t.mathematical_form as mathematical_form,
                       size(t.conditions) as condition_count
            """).values()
            
            for theorem_name, description, usage_count, mathematical_form, condition_count in theorems:
                validation = {
                    'theorem_name': theorem_name,
                    'has_description': bool(description and len(description) > 10),
                    'has_mathematical_form': bool(mathematical_form),
                    'has_conditions': condition_count > 0,
                    'sufficient_usage': usage_count >= 1,
                    'overall_quality': 'good' if (description and mathematical_form and condition_count > 0) else 'needs_improvement'
                }
                validation_results['theorem_validation'].append(validation)
            
            # Validate shapes
            shapes = session.run("""
                MATCH (s:Shape)
                OPTIONAL MATCH (s)-[:APPLICABLE_TO]->(t:Theorem)
                RETURN s.name as name, s.occurrence_count as occurrence_count,
                       count(t) as related_theorems
            """).values()
            
            for shape_name, occurrence_count, related_theorems in shapes:
                validation = {
                    'shape_name': shape_name,
                    'sufficient_occurrences': occurrence_count >= 1,
                    'has_related_theorems': related_theorems > 0,
                    'overall_quality': 'good' if (occurrence_count >= 1 and related_theorems > 0) else 'needs_improvement'
                }
                validation_results['shape_validation'].append(validation)
            
            # Coverage analysis
            shape_types = session.run("""
                MATCH (s:Shape)
                RETURN s.shape_type as type, count(s) as count
                ORDER BY count DESC
            """).values()
            
            validation_results['coverage_analysis'] = {
                'shape_types_covered': len(shape_types),
                'shape_type_distribution': dict(shape_types)
            }
            
            # Quality metrics
            validation_results['quality_metrics'] = {
                'theorems_with_good_quality': len([t for t in validation_results['theorem_validation'] if t['overall_quality'] == 'good']),
                'total_theorems': len(validation_results['theorem_validation']),
                'shapes_with_good_quality': len([s for s in validation_results['shape_validation'] if s['overall_quality'] == 'good']),
                'total_shapes': len(validation_results['shape_validation'])
            }
        
        return validation_results
    
    def search_knowledge_base(self, query: str, search_type: str = 'theorem') -> List[Dict]:
        """Search the knowledge base"""
        
        if search_type == 'theorem':
            return self.kb.query_theorems_by_similarity(query)
        elif search_type == 'shape':
            with self.kb.driver.session() as session:
                results = session.run("""
                    MATCH (s:Shape)
                    WHERE s.name CONTAINS $query OR s.shape_type CONTAINS $query
                    RETURN s.name as shape_name, s.shape_type as shape_type,
                           s.occurrence_count as occurrence_count
                    ORDER BY s.occurrence_count DESC
                    LIMIT 10
                """, {'query': query.lower()})
                
                return [dict(record) for record in results]
        
        return []
    
    def get_shape_theorems(self, shape_name: str) -> List[Dict]:
        """Get all theorems applicable to a shape"""
        return self.kb.get_theorems_for_shape(shape_name)
    
    def get_theorem_shapes(self, theorem_name: str) -> List[Dict]:
        """Get all shapes that a theorem applies to"""
        return self.kb.get_shapes_for_theorem(theorem_name)
    
    def close(self):
        """Close knowledge base"""
        self.kb.close()

@dataclass
class GeometricTheorem:
    """Represents a geometric theorem extracted from problem solutions"""
    name: str
    description: str
    conditions: List[str]
    conclusions: List[str]
    applicable_shapes: List[str]
    mathematical_form: str = ""
    confidence: float = 1.0
    usage_count: int = 0
    success_rate: float = 1.0
    application_examples: List[str] = None  # Store example applications, not full problems
    
    def __post_init__(self):
        if self.application_examples is None:
            self.application_examples = []
        self.theorem_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique theorem ID"""
        content = f"{self.name}_{self.description}_{str(self.conditions)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class GeometricShape:
    """Represents a geometric shape type in the knowledge base"""
    name: str
    shape_type: str  # triangle, circle, quadrilateral, line, point
    properties: List[str]
    constraints: List[str]
    related_theorems: List[str] = None
    occurrence_count: int = 0
    
    def __post_init__(self):
        if self.related_theorems is None:
            self.related_theorems = []

# Remove GLLAVAProblem dataclass as we don't store problem details

class OpenAIImageToTextProcessor:
    """Processes geometric images using OpenAI Vision API"""
    
    def __init__(self, openai_client, model_name: str = "gpt-4o"):
        """Initialize OpenAI Vision processor"""
        self.openai_client = openai_client
        self.model_name = model_name
        
        # Validate model name
        valid_models = ["gpt-4-vision-preview", "gpt-4o", "gpt-4o-mini"]
        if model_name not in valid_models:
            logger.warning(f"Model {model_name} not in recommended list: {valid_models}")
            logger.warning("Using gpt-4o as default")
            self.model_name = "gpt-4o"
        
        logger.info(f"Initialized OpenAI Vision processor with model: {self.model_name}")
    
    def describe_geometric_image(self, image_path: str, question: str = "") -> Dict[str, str]:
        """Generate detailed description of geometric image using OpenAI Vision API"""
        
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image_path)
            
            # Generate general description
            general_description = self._analyze_with_openai_vision(
                image_base64,
                "Analyze this geometric diagram in detail. Describe all visible shapes, lines, points, angles, and any measurements or labels you can see."
            )
            
            # Generate shape-focused description
            shape_description = self._analyze_with_openai_vision(
                image_base64,
                "Focus specifically on identifying all geometric shapes in this image. List each shape type (triangle, circle, rectangle, etc.), count how many of each type, and describe their properties and relationships to each other."
            )
            
            # Generate measurement-focused description
            measurement_description = self._analyze_with_openai_vision(
                image_base64,
                "Identify and list all numerical values, measurements, angles, lengths, or any other quantitative information shown in this geometric diagram. Include units if visible."
            )
            
            # Generate question-specific description if question provided
            question_specific = ""
            if question:
                question_specific = self._analyze_with_openai_vision(
                    image_base64,
                    f"Given this geometric diagram, analyze what information is available to answer this question: '{question}'. What geometric elements, measurements, or relationships are relevant?"
                )
            
            return {
                'general_description': general_description,
                'shape_description': shape_description,
                'measurement_description': measurement_description,
                'question_specific_description': question_specific,
                'combined_description': self._combine_descriptions(
                    general_description, shape_description, measurement_description, question_specific
                )
            }
            
        except Exception as e:
            logger.error(f"Error describing image {image_path}: {e}")
            return {
                'general_description': f"Error processing image: {str(e)}",
                'shape_description': "",
                'measurement_description': "",
                'question_specific_description': "",
                'combined_description': f"Error processing image: {str(e)}"
            }
    
    def _analyze_with_openai_vision(self, image_base64: str, prompt: str) -> str:
        """Analyze image with OpenAI Vision API using specific prompt"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"  # Use high detail for better geometric analysis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent, factual descriptions
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            return f"Vision analysis failed: {str(e)}"
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string for OpenAI API"""
        try:
            # Load and process image
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize if image is too large (OpenAI has size limits)
                max_size = 2048
                if max(image.size) > max_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=85, optimize=True)
                
                return base64.b64encode(buffer.getvalue()).decode()
                
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def batch_analyze_images(self, image_paths: List[str], questions: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Analyze multiple images in batch with rate limiting"""
        
        results = []
        questions = questions or [""] * len(image_paths)
        
        for i, (image_path, question) in enumerate(zip(image_paths, questions)):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                result = self.describe_geometric_image(image_path, question)
                result['image_path'] = image_path
                result['batch_index'] = i
                results.append(result)
                
                # Rate limiting to respect OpenAI API limits
                if i < len(image_paths) - 1:  # Don't sleep after the last image
                    time.sleep(0.5)  # 2 requests per second to be conservative
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'batch_index': i,
                    'general_description': f"Error: {str(e)}",
                    'shape_description': "",
                    'measurement_description': "",
                    'question_specific_description': "",
                    'combined_description': f"Error: {str(e)}"
                })
        
        return results
    
    def _combine_descriptions(self, general: str, shapes: str, measurements: str, question_specific: str) -> str:
        """Combine all descriptions into comprehensive description"""
        
        parts = []
        
        if general and general != "Unable to generate description":
            parts.append(f"General: {general}")
        
        if shapes:
            parts.append(f"Shapes and Elements: {shapes}")
        
        if measurements:
            parts.append(f"Measurements: {measurements}")
        
        if question_specific:
            parts.append(f"Question Context: {question_specific}")
        
        return " | ".join(parts)

class GeometricKnowledgeExtractor:
    """Extracts and processes geometric knowledge from problem solutions"""
    
    def __init__(self, openai_client, image_processor: OpenAIImageToTextProcessor):
        self.openai_client = openai_client
        self.image_processor = image_processor
    
    def process_geometric_problem(self, image_path: str, question: str, 
                                answer: Optional[str] = None) -> Dict[str, Any]:
        """Process a geometric problem to extract reusable knowledge"""
        
        logger.info(f"Processing geometric problem from {os.path.basename(image_path)}")
        
        # Step 1: Analyze image
        image_descriptions = self.image_processor.describe_geometric_image(image_path, question)
        logger.info(f"Image Description: {image_descriptions}")
        # Step 2: Solve problem
        solution_result = self._solve_with_llm(question, image_descriptions['combined_description'], answer)
        logger.info(f"Solution Result : {solution_result}")
        # Step 3: Extract geometric knowledge (shapes and theorems only)
        knowledge_extraction = self._extract_geometric_knowledge(
            question, image_descriptions, solution_result
        )
        logger.info(f"Extracted knowledge Result : {knowledge_extraction}")
        return {
            'success': solution_result.get('success', False),
            'extracted_shapes': knowledge_extraction.get('shapes_identified', []),
            'extracted_theorems': knowledge_extraction.get('theorems_used', []),
            'geometric_relationships': knowledge_extraction.get('geometric_relationships', []),
            'confidence': solution_result.get('confidence', 0.0)
        }
    
    def _solve_with_llm(self, question: str, image_description: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Solve geometric problem using LLM"""
        
        # Create comprehensive prompt
        prompt = f"""
        You are an expert geometry tutor solving a geometric problem step by step.

        IMAGE DESCRIPTION:
        {image_description}

        QUESTION:
        {question}

        TASK:
        1. Analyze the geometric elements described in the image
        2. Identify what information is given and what needs to be found
        3. Determine which geometric theorems, formulas, or principles apply
        4. Solve the problem step by step with clear mathematical reasoning
        5. Show all calculations and intermediate steps
        6. Verify your answer makes sense

        Provide your response in this JSON format:
        {{
            "analysis": "what you observe from the image description",
            "given_information": ["list of given facts"],
            "target": "what needs to be found",
            "applicable_theorems": ["theorems/formulas that apply"],
            "solution_steps": [
                {{
                    "step_number": 1,
                    "description": "what you're doing in this step",
                    "theorem_used": "name of theorem/formula used",
                    "calculation": "mathematical work",
                    "result": "result of this step"
                }}
            ],
            "final_answer": "the final numerical or descriptive answer",
            "verification": "check if answer is reasonable",
            "confidence": 0.9
        }}

        Be precise with mathematical calculations and geometric reasoning.
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert geometry problem solver. Provide detailed step-by-step solutions in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                solution_data = json.loads(json_match.group())
            else:
                solution_data = json.loads(result_text)
            
            # Verify solution if ground truth available
            if ground_truth:
                verification_result = self._verify_solution(solution_data, ground_truth)
                solution_data['ground_truth_verification'] = verification_result
            
            solution_data['success'] = True
            solution_data['solution'] = solution_data.get('final_answer', '')
            
            return solution_data
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                'success': False,
                'error': str(e),
                'solution': '',
                'confidence': 0.0
            }
    
    def _verify_solution(self, llm_solution: Dict, ground_truth: str) -> Dict[str, Any]:
        """Verify LLM solution against ground truth"""
        
        prompt = f"""
        Compare the LLM solution with the ground truth answer:

        LLM SOLUTION: {llm_solution.get('final_answer', '')}
        GROUND TRUTH: {ground_truth}

        Determine:
        1. Are they equivalent (considering different valid forms)?
        2. What's the accuracy level?
        3. Are there any errors in the LLM reasoning?

        Return JSON:
        {{
            "match": true/false,
            "accuracy_score": 0.0-1.0,
            "explanation": "why they match/don't match",
            "errors_identified": ["list of any errors"]
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Compare geometry solutions objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(result_text)
                
        except Exception as e:
            logger.error(f"Error verifying solution: {e}")
            return {'match': False, 'accuracy_score': 0.0, 'explanation': 'Verification failed'}
    
    def _extract_geometric_knowledge(self, question: str, image_descriptions: Dict, solution_result: Dict) -> Dict[str, Any]:
        """Extract geometric knowledge from problem and solution"""
        
        prompt = f"""
        Analyze this solved geometry problem to extract reusable geometric knowledge.

        QUESTION: {question}
        IMAGE DESCRIPTION: {image_descriptions.get('combined_description', '')}
        SOLUTION: {json.dumps(solution_result, indent=2)}

        Extract:
        1. What geometric shapes/elements were involved?
        2. Which theorems, formulas, or principles were applied?
        3. What general patterns or relationships were used?
        4. What conditions must be met to apply these theorems?

        Return JSON:
        {{
            "shapes_identified": ["triangle", "circle", "line", etc.],
            "theorems_used": [
                {{
                    "name": "theorem name",
                    "description": "what the theorem states",
                    "conditions": ["when it can be applied"],
                    "mathematical_form": "formula if applicable",
                    "application_context": "how it was used in this problem",
                    "applicable_shapes":["triangle", "circle", "line", etc.]
                }}
            ],
            "geometric_relationships": ["relationships discovered"],
            "problem_type": "classification of problem type",
            "key_insights": ["important insights for future problems"]
        }}
        
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract geometric knowledge patterns from solved problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(result_text)
                
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
            return {'shapes_identified': [], 'theorems_used': [], 'geometric_relationships': []}

class GeometricKnowledgeBase:
    """Generic geometric knowledge base storing shapes and theorems"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        """Initialize the knowledge base"""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize knowledge extractor
        self.image_processor = OpenAIImageToTextProcessor(self.openai_client)
        self.knowledge_extractor = GeometricKnowledgeExtractor(self.openai_client, self.image_processor)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize Neo4j database schema for generic knowledge base"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT theorem_id IF NOT EXISTS FOR (t:Theorem) REQUIRE t.theorem_id IS UNIQUE",
                "CREATE CONSTRAINT shape_name IF NOT EXISTS FOR (s:Shape) REQUIRE s.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation failed: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX theorem_name_idx IF NOT EXISTS FOR (t:Theorem) ON (t.name)",
                "CREATE INDEX shape_type_idx IF NOT EXISTS FOR (s:Shape) ON (s.shape_type)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
    
    def build_knowledge_from_dataset(self, dataset_path: str, max_problems: Optional[int] = None) -> Dict[str, Any]:
        """Build generic knowledge base from geometric problem dataset"""
        
        logger.info(f"Building generic knowledge base from: {dataset_path}")
        
        # Load dataset
        # problems = self._load_dataset(dataset_path)
        
        # if max_problems:
        #     problems = problems[:max_problems]
        
        # logger.info(f"Processing {len(problems)} problems for knowledge extraction")
        
        # Process problems to extract knowledge
        results = {
            'total_problems_processed': 1,
            'successful_extractions': 0,
            'unique_theorems_extracted': 0,
            'unique_shapes_identified': 0,
            'failed_extractions': []
        }
        
        extracted_theorems = set()
        extracted_shapes = set()
        problem_data = {}
        problem_data['image'] = "images/geo3k/train/2/img_diagram.png"
        problem_data['question'] = """
        Question: Find m \\angle C.
        Choices: [
            "112",
    "116",
    "117",
    "127"
        ]
        """
        problem_data['answer'] = "D"
        # for i, problem_data in enumerate(problems):
        try:
                # logger.info(f"Processing problem {i+1}/{len(problems)}")
                
                # Extract knowledge without storing problem details
                extraction_result = self.knowledge_extractor.process_geometric_problem(
                    image_path=problem_data['image'],
                    question=problem_data['question'],
                    answer=problem_data.get('answer')
                )
                # extraction_result = {'success': True, 'extracted_shapes': ['Parallelogram', 'Triangle'], 'extracted_theorems': [{'name': 'Opposite sides of a parallelogram are equal',"applicable_shapes":["Parallelogram"], 'description': 'In a parallelogram, the opposite sides are equal in length.', 'conditions': ['The shape must be a parallelogram'], 'mathematical_form': 'If ABCD is a parallelogram, then AB = DC and BC = AD', 'application_context': 'This theorem was used to equate the lengths of sides MR and NP, which allowed for the creation of an equation to solve for y.'}], 'geometric_relationships': ['The diagonals of a parallelogram bisect each other', 'The sum of the angles in a triangle is 180 degrees', 'The angles around a point sum to 360 degrees'], 'confidence': 0.9}
                logger.info(f"Extracted knowledge: {extraction_result}")
                if extraction_result['success']:
                    # Store extracted knowledge in KB
                    self._store_extracted_knowledge(extraction_result)
                    results['successful_extractions'] += 1
                    
                    # Track unique theorems and shapes
                    for theorem in extraction_result['extracted_theorems']:
                        extracted_theorems.add(theorem['name'])
                    
                    for shape in extraction_result['extracted_shapes']:
                        extracted_shapes.add(shape)
                else:
                    results['failed_extractions'].append({
                        'problem_index': 1,
                        'image_path': problem_data['image']
                    })
                
        except Exception as e:
                logger.error(f"Error processing problem 1: {e}")
                logger.error(e.with_traceback)
                results['failed_extractions'].append({
                    'problem_index': 1,
                    'error': str(e)
                })
        
        results['unique_theorems_extracted'] = len(extracted_theorems)
        results['unique_shapes_identified'] = len(extracted_shapes)
        
        # Generate final statistics
        results['knowledge_base_stats'] = self._generate_knowledge_base_stats()
        
        logger.info(f"Knowledge base construction completed:")
        logger.info(f"- Success rate: {results['successful_extractions']}/{results['total_problems_processed']}")
        logger.info(f"- Unique theorems: {results['unique_theorems_extracted']}")
        logger.info(f"- Unique shapes: {results['unique_shapes_identified']}")
        
        return results
    
    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load geometric problem dataset (GLLAVA or other formats)"""
        
        problems = []
        
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    problems = data
                else:
                    problems = data.get('data', data.get('problems', []))
        
        elif dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        problems.append(json.loads(line.strip()))
        
        # Validate that problems have required fields
        valid_problems = []
        for problem in problems:
            if 'image' in problem and 'question' in problem:
                # Ensure image path is accessible
                image_path = problem['image']
                if not os.path.isabs(image_path):
                    dataset_dir = os.path.dirname(dataset_path)
                    image_path = os.path.join(dataset_dir, image_path)
                
                if os.path.exists(image_path):
                    problem['image'] = image_path
                    valid_problems.append(problem)
                else:
                    logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Loaded {len(valid_problems)} valid problems from {len(problems)} total")
        return valid_problems
    
    def _store_extracted_knowledge(self, extraction_result: Dict):
        """Store only extracted shapes and theorems (no problem details)"""
        
        with self.driver.session() as session:
            # Store extracted theorems
            for theorem_data in extraction_result['extracted_theorems']:
                logger.info(f"Storing theorem: {theorem_data['name']}")
                self._store_or_update_theorem(session, theorem_data)
            
            # Store extracted shapes
            for shape in extraction_result['extracted_shapes']:
                shape_name = shape['name']
                properties = shape['properties']
                logger.info(f"Storing shape: {shape_name}")
                self._store_or_update_shape(session, shape_name, properties)
            
            # Create relationships between shapes and theorems
            for theorem_data in extraction_result['extracted_theorems']:
                for shape in theorem_data.get('applicable_shapes', []):
                    self._link_shape_to_theorem(session, shape, theorem_data['name'])
    
    def _store_or_update_theorem(self, session, theorem_data: Dict):
        """Store new theorem or update existing one"""
        
        theorem_name = theorem_data['name']
        
        # Check if theorem exists
        existing = session.run(
            "MATCH (t:Theorem {name: $name}) RETURN t.theorem_id as id",
            {'name': theorem_name}
        ).single()
        
        if existing:
            
            logger.info(f"Theorem already exists: {theorem_name}")
        else:
            # Create new theorem
            theorem = GeometricTheorem(
                name=theorem_name,
                description=theorem_data.get('description', ''),
                conditions=theorem_data.get('conditions', []),
                conclusions=theorem_data.get('conclusions', []),
                applicable_shapes=theorem_data.get('applicable_shapes', []),
                mathematical_form=theorem_data.get('mathematical_form', ''),
                application_examples=[theorem_data.get('application_context', '')],
                usage_count=1
            )
            
            self._store_new_theorem(session, theorem)
            logger.info(f"Added new theorem: {theorem_name}")
    
    def _store_or_update_shape(self, session, shape_name: str, properties = []):
        """Store new shape or update existing one"""
        
        # Check if shape exists
        existing = session.run(
            "MATCH (s:Shape {name: $name}) Return s.name",
            {'name': shape_name.lower()}
        ).single()
        
        if existing:
            # Update occurrence count
            logger.info(f"Shape already exists: {shape_name}")
        else:
            # Create new shape
            session.run("""
                CREATE (s:Shape {
                    name: $name,
                    shape_type: $shape_type,
                    properties: $properties,
                    constraints: $constraints,
                    occurrence_count: 1,
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """, {
                'name': shape_name.lower(),
                'shape_type': self._classify_shape_type(shape_name),
                'properties': properties.extend(self._get_default_shape_properties(shape_name)),
                'constraints': self._get_default_shape_constraints(shape_name)
            })
            logger.info(f"Added new shape: {shape_name}")
    
    def _link_shape_to_theorem(self, session, shape_name: str, theorem_name: str):
        """Create relationship between shape and theorem"""
        session.run("""
            MATCH (s:Shape {name: $shape_name})
            MATCH (t:Theorem {name: $theorem_name})
            MERGE (s)-[:APPLICABLE_TO]->(t)
        """, {'shape_name': shape_name.lower(), 'theorem_name': theorem_name})
    
    def _classify_shape_type(self, shape_name: str) -> str:
        """Classify shape type from name"""
        shape_name = shape_name.lower()
        if 'triangle' in shape_name:
            return 'triangle'
        elif 'circle' in shape_name:
            return 'circle'
        elif 'square' in shape_name or 'rectangle' in shape_name or 'parallelogram' in shape_name or 'trapezoid' in shape_name:
            return 'quadrilateral'
        elif 'line' in shape_name:
            return 'line'
        elif 'point' in shape_name:
            return 'point'
        else:
            return 'polygon'
    
    def _get_default_shape_properties(self, shape_name: str) -> List[str]:
        """Get default properties for shape type"""
        shape_type = self._classify_shape_type(shape_name)
        
        property_map = {
            'triangle': ['3 sides', '3 angles', '3 vertices', 'angle sum = 180°'],
            'circle': ['curved boundary', 'constant radius', 'center point', 'area = πr²'],
            'quadrilateral': ['4 sides', '4 angles', '4 vertices', 'angle sum = 360°'],
            'line': ['infinite length', 'no width', 'straight'],
            'point': ['no dimension', 'position only']
        }
        
        return property_map.get(shape_type, ['geometric object'])
    
    def _get_default_shape_constraints(self, shape_name: str) -> List[str]:
        """Get default constraints for shape type"""
        shape_type = self._classify_shape_type(shape_name)
        
        constraint_map = {
            'triangle': ['sum of angles = 180°', 'triangle inequality'],
            'circle': ['all points equidistant from center'],
            'quadrilateral': ['sum of angles = 360°'],
            'line': ['contains infinite points'],
            'point': ['zero-dimensional']
        }
        
        return constraint_map.get(shape_type, [])
    
    def _store_new_theorem(self, session, theorem: GeometricTheorem):
        """Store a new theorem in the database"""
        
        query = """
        CREATE (t:Theorem {
            theorem_id: $theorem_id,
            name: $name,
            description: $description,
            conditions: $conditions,
            conclusions: $conclusions,
            applicable_shapes: $applicable_shapes,
            mathematical_form: $mathematical_form,
            confidence: $confidence,
            usage_count: $usage_count,
            success_rate: $success_rate,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN t
        """
        
        session.run(query, {
            'theorem_id': theorem.theorem_id,
            'name': theorem.name,
            'description': theorem.description,
            'conditions': theorem.conditions,
            'conclusions': theorem.conclusions,
            'applicable_shapes': theorem.applicable_shapes,
            'mathematical_form': theorem.mathematical_form,
            'confidence': theorem.confidence,
            'usage_count': theorem.usage_count,
            'success_rate': theorem.success_rate
        })
        
        # Store embedding
        theorem_text = f"{theorem.name}: {theorem.description}"
        # embedding = self.embedder.encode([theorem_text])[0].tolist()
        # self._store_embedding(session, theorem.theorem_id, embedding, "theorem")
    
    def _store_shape(self, session, shape_name: str):
        """Store geometric shape"""
        session.run("""
            MERGE (s:Shape {name: $name})
            SET s.updated_at = datetime()
        """, {'name': shape_name.lower()})
    
    
    def _store_embedding(self, session, entity_id: str, embedding: List[float], entity_type: str):
        """Store embeddings for similarity search"""
        logger.info(f"Storing Embedding for the theorem {entity_id}")
        session.run("""
            MERGE (e:Embedding {entity_id: $entity_id, entity_type: $entity_type})
            SET e.vector = $embedding,
                e.updated_at = datetime()
        """, {'entity_id': entity_id, 'entity_type': entity_type, 'embedding': embedding})
    
    def _generate_knowledge_base_stats(self) -> Dict[str, Any]:
        """Generate statistics about the knowledge base"""
        
        with self.driver.session() as session:
            # Count theorems
            theorem_count = session.run("MATCH (t:Theorem) RETURN count(t) as count").single()['count']
            
            # Count shapes
            shape_count = session.run("MATCH (s:Shape) RETURN count(s) as count").single()['count']
            
            # Most used theorems
            top_theorems = session.run("""
                MATCH (t:Theorem)
                RETURN t.name as theorem, t.usage_count as usage
                ORDER BY t.usage_count DESC
                LIMIT 10
            """).values()
            
            relations = session.run("MATCH p=()-[:APPLICABLE_TO]->() RETURN count(p) as count").single()['count']
           
            return {
                'total_theorems': theorem_count,
                'total_shapes': shape_count,
                'top_theorems': top_theorems,
                'shape_theorem_relationships':relations
            }
    
    def query_theorems_by_similarity(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Find similar theorems using semantic similarity"""
        
        query_embedding = self.embedder.encode([query_text])[0]
        
        with self.driver.session() as session:
            embeddings_result = session.run("""
                MATCH (e:Embedding {entity_type: 'theorem'})
                MATCH (t:Theorem {theorem_id: e.entity_id})
                RETURN t.theorem_id as id, t.name as name, t.description as description,
                       t.usage_count as usage_count, e.vector as embedding
            """)
            
            similarities = []
            for record in embeddings_result:
                theorem_embedding = np.array(record['embedding'])
                similarity = cosine_similarity([query_embedding], [theorem_embedding])[0][0]
                similarities.append({
                    'theorem_id': record['id'],
                    'name': record['name'],
                    'description': record['description'],
                    'usage_count': record['usage_count'],
                    'similarity': similarity
                })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
    
    def get_theorem_application_examples(self, theorem_name: str) -> List[Dict]:
        """Get examples of how a theorem was applied in solved problems"""
        
        with self.driver.session() as session:
            results = session.run("""
                MATCH (t:Theorem {name: $theorem_name})-[:USED_BY]-(p:GLLAVAProblem)
                RETURN p.problem_id as problem_id, p.question as question,
                       p.llm_solution as solution, p.confidence_score as confidence
                ORDER BY p.confidence_score DESC
                LIMIT 10
            """, {'theorem_name': theorem_name})
            
            return [dict(record) for record in results]
    
    def continuously_improve_theorems(self):
        """Continuously improve theorem descriptions based on usage patterns"""
        
        logger.info("Starting theorem improvement process...")
        
        with self.driver.session() as session:
            # Get theorems with multiple usage examples
            theorems_to_improve = session.run("""
                MATCH (t:Theorem)
                WHERE t.usage_count >= 3
                RETURN t.theorem_id as theorem_id, t.name as name, 
                       t.description as description, t.source_problems as source_problems
            """).values()
            
            for theorem_record in theorems_to_improve:
                try:
                    self._improve_single_theorem(session, theorem_record)
                except Exception as e:
                    logger.error(f"Error improving theorem {theorem_record[1]}: {e}")
    
    def _improve_single_theorem(self, session, theorem_record):
        """Improve a single theorem based on its usage examples"""
        
        theorem_id, theorem_name, current_description, source_problems = theorem_record
        
        # Get usage examples
        examples = []
        for problem_id in source_problems[:5]:  # Limit to 5 examples
            problem_data = session.run("""
                MATCH (p:GLLAVAProblem {problem_id: $problem_id})
                RETURN p.question as question, p.llm_solution as solution
            """, {'problem_id': problem_id}).single()
            
            if problem_data:
                examples.append({
                    'question': problem_data['question'],
                    'solution': problem_data['solution']
                })
        
        if not examples:
            return
        
        # Use LLM to improve theorem description
        improved_description = self._generate_improved_theorem_description(
            theorem_name, current_description, examples
        )
        
        if improved_description and improved_description != current_description:
            # Update theorem
            session.run("""
                MATCH (t:Theorem {theorem_id: $theorem_id})
                SET t.description = $new_description,
                    t.improved_at = datetime()
            """, {
                'theorem_id': theorem_id,
                'new_description': improved_description
            })
            
            logger.info(f"Improved theorem: {theorem_name}")
    
    def _generate_improved_theorem_description(self, theorem_name: str, current_description: str, examples: List[Dict]) -> str:
        """Generate improved theorem description using LLM"""
        
        examples_text = []
        for i, example in enumerate(examples, 1):
            examples_text.append(f"Example {i}:\nQuestion: {example['question']}\nSolution: {example['solution'][:200]}...")
        
        prompt = f"""
        Improve the description of this geometric theorem based on how it's been used in practice.

        THEOREM NAME: {theorem_name}
        CURRENT DESCRIPTION: {current_description}

        USAGE EXAMPLES:
        {chr(10).join(examples_text)}

        Create an improved description that:
        1. Is more precise and clear
        2. Includes key conditions for application
        3. Reflects how it's actually used in practice
        4. Maintains mathematical accuracy
        5. Is educational and helpful

        Return only the improved description text, nothing else.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Improve geometric theorem descriptions based on usage examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating improved description: {e}")
            return current_description
    
    def export_knowledge_base(self, output_path: str):
        """Export knowledge base to JSON file"""
        
        with self.driver.session() as session:
            # Export theorems
            theorems_result = session.run("""
                MATCH (t:Theorem)
                RETURN t.theorem_id as theorem_id, t.name as name, t.description as description,
                       t.conditions as conditions, t.mathematical_form as mathematical_form,
                       t.usage_count as usage_count, t.source_problems as source_problems
            """)
            
            theorems = [dict(record) for record in theorems_result]
            
            
            # Export shapes
            shapes_result = session.run("""
                MATCH (s:Shape)
                RETURN s.name as name
            """)
            
            shapes = [record['name'] for record in shapes_result]
            
            # Create export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self._generate_knowledge_base_stats(),
                'theorems': theorems,
                'shapes': shapes
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge base exported to: {output_path}")
    
    def close(self):
        """Close database connection"""
        self.driver.close()


def main():
    """Example usage of generic geometric knowledge base builder with OpenAI Vision"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = ""
    OPENAI_API_KEY = ""

    # Configuration
    config = {
        'neo4j_uri': NEO4J_URI,
        'neo4j_user': NEO4J_USER,
        'neo4j_password': NEO4J_PASSWORD,
        'openai_api_key': OPENAI_API_KEY
    }
    
    # Initialize knowledge base manager
    kb_manager = GeometricKnowledgeBaseManager(config)
    
    try:
        # Build generic knowledge base from geometric dataset
        dataset_path = "images/geo3k/train/0"  # Can be GLLAVA or any other format
        
        # Estimate costs before processing
        num_problems = 1
        results = kb_manager.build_knowledge_base_from_dataset(
            dataset_path=dataset_path,
            max_problems=num_problems,
            continuous_improvement=True
        )
        
        print("Generic Knowledge Base Construction Results:")
        print(f"- Unique theorems: {results['processing_results']['unique_theorems_extracted']}")
        print(f"- Unique shapes: {results['processing_results']['unique_shapes_identified']}")
        print()
        
        # Show final statistics
        stats = results['final_statistics']
        print("Final Knowledge Base Statistics:")
        print(f"- Total theorems: {stats['total_theorems']}")
        print(f"- Total shapes: {stats['total_shapes']}")
        print(f"- Shape-theorem relationships: {stats['shape_theorem_relationships']}")
        print(f"- Top theorems: {[t[0] for t in stats['top_theorems'][:3]]}")
        
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        kb_manager.close()

if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    