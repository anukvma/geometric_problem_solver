"""
Scalable LLM-Driven Geometric Reasoning Engine - Part 3 Implementation
Uses LLMs to dynamically apply any theorem from the knowledge base for problem solving
"""

import math
import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import copy

# Core dependencies
import neo4j
from neo4j import GraphDatabase
import openai
import numpy as np
from collections import defaultdict, deque

# Import from previous parts
from knowledge_base_builder_openai1 import GeometricKnowledgeBase
from image_to_graph_builder_openai import GeometricImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningStepType(Enum):
    """Types of reasoning steps"""
    THEOREM_APPLICATION = "theorem_application"
    PROPERTY_CALCULATION = "property_calculation" 
    CONSTRAINT_PROPAGATION = "constraint_propagation"
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"

@dataclass
class GeometricVariable:
    """Represents a variable in the geometric problem"""
    name: str
    entity_type: str  # angle, length, area, etc.
    entity_id: str  # which shape/line/angle this refers to
    value: Optional[float] = None
    constraints: List[str] = field(default_factory=list)
    derivation_path: List[str] = field(default_factory=list)
    confidence: float = 1.0
    units: str = ""

@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process"""
    step_id: str
    step_type: ReasoningStepType
    theorem_used: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0
    mathematical_expression: str = ""
    llm_reasoning: str = ""

@dataclass
class ProblemContext:
    """Complete context for problem solving"""
    graph_id: str
    problem_text: str
    known_variables: Dict[str, GeometricVariable] = field(default_factory=dict)
    target_variables: List[str] = field(default_factory=list)
    available_theorems: List[Dict] = field(default_factory=list)
    graph_structure: Dict = field(default_factory=dict)
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)

class LLMTheoremApplicator:
    """Uses LLM to dynamically apply theorems from knowledge base"""
    
    def __init__(self, kb: GeometricKnowledgeBase):
        self.kb = kb
        self.openai_client = kb.openai_client
    
    def find_applicable_theorems(self, problem_context: ProblemContext) -> List[Dict]:
        """Use LLM to identify which theorems could be applied"""
        
        # Get current state summary
        state_summary = self._create_state_summary(problem_context)
        
        # Create theorem options
        theorem_options = self._format_theorem_options(problem_context.available_theorems)
        
        prompt = f"""
        You are a geometry expert analyzing what theorems can be applied to solve a problem.

        CURRENT STATE:
        Problem: {problem_context.problem_text}
        Known values: {state_summary['known']}
        Need to find: {problem_context.target_variables}
        Available shapes: {state_summary['shapes']}

        AVAILABLE THEOREMS:
        {theorem_options}

        For each theorem that could potentially be applied, determine:
        1. Can it be applied with current known values?
        2. What additional information would it provide?
        3. How helpful would it be for solving the problem?

        Return JSON format:
        {{
            "applicable_theorems": [
                {{
                    "theorem_name": "theorem name",
                    "theorem_id": "theorem_id",
                    "can_apply": true/false,
                    "reasoning": "why it can/cannot be applied",
                    "required_inputs": ["list of required variables"],
                    "possible_outputs": ["list of variables it could determine"],
                    "helpfulness_score": 0.0-1.0,
                    "missing_requirements": ["what's missing to apply it"]
                }}
            ]
        }}
        
        Focus on theorems that either:
        - Can be applied immediately with known values
        - Would directly help find target variables
        - Are one step away from being applicable
        
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a geometry expert. Analyze theorem applicability systematically. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = self._extract_json_from_response(response.choices[0].message.content)
            return result.get("applicable_theorems", [])
            
        except Exception as e:
            logger.error(f"Error finding applicable theorems: {e}")
            return []
    
    def apply_theorem(self, theorem_info: Dict, problem_context: ProblemContext) -> Optional[ReasoningStep]:
        """Use LLM to apply a specific theorem and calculate results"""
        
        theorem_details = self._get_theorem_details(theorem_info['theorem_id'])
        if not theorem_details:
            return None
        
        # Create detailed context for theorem application
        application_context = self._create_application_context(
            problem_context, theorem_info, theorem_details
        )
        
        logger.info(f"Application context {application_context}")

        prompt = f"""
        Apply the geometric theorem to solve for unknown values.

        THEOREM TO APPLY:
        Name: {theorem_details['name']}
        Description: {theorem_details['description']}
        Mathematical Form: {theorem_details['mathematical_form']}
        Conditions: {theorem_details['conditions']}
        Conclusions: {theorem_details['conclusions']}

        CURRENT PROBLEM STATE:
        {application_context}

        TASK:
        1. Verify all conditions are met for applying this theorem
        2. Apply the theorem using the known values
        3. Calculate any new unknown values
        4. Show all mathematical work step by step

        Return JSON format:
        {{
            "can_apply": true/false,
            "verification": "explanation of why conditions are/aren't met",
            "mathematical_work": "step by step calculation with actual numbers",
            "new_variables": {{
                "variable_name": {{
                    "value": numerical_value,
                    "units": "units if applicable",
                    "confidence": 0.0-1.0
                }}
            }},
            "reasoning_explanation": "clear explanation of the reasoning process",
            "formula_used": "the specific formula/equation used with actual values"
        }}
        
        Be precise with calculations and show all intermediate steps.
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a geometry expert. Apply theorems precisely with careful calculations. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = self._extract_json_from_response(response.choices[0].message.content)
            
            if result.get("can_apply"):
                return self._create_reasoning_step_from_llm_result(
                    theorem_info, theorem_details, result, problem_context
                )
            else:
                logger.info(f"Cannot apply theorem {theorem_info['theorem_name']}: {result.get('verification')}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying theorem: {e}")
            return None
    
    def _create_state_summary(self, problem_context: ProblemContext) -> Dict:
        """Create a concise summary of current problem state"""
        
        known_vars = {}
        for name, var in problem_context.known_variables.items():
            known_vars[name] = f"{var.value} {var.units}".strip() if var.value is not None else "unknown"
        
        shapes = []
        for shape_type, shape_list in problem_context.graph_structure.get('shapes_by_type', {}).items():
            shapes.extend([f"{shape_type}_{i}" for i in range(len(shape_list))])
        
        return {
            "known": known_vars,
            "shapes": shapes,
            "relationships": problem_context.graph_structure.get('relationships', [])
        }
    
    def _format_theorem_options(self, available_theorems: List[Dict]) -> str:
        """Format available theorems for LLM prompt"""
        
        theorem_texts = []
        for i, theorem in enumerate(available_theorems, 1):
            theorem_texts.append(f"""
            {i}. {theorem['theorem_name']} (ID: {theorem['theorem_id']})
               Description: {theorem.get('description', 'N/A')}
               Applicable to: {theorem.get('applicable_shapes', [])}
               Score: {theorem.get('score', 0)}
            """)
        
        return "\n".join(theorem_texts)
    
    def _get_theorem_details(self, theorem_id: str) -> Optional[Dict]:
        """Retrieve complete theorem details from knowledge base"""
        
        with self.kb.driver.session() as session:
            result = session.run("""
                MATCH (t:Theorem {theorem_id: $theorem_id})
                RETURN t.name as name, t.description as description,
                       t.conditions as conditions, t.conclusions as conclusions,
                       t.mathematical_form as mathematical_form,
                       t.applicable_shapes as applicable_shapes
            """, {'theorem_id': theorem_id}).single()
            
            if result:
                return dict(result)
            return None
    
    def _create_application_context(self, problem_context: ProblemContext, 
                                  theorem_info: Dict, theorem_details: Dict) -> str:
        """Create detailed context for theorem application"""
        
        context_parts = []
        
        # Known variables relevant to this theorem
        relevant_vars = {}
        for name, var in problem_context.known_variables.items():
            if any(shape in name.lower() for shape in theorem_details.get('applicable_shapes', [])):
                relevant_vars[name] = f"{var.value} {var.units}".strip()
        
        context_parts.append(f"Known values relevant to this theorem: {relevant_vars}")
        
        # Graph structure information
        shapes_info = []
        for shape_type, shapes in problem_context.graph_structure.get('shapes_by_type', {}).items():
            if shape_type in theorem_details.get('applicable_shapes', []):
                for shape in shapes:
                    shape_props = shape.get('properties', {})
                    shapes_info.append(f"{shape_type} {shape['shape_id']}: {shape_props}")
        
        if shapes_info:
            context_parts.append(f"Relevant shapes: {shapes_info}")
        
        # Target variables
        context_parts.append(f"Variables we need to find: {problem_context.target_variables}")
        
        return "\n".join(context_parts)
    
    def _create_reasoning_step_from_llm_result(self, theorem_info: Dict, theorem_details: Dict, 
                                             llm_result: Dict, problem_context: ProblemContext) -> ReasoningStep:
        """Create a ReasoningStep from LLM application result"""
        
        # Extract new variables
        new_vars = {}
        for var_name, var_info in llm_result.get("new_variables", {}).items():
            new_vars[var_name] = var_info.get("value")
        
        # Create reasoning step
        step = ReasoningStep(
            step_id=f"llm_apply_{theorem_info['theorem_id']}_{len(problem_context.reasoning_chain)}",
            step_type=ReasoningStepType.THEOREM_APPLICATION,
            theorem_used=theorem_details['name'],
            inputs={name: var.value for name, var in problem_context.known_variables.items()},
            outputs=new_vars,
            reasoning=llm_result.get("reasoning_explanation", "Applied theorem via LLM"),
            mathematical_expression=llm_result.get("formula_used", ""),
            llm_reasoning=llm_result.get("mathematical_work", ""),
            confidence=min([var_info.get("confidence", 0.8) 
                          for var_info in llm_result.get("new_variables", {}).values()] + [0.8])
        )
        
        return step
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            return {}

class LLMPropertyCalculator:
    """Uses LLM to calculate geometric properties dynamically"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def find_calculable_properties(self, problem_context: ProblemContext) -> List[Dict]:
        """Find properties that can be calculated with current knowledge"""
        
        shapes_info = self._format_shapes_for_llm(problem_context.graph_structure)
        known_values = {name: var.value for name, var in problem_context.known_variables.items()}
        
        prompt = f"""
        Analyze what geometric properties can be calculated with the current known values.

        AVAILABLE SHAPES AND THEIR PROPERTIES:
        {shapes_info}

        CURRENTLY KNOWN VALUES:
        {known_values}

        NEED TO FIND:
        {problem_context.target_variables}

        Consider basic geometric properties like:
        - Perimeter, area, circumference
        - Side lengths from coordinates
        - Angle measures from trigonometry
        - Similar/congruent triangle ratios

        Return JSON format:
        {{
            "calculable_properties": [
                {{
                    "property_name": "name of property (e.g., area_triangle_1)",
                    "calculation_type": "area|perimeter|angle|length|etc",
                    "required_inputs": ["list of known values needed"],
                    "calculation_method": "description of how to calculate",
                    "helpfulness_score": 0.0-1.0,
                    "confidence": 0.0-1.0
                }}
            ]
        }}
        
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Identify calculable geometric properties. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = self._extract_json_from_response(response.choices[0].message.content)
            return result.get("calculable_properties", [])
            
        except Exception as e:
            logger.error(f"Error finding calculable properties: {e}")
            return []
    
    def calculate_property(self, property_info: Dict, problem_context: ProblemContext) -> Optional[ReasoningStep]:
        """Calculate a specific geometric property using LLM"""
        
        known_values = {name: var.value for name, var in problem_context.known_variables.items() 
                       if name in property_info.get('required_inputs', [])}
        
        prompt = f"""
        Calculate the geometric property using the known values.

        PROPERTY TO CALCULATE:
        Name: {property_info['property_name']}
        Type: {property_info['calculation_type']}
        Method: {property_info['calculation_method']}

        KNOWN VALUES:
        {known_values}

        TASK:
        Calculate the property value step by step with precise mathematics.

        Return JSON format:
        {{
            "calculation_steps": "detailed step-by-step calculation",
            "final_value": numerical_result,
            "units": "appropriate units",
            "formula_used": "mathematical formula with actual numbers",
            "confidence": 0.0-1.0
        }}
        
        Show all mathematical work clearly.
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Calculate geometric properties precisely. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            result = self._extract_json_from_response(response.choices[0].message.content)
            
            if result.get("final_value") is not None:
                return ReasoningStep(
                    step_id=f"calc_{property_info['property_name']}_{len(problem_context.reasoning_chain)}",
                    step_type=ReasoningStepType.PROPERTY_CALCULATION,
                    inputs=known_values,
                    outputs={property_info['property_name']: result['final_value']},
                    reasoning=f"Calculated {property_info['calculation_type']} using basic geometry",
                    mathematical_expression=result.get('formula_used', ''),
                    llm_reasoning=result.get('calculation_steps', ''),
                    confidence=result.get('confidence', 0.8)
                )
            
        except Exception as e:
            logger.error(f"Error calculating property: {e}")
        
        return None
    
    def _format_shapes_for_llm(self, graph_structure: Dict) -> str:
        """Format shape information for LLM"""
        
        shape_descriptions = []
        for shape_type, shapes in graph_structure.get('shapes_by_type', {}).items():
            for shape in shapes:
                properties = shape.get('properties', {})
                shape_descriptions.append(f"{shape_type} {shape['shape_id']}: {properties}")
        
        return "\n".join(shape_descriptions)
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text.strip())
        except json.JSONDecodeError:
            return {}

class LLMStepSelector:
    """Uses LLM to intelligently select the best reasoning step"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def select_best_step(self, candidate_actions: List[Dict], 
                        problem_context: ProblemContext) -> Optional[Dict]:
        """Use LLM to select the most promising reasoning step"""
        
        if not candidate_actions:
            return None
        
        # Format candidates for LLM
        candidates_text = self._format_candidates(candidate_actions)
        progress_summary = self._create_progress_summary(problem_context)
        
        prompt = f"""
        You are solving a geometry problem step by step. Choose the best next action.

        PROBLEM: {problem_context.problem_text}
        
        CURRENT PROGRESS:
        {progress_summary}
        
        TARGET VARIABLES: {problem_context.target_variables}

        POSSIBLE NEXT ACTIONS:
        {candidates_text}

        Choose the action that will most effectively help solve the problem by considering:
        1. Does it directly find a target variable?
        2. Does it provide information needed for other steps?
        3. How confident/reliable is the action?
        4. Will it make meaningful progress?

        Return JSON format:
        {{
            "selected_action": {{
                "action_type": "theorem_application|property_calculation",
                "action_id": "the ID/identifier of the chosen action",
                "reasoning": "why this action was chosen",
                "expected_benefit": "what this will help accomplish"
            }}
        }}
        
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Choose the most effective geometry problem solving step. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            result = self._extract_json_from_response(response.choices[0].message.content)
            
            # Find the selected action in candidates
            selected_info = result.get("selected_action", {})
            action_id = selected_info.get("action_id")
            
            for candidate in candidate_actions:
                if (candidate.get("theorem_id") == action_id or 
                    candidate.get("property_name") == action_id or
                    candidate.get("action_id") == action_id):
                    candidate["selection_reasoning"] = selected_info.get("reasoning", "")
                    return candidate
            
        except Exception as e:
            logger.error(f"Error in step selection: {e}")
        
        # Fallback to first candidate if LLM selection fails
        return candidate_actions[0] if candidate_actions else None
    
    def _format_candidates(self, candidates: List[Dict]) -> str:
        """Format candidate actions for LLM"""
        
        formatted = []
        for i, candidate in enumerate(candidates, 1):
            if "theorem_name" in candidate:  # Theorem application
                formatted.append(f"""
                {i}. THEOREM: {candidate['theorem_name']}
                   ID: {candidate['theorem_id']}
                   Can apply: {candidate.get('can_apply', False)}
                   Would find: {candidate.get('possible_outputs', [])}
                   Helpfulness: {candidate.get('helpfulness_score', 0)}
                   Reasoning: {candidate.get('reasoning', 'N/A')}
                """)
            elif "property_name" in candidate:  # Property calculation
                formatted.append(f"""
                {i}. CALCULATION: {candidate['property_name']}
                   Type: {candidate.get('calculation_type', 'N/A')}
                   Method: {candidate.get('calculation_method', 'N/A')}
                   Helpfulness: {candidate.get('helpfulness_score', 0)}
                   Confidence: {candidate.get('confidence', 0)}
                """)
        
        return "\n".join(formatted)
    
    def _create_progress_summary(self, problem_context: ProblemContext) -> str:
        """Create a summary of current progress"""
        
        summary_parts = []
        
        # Known variables
        known_vars = [f"{name}: {var.value}" for name, var in problem_context.known_variables.items()]
        summary_parts.append(f"Known values: {', '.join(known_vars) if known_vars else 'None'}")
        
        # Steps taken so far
        if problem_context.reasoning_chain:
            recent_steps = [step.theorem_used or step.step_type.value 
                          for step in problem_context.reasoning_chain[-3:]]
            summary_parts.append(f"Recent steps: {' → '.join(recent_steps)}")
        
        # Progress toward targets
        found_targets = [target for target in problem_context.target_variables 
                        if target in problem_context.known_variables]
        remaining_targets = [target for target in problem_context.target_variables 
                           if target not in problem_context.known_variables]
        
        summary_parts.append(f"Found targets: {found_targets}")
        summary_parts.append(f"Still need: {remaining_targets}")
        
        return "\n".join(summary_parts)
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text.strip())
        except json.JSONDecodeError:
            return {}

class ScalableLLMReasoner:
    """Main scalable reasoning engine using LLMs"""
    
    def __init__(self, kb: GeometricKnowledgeBase, image_processor: GeometricImageProcessor):
        self.kb = kb
        self.image_processor = image_processor
        self.theorem_applicator = LLMTheoremApplicator(kb)
        self.property_calculator = LLMPropertyCalculator(kb.openai_client)
        self.step_selector = LLMStepSelector(kb.openai_client)
        self.openai_client = kb.openai_client
    
    def solve_problem(self, problem_text: str, graph_id: str, 
                     max_iterations: int = 3) -> Dict[str, Any]:
        """Main scalable problem solving method"""
        
        logger.info(f"Starting scalable LLM-based problem solving for graph: {graph_id}")
        
        # Initialize problem context
        problem_context = self._initialize_problem_context(problem_text, graph_id)
        
        if not problem_context.target_variables:
            return {
                'success': False,
                'error': 'Could not identify target variables',
                'problem_text': problem_text,
                'graph_id': graph_id
            }
        
        # Main reasoning loop
        iteration = 0
        no_progress_count = 0
        
        while (iteration < max_iterations and 
               not self._all_targets_found(problem_context) and
               no_progress_count < 3):
            
            logger.info(f"Reasoning iteration {iteration + 1}")
            
            # Find all possible actions
            theorem_candidates = self.theorem_applicator.find_applicable_theorems(problem_context)
            property_candidates = self.property_calculator.find_calculable_properties(problem_context)
            logger.info(f"theorem_candidates {theorem_candidates}")
            logger.info(f"property_candidates {property_candidates}")
            # Combine all candidates
            all_candidates = []
            
            # Add theorem applications
            for theorem in theorem_candidates:
                if theorem.get('can_apply') or theorem.get('helpfulness_score', 0) > 0.3:
                    theorem['action_type'] = 'theorem_application'
                    theorem['action_id'] = theorem['theorem_id']
                    all_candidates.append(theorem)
            
            # Add property calculations
            for prop in property_candidates:
                if prop.get('helpfulness_score', 0) > 0.3:
                    prop['action_type'] = 'property_calculation'
                    prop['action_id'] = prop['property_name']
                    all_candidates.append(prop)
            
            if not all_candidates:
                logger.info("No applicable actions found")
                no_progress_count += 1
                iteration += 1
                continue
            
            # Select best action using LLM
            selected_action = self.step_selector.select_best_step(all_candidates, problem_context)
            
            if not selected_action:
                logger.info("No action selected")
                no_progress_count += 1
                iteration += 1
                continue
            
            # Execute selected action
            executed_step = None
            
            if selected_action['action_type'] == 'theorem_application':
                logger.info(f"Applying Threorems {selected_action} for problem {problem_context}")
                executed_step = self.theorem_applicator.apply_theorem(selected_action, problem_context)
            elif selected_action['action_type'] == 'property_calculation':
                logger.info(f"Calculating property {selected_action} for problem {problem_context}")
                
                executed_step = self.property_calculator.calculate_property(selected_action, problem_context)
            
            if executed_step and executed_step.outputs:
                # Update problem context with new variables
                self._apply_reasoning_step(executed_step, problem_context)
                no_progress_count = 0
                logger.info(f"Applied {executed_step.theorem_used or executed_step.step_type.value}: "
                          f"found {list(executed_step.outputs.keys())}")
            else:
                no_progress_count += 1
                logger.info("Selected action produced no results")
            
            iteration += 1
        
        # Generate final solution
        return self._generate_solution(problem_context)
    
    def _initialize_problem_context(self, problem_text: str, graph_id: str) -> ProblemContext:
        """Initialize the problem context with all necessary information"""
        
        # Load graph structure
        graph_structure = self._load_graph_structure(graph_id)
        logger.info(f"Retrieved Graph Structure {graph_structure}")
        # Get available theorems
        available_theorems = self._get_available_theorems(graph_id)
        logger.info(f"Retrieved Available Theorems {available_theorems}")
        # Extract known variables and targets using LLM
        problem_analysis = self._analyze_problem_with_llm(problem_text, graph_structure)
        logger.info(f"Problem Analysis {problem_analysis}")
        # Create known variables
        known_variables = {}
        for var_name, var_info in problem_analysis.get('known_variables', {}).items():
            known_variables[var_name] = GeometricVariable(
                name=var_name,
                entity_type=var_info.get('type', 'unknown'),
                entity_id=var_info.get('entity_id', ''),
                value=var_info.get('value'),
                confidence=var_info.get('confidence', 0.9),
                units=var_info.get('units', '')
            )
        
        return ProblemContext(
            graph_id=graph_id,
            problem_text=problem_text,
            known_variables=known_variables,
            target_variables=problem_analysis.get('target_variables', []),
            available_theorems=available_theorems,
            graph_structure=graph_structure
        )
    
    def _load_graph_structure(self, graph_id: str) -> Dict:
        """Load complete graph structure information"""
        
        with self.kb.driver.session() as session:
            # Get shapes
            shapes_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s:Shape)
                RETURN s.shape_id as shape_id, s.shape_type as shape_type, 
                       s.properties as properties, s.confidence as confidence
            """, {'graph_id': graph_id})
            
            shapes_by_type = defaultdict(list)
            shapes_by_id = {}
            
            for record in shapes_result:
                shape_info = {
                    'shape_id': record['shape_id'],
                    'shape_type': record['shape_type'],
                    'properties': json.loads(record['properties'] or '{}'),
                    'confidence': record['confidence']
                }
                shapes_by_type[record['shape_type']].append(shape_info)
                shapes_by_id[record['shape_id']] = shape_info
            
            # Get relationships
            relationships_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s1:Shape)
                MATCH (s1)-[r:RELATES_TO]->(s2:Shape)
                RETURN s1.shape_id as shape1_id, s2.shape_id as shape2_id,
                       r.relationship_type as rel_type, r.properties as rel_props
            """, {'graph_id': graph_id})
            
            relationships = []
            for record in relationships_result:
                relationships.append({
                    'shape1_id': record['shape1_id'],
                    'shape2_id': record['shape2_id'],
                    'relationship_type': record['rel_type'],
                    'properties': json.loads(record['rel_props'] or '{}')
                })
            
            # Get points and lines
            points_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(p:Point)
                RETURN p.point_id as point_id, p.x as x, p.y as y, p.label as label
            """, {'graph_id': graph_id})
            
            points = []
            for record in points_result:
                points.append({
                    'point_id': record['point_id'],
                    'x': record['x'],
                    'y': record['y'],
                    'label': record['label']
                })
            
            lines_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(l:Line)
                RETURN l.line_id as line_id, l.length as length, l.angle as angle
            """, {'graph_id': graph_id})
            
            lines = []
            for record in lines_result:
                lines.append({
                    'line_id': record['line_id'],
                    'length': record['length'],
                    'angle': record['angle']
                })
        
        return {
            'shapes_by_type': dict(shapes_by_type),
            'shapes_by_id': shapes_by_id,
            'relationships': relationships,
            'points': points,
            'lines': lines
        }
    
    def _get_available_theorems(self, graph_id: str) -> List[Dict]:
        """Get all theorems applicable to this graph"""
        
        with self.kb.driver.session() as session:
            result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s:Shape)
                MATCH (s)-[r:CAN_APPLY]->(t:Theorem)
                RETURN DISTINCT t.theorem_id as theorem_id, t.name as theorem_name, 
                       t.description as description, t.mathematical_form as mathematical_form,
                       t.conditions as conditions, t.conclusions as conclusions,
                       t.applicable_shapes as applicable_shapes,
                       avg(r.applicability_score) as avg_score
                ORDER BY avg_score DESC
            """, {'graph_id': graph_id})
            
            theorems = []
            for record in result:
                theorems.append({
                    'theorem_id': record['theorem_id'],
                    'theorem_name': record['theorem_name'],
                    'description': record['description'],
                    'mathematical_form': record['mathematical_form'],
                    'conditions': record['conditions'],
                    'conclusions': record['conclusions'],
                    'applicable_shapes': record['applicable_shapes'],
                    'score': record['avg_score']
                })
            
            return theorems
    
    def _analyze_problem_with_llm(self, problem_text: str, graph_structure: Dict) -> Dict:
        """Analyze problem using LLM to extract variables and targets"""
        
        # Format graph structure for LLM
        structure_summary = self._format_structure_for_llm(graph_structure)
        
        prompt = f"""
        Analyze this geometry problem to identify known values and what needs to be found.
        
        PROBLEM: {problem_text}
        
        AVAILABLE GEOMETRIC ELEMENTS:
        {structure_summary}
        
        Extract:
        1. All known numerical values (given measurements, angles, etc.)
        2. What the problem is asking to find
        3. Map these to the geometric elements available
        
        Use consistent naming:
        - For angles: angle_[shape_id]_[vertex] or angle_A, angle_B, etc.
        - For sides: side_[shape_id]_[index] or side_AB, side_BC, etc.
        - For areas: area_[shape_id] or area_triangle, area_circle, etc.
        - For other properties: radius_[shape_id], perimeter_[shape_id], etc.
        
        Return JSON format:
        {{
            "known_variables": {{
                "variable_name": {{
                    "value": numerical_value,
                    "type": "angle|length|area|etc",
                    "entity_id": "which shape/element this refers to",
                    "units": "degrees|cm|etc",
                    "confidence": 0.0-1.0
                }}
            }},
            "target_variables": ["list", "of", "variables", "to", "find"],
            "problem_type": "triangle_solving|area_calculation|angle_finding|etc",
            "key_relationships": ["important geometric relationships mentioned"]
        }}
        
        Be precise with variable names and ensure they match the available geometric elements.
        Return only valid JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Analyze geometry problems precisely. Extract variables and targets carefully. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = self._extract_json_from_response(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing problem: {e}")
            return {"known_variables": {}, "target_variables": []}
    
    def _format_structure_for_llm(self, graph_structure: Dict) -> str:
        """Format graph structure for LLM understanding"""
        
        structure_parts = []
        
        # Shapes
        if graph_structure.get('shapes_by_type'):
            structure_parts.append("SHAPES:")
            for shape_type, shapes in graph_structure['shapes_by_type'].items():
                for shape in shapes:
                    props = shape.get('properties', {})
                    structure_parts.append(f"  {shape_type} {shape['shape_id']}: {props}")
        
        # Points
        if graph_structure.get('points'):
            structure_parts.append("POINTS:")
            for point in graph_structure['points']:
                label = f" ({point['label']})" if point.get('label') else ""
                structure_parts.append(f"  Point {point['point_id']}{label}: ({point['x']:.1f}, {point['y']:.1f})")
        
        # Lines
        if graph_structure.get('lines'):
            structure_parts.append("LINES:")
            for line in graph_structure['lines']:
                structure_parts.append(f"  Line {line['line_id']}: length={line.get('length', 'unknown')}")
        
        # Relationships
        if graph_structure.get('relationships'):
            structure_parts.append("RELATIONSHIPS:")
            for rel in graph_structure['relationships']:
                structure_parts.append(f"  {rel['shape1_id']} {rel['relationship_type']} {rel['shape2_id']}")
        
        return "\n".join(structure_parts)
    
    def _all_targets_found(self, problem_context: ProblemContext) -> bool:
        """Check if all target variables have been found"""
        for target in problem_context.target_variables:
            if target not in problem_context.known_variables:
                return False
        return True
    
    def _apply_reasoning_step(self, step: ReasoningStep, problem_context: ProblemContext):
        """Apply reasoning step and update context"""
        
        # Add new variables to known variables
        for var_name, value in step.outputs.items():
            if isinstance(value, (int, float)) and value is not None:
                # Determine variable properties
                var_parts = var_name.split('_')
                entity_type = var_parts[0] if var_parts else 'unknown'
                entity_id = var_parts[1] if len(var_parts) > 1 else ''
                
                # Determine units based on type
                units = ""
                if entity_type in ['angle']:
                    units = "degrees"
                elif entity_type in ['side', 'length', 'radius', 'perimeter', 'circumference']:
                    units = "units"
                elif entity_type in ['area']:
                    units = "square units"
                
                variable = GeometricVariable(
                    name=var_name,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    value=value,
                    derivation_path=[step.theorem_used or step.step_type.value],
                    confidence=step.confidence,
                    units=units
                )
                
                problem_context.known_variables[var_name] = variable
        
        # Add step to reasoning chain
        problem_context.reasoning_chain.append(step)
    
    def _generate_solution(self, problem_context: ProblemContext) -> Dict[str, Any]:
        """Generate comprehensive solution from problem context"""
        
        # Determine success
        found_targets = []
        missing_targets = []
        
        for target in problem_context.target_variables:
            if target in problem_context.known_variables:
                found_targets.append(target)
            else:
                missing_targets.append(target)
        
        success = len(missing_targets) == 0
        completion_rate = len(found_targets) / len(problem_context.target_variables) if problem_context.target_variables else 0
        
        # Generate solution summary
        solution = {
            'success': success,
            'completion_rate': completion_rate,
            'problem_text': problem_context.problem_text,
            'graph_id': problem_context.graph_id,
            'target_variables': problem_context.target_variables,
            'found_variables': {},
            'missing_variables': missing_targets,
            'reasoning_steps': [],
            'total_steps': len(problem_context.reasoning_chain),
            'confidence': 0.0
        }
        
        # Add found variables
        for target in found_targets:
            var = problem_context.known_variables[target]
            solution['found_variables'][target] = {
                'value': var.value,
                'units': var.units,
                'confidence': var.confidence,
                'derivation_path': var.derivation_path
            }
        
        # Add reasoning steps
        for step in problem_context.reasoning_chain:
            step_info = {
                'step_id': step.step_id,
                'step_type': step.step_type.value,
                'theorem_used': step.theorem_used,
                'reasoning': step.reasoning,
                'mathematical_expression': step.mathematical_expression,
                'llm_reasoning': step.llm_reasoning,
                'inputs': step.inputs,
                'outputs': step.outputs,
                'confidence': step.confidence
            }
            solution['reasoning_steps'].append(step_info)
        
        # Calculate overall confidence
        if found_targets:
            confidences = [problem_context.known_variables[t].confidence for t in found_targets]
            solution['confidence'] = sum(confidences) / len(confidences)
        
        # Generate natural language explanation
        solution['explanation'] = self._generate_natural_language_solution(solution, problem_context)
        
        return solution
    
    def _generate_natural_language_solution(self, solution: Dict, problem_context: ProblemContext) -> str:
        """Generate human-readable explanation using LLM"""
        
        # Prepare solution data for LLM
        solution_summary = {
            'problem': problem_context.problem_text,
            'found_values': solution['found_variables'],
            'reasoning_steps': [
                {
                    'theorem': step.get('theorem_used', 'Calculation'),
                    'reasoning': step.get('reasoning', ''),
                    'formula': step.get('mathematical_expression', ''),
                    'outputs': step.get('outputs', {})
                }
                for step in solution['reasoning_steps']
            ]
        }
        
        prompt = f"""
        Generate a clear, step-by-step explanation for this geometry problem solution.
        
        PROBLEM: {problem_context.problem_text}
        
        SOLUTION DATA: {json.dumps(solution_summary, indent=2)}
        
        Create a natural language explanation that:
        1. States what was found
        2. Explains each reasoning step clearly
        3. Shows the mathematical work
        4. Uses proper geometric terminology
        5. Is educational and easy to follow
        
        Format as a clear, structured explanation suitable for a student.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate clear, educational explanations of geometry problem solutions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            
            # Fallback explanation
            explanation_parts = []
            
            if solution['success']:
                explanation_parts.append("Solution found successfully!")
                explanation_parts.append("\nResults:")
                for var_name, var_info in solution['found_variables'].items():
                    explanation_parts.append(f"• {var_name}: {var_info['value']:.2f} {var_info['units']}")
                
                explanation_parts.append(f"\nSolved in {solution['total_steps']} steps using geometric theorems and calculations.")
            else:
                explanation_parts.append("Partial solution found.")
                explanation_parts.append(f"Found {len(solution['found_variables'])} out of {len(solution['target_variables'])} target variables.")
            
            return "\n".join(explanation_parts)
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {}

class ScalableGeometricProblemSolver:
    """Main scalable problem solver using LLM-driven reasoning"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        
        # Initialize components
        self.kb = GeometricKnowledgeBase(
            neo4j_uri=config['neo4j_uri'],
            neo4j_user=config['neo4j_user'],
            neo4j_password=config['neo4j_password'],
            openai_api_key=config['openai_api_key']
        )
        
        self.image_processor = GeometricImageProcessor(config)
        self.reasoner = ScalableLLMReasoner(self.kb, self.image_processor)
    
    def solve_geometric_problem(self, image_path: str, problem_text: str, 
                               problem_id: Optional[str] = None) -> Dict[str, Any]:
        """Complete scalable pipeline for geometric problem solving"""
        
        try:
            logger.info(f"Solving geometric problem with scalable LLM approach: {problem_id or 'unnamed'}")
            
            # Step 1: Process image and create graph
            logger.info("Step 1: Processing image and creating graph...")
            graph_result = self.image_processor.process_image(image_path, problem_id)
            
            if not graph_result['success']:
                return {
                    'success': False,
                    'error': f"Graph creation failed: {graph_result['error']}",
                    'stage': 'graph_creation'
                }
            
            graph_id = graph_result['graph_id']
            logger.info(f"Created graph: {graph_id}")
            
            # Step 2: Apply scalable LLM reasoning
            logger.info("Step 2: Applying scalable LLM reasoning...")
            solution = self.reasoner.solve_problem(problem_text, graph_id)
            
            # Step 3: Enhance solution with graph context
            solution.update({
                'image_path': image_path,
                'graph_summary': graph_result['summary'],
                'available_theorems': graph_result['applicable_theorems'],
                'shape_relationships': graph_result['shape_relationships'],
                'approach': 'scalable_llm_reasoning'
            })
            
            logger.info(f"Scalable problem solving completed. Success: {solution['success']}, "
                       f"Completion: {solution.get('completion_rate', 0):.2%}")
            
            return solution
            
        except Exception as e:
            logger.error(f"Error in scalable problem solving pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'unknown',
                'approach': 'scalable_llm_reasoning'
            }
    
    def batch_solve_problems(self, problems: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Solve multiple geometric problems using scalable approach"""
        
        results = []
        success_count = 0
        
        for i, problem in enumerate(problems):
            logger.info(f"Solving problem {i+1}/{len(problems)} with scalable approach")
            
            result = self.solve_geometric_problem(
                image_path=problem['image_path'],
                problem_text=problem['problem_text'],
                problem_id=problem.get('problem_id', f'scalable_problem_{i}')
            )
            
            result['batch_index'] = i
            results.append(result)
            
            if result.get('success'):
                success_count += 1
        
        logger.info(f"Batch processing completed: {success_count}/{len(problems)} problems solved successfully")
        return results
    
    def compare_with_traditional_approach(self, problems: List[Dict[str, str]]) -> Dict[str, Any]:
        """Compare scalable LLM approach with traditional hardcoded approach"""
        
        # This would run both approaches and compare results
        # Implementation would involve running both solvers and analyzing:
        # - Success rates
        # - Solution quality
        # - Processing time
        # - Theorem utilization
        
        comparison_results = {
            'total_problems': len(problems),
            'scalable_llm_results': self.batch_solve_problems(problems),
            'comparison_metrics': {
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_steps': 0.0,
                'theorem_diversity': 0.0
            }
        }
        
        # Calculate metrics
        successful_solutions = [r for r in comparison_results['scalable_llm_results'] if r.get('success')]
        
        if successful_solutions:
            comparison_results['comparison_metrics']['success_rate'] = len(successful_solutions) / len(problems)
            comparison_results['comparison_metrics']['avg_confidence'] = sum(r.get('confidence', 0) for r in successful_solutions) / len(successful_solutions)
            comparison_results['comparison_metrics']['avg_steps'] = sum(r.get('total_steps', 0) for r in successful_solutions) / len(successful_solutions)
            
            # Calculate theorem diversity (unique theorems used)
            all_theorems = set()
            for result in successful_solutions:
                for step in result.get('reasoning_steps', []):
                    if step.get('theorem_used'):
                        all_theorems.add(step['theorem_used'])
            comparison_results['comparison_metrics']['theorem_diversity'] = len(all_theorems)
        
        return comparison_results
    
    def close(self):
        """Clean up resources"""
        self.image_processor.close()
        self.kb.close()

# Example usage demonstrating scalability
def main():
    """Demonstrate scalable LLM-driven geometric problem solving"""
    
    # Configuration
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
    
    # Initialize scalable solver
    solver = ScalableGeometricProblemSolver(config)
    
    try:
        # Example 1: Complex triangle problem
        complex_problem = {
            'image_path': "/Users/anupreksha/Documents/23.png",
            'problem_text': "Given parallelogram PQRS. Find length of RS",
            'problem_id': "complex_triangle_005"
        }
        
        result = solver.solve_geometric_problem(**complex_problem)
        print("Complex Problem Result:")
        print(f"Success: {result['success']}")
        print(f"Completion Rate: {result.get('completion_rate', 0):.2%}")
        print(f"Steps taken: {result.get('total_steps', 0)}")
        print(f"Explanation: {result.get('explanation', 'N/A')[:200]}...")
        print()
        
        
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        solver.close()

if __name__ == "__main__":
    main()