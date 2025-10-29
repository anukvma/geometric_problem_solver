"""
Scalable LLM-Driven Geometric Reasoning Engine - Part 3 Implementation (Gemini Version)
Uses LLMs to dynamically apply any theorem from the knowledge base for problem solving
"""

import math
import logging
import json
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import copy

# Core dependencies
import neo4j
from neo4j import GraphDatabase
import google.generativeai as genai
import numpy as np
from collections import defaultdict, deque

# Import from previous parts
from knowledge_base_builder import GeometricKnowledgeBase
from image_to_graph_builder import GeometricImageProcessor

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
        self.gemini_model = kb.gemini_model

    def _safe_get_response_text(self, response) -> Tuple[Optional[str], Optional[str]]:
        """Safely extract text from Gemini response, handling different finish reasons

        Returns:
            Tuple of (response_text, error_message)
            - If successful: (text, None)
            - If error: (None, error_message)
        """
        try:
            if not response.candidates:
                return None, "No candidates in response"

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            if finish_reason == 1:  # STOP - success
                try:
                    return response.text, None
                except ValueError as e:
                    logger.error(f"Error accessing response.text: {e}")
                    return None, f"Response text access error: {str(e)}"

            elif finish_reason == 2:  # MAX_TOKENS
                logger.warning("Response truncated due to MAX_TOKENS")
                try:
                    if candidate.content and candidate.content.parts:
                        partial_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        if partial_text:
                            logger.info(f"Extracted partial text ({len(partial_text)} chars)")
                            return partial_text, "MAX_TOKENS"
                except Exception as e:
                    logger.error(f"Could not extract partial text: {e}")
                return None, "MAX_TOKENS - Response truncated"

            elif finish_reason == 3:  # SAFETY
                logger.error("Response blocked by safety filters")
                return None, "SAFETY - Response blocked by safety filters"

            elif finish_reason == 4:  # RECITATION
                logger.error("Response blocked due to recitation")
                return None, "RECITATION - Response blocked"

            else:
                logger.error(f"Unexpected finish_reason: {finish_reason}")
                return None, f"Unexpected finish_reason: {finish_reason}"

        except Exception as e:
            logger.error(f"Error in _safe_get_response_text: {e}")
            return None, f"Exception accessing response: {str(e)}"

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
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 16000,
                }
            )

            response_text, error_msg = self._safe_get_response_text(response)
            if response_text is None:
                logger.error(f"Failed to get response text: {error_msg}")
                return []

            result = self._extract_json_from_response(response_text)
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
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 16000,
                }
            )

            response_text, error_msg = self._safe_get_response_text(response)
            if response_text is None:
                logger.error(f"Failed to get response text: {error_msg}")
                return None

            result = self._extract_json_from_response(response_text)

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

    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def _safe_get_response_text(self, response) -> Tuple[Optional[str], Optional[str]]:
        """Safely extract text from Gemini response, handling different finish reasons"""
        try:
            if not response.candidates:
                return None, "No candidates in response"

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            if finish_reason == 1:  # STOP - success
                try:
                    return response.text, None
                except ValueError as e:
                    logger.error(f"Error accessing response.text: {e}")
                    return None, f"Response text access error: {str(e)}"

            elif finish_reason == 2:  # MAX_TOKENS
                logger.warning("Response truncated due to MAX_TOKENS")
                try:
                    if candidate.content and candidate.content.parts:
                        partial_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        if partial_text:
                            return partial_text, "MAX_TOKENS"
                except Exception as e:
                    logger.error(f"Could not extract partial text: {e}")
                return None, "MAX_TOKENS - Response truncated"

            elif finish_reason == 3:  # SAFETY
                logger.error("Response blocked by safety filters")
                return None, "SAFETY - Response blocked"

            elif finish_reason == 4:  # RECITATION
                logger.error("Response blocked due to recitation")
                return None, "RECITATION - Response blocked"

            else:
                logger.error(f"Unexpected finish_reason: {finish_reason}")
                return None, f"Unexpected finish_reason: {finish_reason}"

        except Exception as e:
            logger.error(f"Error in _safe_get_response_text: {e}")
            return None, f"Exception accessing response: {str(e)}"

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
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 16000,
                }
            )

            response_text, error_msg = self._safe_get_response_text(response)
            if response_text is None:
                logger.error(f"Failed to get response text: {error_msg}")
                return []

            result = self._extract_json_from_response(response_text)
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
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 16000,
                }
            )

            response_text, error_msg = self._safe_get_response_text(response)
            if response_text is None:
                logger.error(f"Failed to get response text: {error_msg}")
                return None

            result = self._extract_json_from_response(response_text)

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

    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def _safe_get_response_text(self, response) -> Tuple[Optional[str], Optional[str]]:
        """Safely extract text from Gemini response, handling different finish reasons"""
        try:
            if not response.candidates:
                return None, "No candidates in response"

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            if finish_reason == 1:  # STOP - success
                try:
                    return response.text, None
                except ValueError as e:
                    logger.error(f"Error accessing response.text: {e}")
                    return None, f"Response text access error: {str(e)}"

            elif finish_reason == 2:  # MAX_TOKENS
                logger.warning("Response truncated due to MAX_TOKENS")
                try:
                    if candidate.content and candidate.content.parts:
                        partial_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                        if partial_text:
                            return partial_text, "MAX_TOKENS"
                except Exception as e:
                    logger.error(f"Could not extract partial text: {e}")
                return None, "MAX_TOKENS - Response truncated"

            elif finish_reason == 3:  # SAFETY
                logger.error("Response blocked by safety filters")
                return None, "SAFETY - Response blocked"

            elif finish_reason == 4:  # RECITATION
                logger.error("Response blocked due to recitation")
                return None, "RECITATION - Response blocked"

            else:
                logger.error(f"Unexpected finish_reason: {finish_reason}")
                return None, f"Unexpected finish_reason: {finish_reason}"

        except Exception as e:
            logger.error(f"Error in _safe_get_response_text: {e}")
            return None, f"Exception accessing response: {str(e)}"

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
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 16000,
                }
            )

            response_text, error_msg = self._safe_get_response_text(response)
            if response_text is None:
                logger.error(f"Failed to get response text: {error_msg}")
                # Fallback to first candidate if LLM fails
                return candidate_actions[0] if candidate_actions else None

            result = self._extract_json_from_response(response_text)

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
        self.gemini_model = kb.gemini_model

    def solve_problem(self, problem_text: str, graph_id: str,
                     max_iterations: int = 1, choices: Optional[List[str]] = None,
                     include_image: bool = False, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Single-iteration problem solving with complete knowledge graph and theorems"""

        logger.info(f"Starting single-iteration LLM-based problem solving for graph: {graph_id}")
        logger.info(f"Include image in prompt: {include_image}")

        # Load complete graph structure
        graph_structure = self._load_graph_structure(graph_id)

        # Get all available theorems
        available_theorems = self._get_available_theorems(graph_id)

        # Solve using single comprehensive LLM call
        solution = self._solve_with_complete_context(
            problem_text=problem_text,
            graph_id=graph_id,
            graph_structure=graph_structure,
            available_theorems=available_theorems,
            choices=choices,
            include_image=include_image,
            image_path=image_path
        )

        return solution

    def _solve_with_complete_context(self, problem_text: str, graph_id: str,
                                    graph_structure: Dict, available_theorems: List[Dict],
                                    choices: Optional[List[str]] = None,
                                    include_image: bool = False,
                                    image_path: Optional[str] = None) -> Dict[str, Any]:
        """Solve problem with complete knowledge graph and theorems in single prompt, supporting parametric/algebraic solutions

        Includes retry logic with progressive token limits to handle LLM response errors.
        """

        # Progressive token limits for retry attempts
        max_attempts = 3
        token_limits = [16000, 32000, 65000]

        for attempt in range(max_attempts):
            try:
                logger.info(f"Solving attempt {attempt + 1}/{max_attempts} with {token_limits[attempt]} tokens")

                result = self._attempt_solve(
                    problem_text, graph_id, graph_structure, available_theorems,
                    choices, include_image, image_path, token_limits[attempt]
                )

                # If successful, return immediately
                if result and not result.get('error'):
                    logger.info(f"Successfully solved on attempt {attempt + 1}")
                    return result

                # If this was the last attempt, return the result (even if it has errors)
                if attempt == max_attempts - 1:
                    logger.warning(f"All {max_attempts} attempts completed, returning last result")
                    return result

                # Otherwise, retry with higher token limit
                logger.warning(f"Attempt {attempt + 1} had issues, retrying with higher token limit...")

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    # Last attempt failed, return error
                    import traceback
                    traceback.print_exc()
                    return {
                        'success': False,
                        'error': str(e),
                        'problem_text': problem_text,
                        'graph_id': graph_id
                    }
                # Retry on next iteration
                logger.warning(f"Retrying after error...")

        # Should not reach here, but return error as fallback
        return {
            'success': False,
            'error': 'All solve attempts failed',
            'problem_text': problem_text,
            'graph_id': graph_id
        }

    def _safe_get_response_text(self, response) -> Tuple[Optional[str], Optional[str]]:
        """Safely extract text from Gemini response, handling different finish reasons

        Returns:
            Tuple of (response_text, error_message)
            - If successful: (text, None)
            - If error: (None, error_message)
        """
        try:
            # Check if response has candidates
            if not response.candidates:
                return None, "No candidates in response"

            candidate = response.candidates[0]

            # Check finish_reason
            # 0: FINISH_REASON_UNSPECIFIED
            # 1: STOP (natural stop - success)
            # 2: MAX_TOKENS (ran out of tokens)
            # 3: SAFETY (safety filters triggered)
            # 4: RECITATION (recitation threshold)
            # 5: OTHER

            finish_reason = candidate.finish_reason

            if finish_reason == 1:  # STOP - success
                try:
                    return response.text, None
                except ValueError as e:
                    logger.error(f"Error accessing response.text despite STOP finish_reason: {e}")
                    return None, f"Response text access error: {str(e)}"

            elif finish_reason == 2:  # MAX_TOKENS
                logger.warning("Response truncated due to MAX_TOKENS (finish_reason=2)")
                # Try to get partial text from parts
                try:
                    if candidate.content and candidate.content.parts:
                        partial_text = ""
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                partial_text += part.text
                        if partial_text:
                            logger.info(f"Extracted partial text ({len(partial_text)} chars) from truncated response")
                            return partial_text, "MAX_TOKENS"  # Return with MAX_TOKENS flag for retry
                except Exception as e:
                    logger.error(f"Could not extract partial text: {e}")

                return None, "MAX_TOKENS - Response truncated, retry with higher token limit needed"

            elif finish_reason == 3:  # SAFETY
                logger.error("Response blocked by safety filters (finish_reason=3)")
                # Try to get the safety ratings for debugging
                if hasattr(candidate, 'safety_ratings'):
                    logger.error(f"Safety ratings: {candidate.safety_ratings}")
                return None, "SAFETY - Response blocked by safety filters"

            elif finish_reason == 4:  # RECITATION
                logger.error("Response blocked due to recitation (finish_reason=4)")
                return None, "RECITATION - Response blocked due to recitation threshold"

            else:  # OTHER or UNSPECIFIED
                logger.error(f"Response finished with unexpected reason: {finish_reason}")
                # Try to get any available text
                try:
                    if candidate.content and candidate.content.parts:
                        text = ""
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text += part.text
                        if text:
                            return text, None
                except:
                    pass
                return None, f"Unexpected finish_reason: {finish_reason}"

        except Exception as e:
            logger.error(f"Error in _safe_get_response_text: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Exception accessing response: {str(e)}"

    def _attempt_solve(self, problem_text: str, graph_id: str,
                      graph_structure: Dict, available_theorems: List[Dict],
                      choices: Optional[List[str]], include_image: bool,
                      image_path: Optional[str], max_tokens: int) -> Dict[str, Any]:
        """Single attempt to solve the problem with specified token limit"""

        # Format knowledge graph for LLM
        kg_representation = self._format_complete_knowledge_graph(graph_structure)

        # Format theorems for LLM
        theorems_representation = self._format_available_theorems(available_theorems)

        # Detect if problem has parameters
        has_parameters = graph_structure.get('has_parameters', False)
        parameters_note = ""
        if has_parameters:
            parameters_note = """
**NOTE: This problem contains PARAMETERS/VARIABLES instead of numeric values.**
- You MUST use algebraic reasoning and symbolic mathematics
- Set up equations using the parametric expressions shown in the knowledge graph
- Solve algebraically by manipulating equations
- Your answer may be in terms of parameters (e.g., "2x + 5" or "3y/2")
- Show all algebraic steps clearly
"""

        # Format choices if this is a multiple choice question
        choices_text = ""
        if choices:
            choices_text = f"""

ANSWER CHOICES:
{chr(10).join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])}

**IMPORTANT: After solving, you MUST select the correct answer choice from the options above.**
"""

        # Add note about image inclusion
        image_note = ""
        if include_image and image_path:
            image_note = """

**NOTE: The geometric diagram image is provided along with the knowledge graph.**
- You can see the actual diagram to verify the knowledge graph
- Use the image to identify any visual details that might not be captured in the knowledge graph
- Cross-reference the image with the knowledge graph for accuracy
"""

        prompt = f"""
You are an expert geometry problem solver. You have been given a complete knowledge graph of a geometric figure and all applicable theorems. Use multi-hop reasoning to solve the problem.

PROBLEM TO SOLVE:
{problem_text}
{choices_text}

COMPLETE KNOWLEDGE GRAPH:
{kg_representation}

AVAILABLE THEOREMS AND PROPERTIES:
{theorems_representation}

{parameters_note}
{image_note}

TASK:
1. Analyze the knowledge graph to understand the geometric figure
2. Identify what information is given (numeric or parametric) and what needs to be found
3. Use multi-hop reasoning to connect known values to unknown values
4. Apply relevant theorems from the available list OR suggest new theorems if needed
5. Show all reasoning steps clearly with intermediate results
6. If the problem has parameters:
   - Set up algebraic equations using the parametric expressions
   - Solve using algebraic manipulation (substitution, elimination, factoring, etc.)
   - Your final answer may be an algebraic expression in terms of parameters
7. If the problem has numeric values, calculate the final numerical answer with precise mathematics

INSTRUCTIONS FOR MULTI-HOP REASONING:
- Start from known values (numeric or parametric) in the knowledge graph
- Identify which theorems can be applied based on available shapes and relationships
- If no suitable theorem exists in the list, you may propose and apply a new theorem
- Chain multiple reasoning steps together if needed
- For each step, explain:
  * What theorem/property you're using
  * Why it's applicable
  * What inputs you're using (values or expressions)
  * What outputs you derive (show intermediate values or expressions)
- For parametric problems:
  * Set up equations using the parametric expressions
  * Show algebraic manipulation steps (e.g., "Substitute x into equation...", "Solve for y...")
  * Simplify expressions step by step
- Continue until you find the target value(s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: JSON FORMAT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  YOUR RESPONSE MUST BE VALID JSON ONLY - NO MARKDOWN, NO CODE BLOCKS, NO EXTRA TEXT
⚠️  DO NOT wrap JSON in ```json or ``` markers
⚠️  DO NOT add explanatory text before or after the JSON
⚠️  ENSURE all strings are properly quoted with double quotes "..."
⚠️  ENSURE all numbers are valid (not NaN, not Infinity)
⚠️  ENSURE all JSON brackets and braces are properly closed
⚠️  DO NOT use trailing commas in arrays or objects
⚠️  ESCAPE special characters in strings (backslash, quotes, etc.)

Return ONLY the following JSON object (no other text):

{{
    "problem_analysis": {{
        "given_information": ["list of known values/expressions from the problem and knowledge graph"],
        "target_variables": ["what needs to be found"],
        "relevant_shapes": ["shapes involved in solution"],
        "key_relationships": ["important geometric relationships"],
        "is_parametric": true,
        "parameters_involved": ["list of parameters if applicable"],
        "is_multiple_choice": true
    }},
    "reasoning_chain": [
        {{
            "step_number": 1,
            "reasoning_type": "theorem_application",
            "theorem_or_property_used": "name of theorem or geometric property",
            "is_new_theorem": false,
            "explanation": "clear explanation of this reasoning step",
            "inputs": {{"variable_name": "value or expression"}},
            "mathematical_work": "step-by-step calculation (numeric) or algebraic manipulation",
            "intermediate_result": "result at this step (number or expression)",
            "outputs": {{"variable_name": "value or expression"}},
            "confidence": 0.95
        }}
    ],
    "new_theorems_proposed": [
        {{
            "name": "theorem name",
            "description": "what the theorem states",
            "conditions": ["when it applies"],
            "conclusions": ["what it concludes"],
            "mathematical_form": "formula",
            "applicable_shapes": ["shape types"]
        }}
    ],
    "final_answer": {{
        "target_variables": {{
            "variable_name": {{
                "value": "numerical_value or algebraic_expression",
                "is_parametric": false,
                "units": "appropriate units",
                "derivation_summary": "how this was derived"
            }}
        }},
        "selected_choice": "A",
        "selected_choice_value": "the actual value of the selected choice",
        "choice_explanation": "brief explanation of why this choice matches the calculated answer",
        "success": true,
        "overall_confidence": 0.90
    }},
    "solution_explanation": "Complete natural language explanation of the solution process and answer"
}}

VALIDATION CHECKLIST BEFORE RESPONDING:
✓ Response starts with {{ and ends with }}
✓ All strings use double quotes "..." not single quotes
✓ All numbers are valid (no NaN, no Infinity, no undefined)
✓ All arrays [...] are properly closed
✓ All objects {{...}} are properly closed
✓ No trailing commas before closing }} or ]
✓ No markdown code blocks (```json) anywhere
✓ No explanatory text outside the JSON
✓ Boolean values are true or false (lowercase, not "true" or "false")
✓ Null values are null (lowercase, not "null" or None)

IMPORTANT REMINDERS:
- Use theorems from the available list when possible
- Propose new theorems only if necessary
- Refer to shapes and points using the IDs from the knowledge graph
- For numeric problems: Show all mathematical calculations with actual numbers
- For parametric problems: Show all algebraic manipulations and express answers in terms of parameters
- Include intermediate results at each step (numeric or algebraic)
- Ensure logical flow between reasoning steps (multi-hop)
- Be precise with geometric terminology and algebraic notation
- confidence values must be between 0.0 and 1.0
- If multiple choice, selected_choice must be one of the provided choices (A, B, C, D, E)

START YOUR RESPONSE WITH {{ AND END WITH }} - RETURN ONLY VALID JSON, NOTHING ELSE.
"""

        try:
            # Generate content with or without image
            if include_image and image_path:
                from PIL import Image
                logger.info(f"Loading image from: {image_path}")
                image = Image.open(image_path)
                # Send both prompt and image to Gemini
                response = self.gemini_model.generate_content(
                    [prompt, image],
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': max_tokens,
                    }
                )
                logger.info(f"Generated solution using image + knowledge graph ({max_tokens} tokens)")
            else:
                # Send only prompt (text-based on knowledge graph)
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': max_tokens,
                    }
                )
                logger.info(f"Generated solution using knowledge graph only ({max_tokens} tokens)")

            # Safely extract response text, handling finish_reason errors
            response_text, error_msg = self._safe_get_response_text(response)

            if response_text is None:
                logger.error(f"Failed to get response text: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg or 'Failed to extract response from Gemini API',
                    'problem_text': problem_text,
                    'graph_id': graph_id
                }

            # Check if we got MAX_TOKENS warning (partial response)
            if error_msg == "MAX_TOKENS":
                logger.warning("Received MAX_TOKENS warning - response may be incomplete")
                # Continue processing but flag for retry if parsing fails

            result = self._extract_json_from_response(response_text)

            # Validate that we got a proper response
            if not result or not isinstance(result, dict):
                logger.warning(f"Invalid LLM response format (not a dict)")
                # If this was due to MAX_TOKENS, signal retry
                if error_msg == "MAX_TOKENS":
                    return {
                        'success': False,
                        'error': 'MAX_TOKENS - Invalid JSON, retry needed',
                        'problem_text': problem_text,
                        'graph_id': graph_id
                    }
                return {
                    'success': False,
                    'error': 'Invalid JSON response from LLM',
                    'problem_text': problem_text,
                    'graph_id': graph_id
                }

            # Check if we have the minimum required fields
            if 'final_answer' not in result:
                logger.warning(f"LLM response missing 'final_answer' field")
                return {
                    'success': False,
                    'error': 'Incomplete LLM response',
                    'problem_text': problem_text,
                    'graph_id': graph_id
                }

            # Print step-by-step solution
            self._print_solution_steps(result, problem_text)

            # Store new theorems and shapes if successful
            if result.get('final_answer', {}).get('success'):
                self._store_new_knowledge(result, graph_structure, graph_id)

            # Convert to standard solution format
            return self._convert_to_solution_format(result, problem_text, graph_id)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {
                'success': False,
                'error': f'JSON parsing error: {str(e)}',
                'problem_text': problem_text,
                'graph_id': graph_id
            }
        except Exception as e:
            logger.error(f"Error in solve attempt: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'problem_text': problem_text,
                'graph_id': graph_id
            }

    def _format_complete_knowledge_graph(self, graph_structure: Dict) -> str:
        """Format complete knowledge graph for LLM comprehension including parametric values"""

        kg_parts = []

        # Parameters section (if present)
        if graph_structure.get('has_parameters') and graph_structure.get('parameters'):
            kg_parts.append("PARAMETERS:")
            for param in graph_structure['parameters']:
                param_name = param.get('parameter_name', 'unknown')
                appears_in = ', '.join(param.get('appears_in', []))
                constraints = param.get('constraints', [])
                description = param.get('description', '')
                kg_parts.append(f"  - Parameter '{param_name}': {description}")
                if appears_in:
                    kg_parts.append(f"    Appears in: {appears_in}")
                if constraints:
                    kg_parts.append(f"    Constraints: {', '.join(constraints)}")
            kg_parts.append("")

        # Shapes section with properties
        if graph_structure.get('shapes_by_type'):
            kg_parts.append("GEOMETRIC SHAPES:")
            for shape_type, shapes in graph_structure['shapes_by_type'].items():
                for shape in shapes:
                    props = shape.get('properties', {})
                    props_str = ", ".join([f"{k}={v}" for k, v in props.items()])
                    label = shape.get('label', shape['shape_id'])
                    kg_parts.append(f"  - {shape_type.upper()} (ID: {shape['shape_id']}, Label: {label})")
                    if props_str:
                        kg_parts.append(f"    Properties: {props_str}")
                    kg_parts.append(f"    Confidence: {shape.get('confidence', 'N/A')}")

        # Points section with angles
        if graph_structure.get('points'):
            kg_parts.append("\nPOINTS:")
            for point in graph_structure['points']:
                label = f" - Label: {point['label']}" if point.get('label') else ""
                kg_parts.append(f"  - Point {point['point_id']}: coordinates ({point['x']:.2f}, {point['y']:.2f}){label}")

                # Include angles at this vertex
                angles = point.get('angles', [])
                if angles:
                    kg_parts.append(f"    Angles at vertex:")
                    for angle in angles:
                        angle_label = angle.get('label', angle.get('angle_id', 'unnamed'))
                        # Handle parametric angle measures
                        is_parametric = angle.get('is_parametric', False)
                        if is_parametric:
                            measure_expr = angle.get('measure_expression', 'unknown')
                            measure_str = f"{measure_expr} (parametric)"
                        else:
                            measure = angle.get('measure')
                            measure_str = f"{measure}°" if measure else "unknown"
                        angle_type = angle.get('angle_type', 'unknown')
                        adjacent = ', '.join(angle.get('adjacent_points', []))
                        kg_parts.append(f"      * {angle_label}: {measure_str} ({angle_type}, adjacent to: {adjacent})")

        # Edges section with relationships and parametric support
        if graph_structure.get('edges'):
            kg_parts.append("\nEDGES:")
            for edge in graph_structure['edges']:
                edge_id = edge['edge_id']
                label = edge.get('label', edge_id)
                start = edge.get('start_point', 'unknown')
                end = edge.get('end_point', 'unknown')

                # Handle parametric edge lengths
                is_parametric = edge.get('is_parametric', False)
                if is_parametric:
                    length_expr = edge.get('length_expression', 'unknown')
                    length_str = f"length={length_expr} (parametric)"
                else:
                    length_str = f"length={edge['length']:.2f}" if edge.get('length') else "length=unknown"

                edge_type = edge.get('edge_type', 'segment')

                kg_parts.append(f"  - Edge {label} ({start} → {end}): {length_str}, type={edge_type}")

                # Include edge relationships
                relationships = edge.get('relationships', {})
                if relationships:
                    if relationships.get('parallel_to'):
                        kg_parts.append(f"    Parallel to: {', '.join(relationships['parallel_to'])}")
                    if relationships.get('perpendicular_to'):
                        kg_parts.append(f"    Perpendicular to: {', '.join(relationships['perpendicular_to'])}")
                    if relationships.get('equal_to'):
                        kg_parts.append(f"    Equal in length to: {', '.join(relationships['equal_to'])}")
                    if relationships.get('bisects'):
                        kg_parts.append(f"    Bisects: {relationships['bisects']}")

        # Shape relationships (containment)
        if graph_structure.get('shape_relationships'):
            kg_parts.append("\nSHAPE CONTAINMENT/HIERARCHY:")
            for rel in graph_structure['shape_relationships']:
                rel_type = rel['relationship_type'].replace('_', ' ').lower()
                kg_parts.append(f"  - {rel['shape1_id']} {rel_type} {rel['shape2_id']}")

        return "\n".join(kg_parts)

    def _format_available_theorems(self, theorems: List[Dict]) -> str:
        """Format all available theorems for LLM"""

        theorem_parts = []

        for i, theorem in enumerate(theorems, 1):
            theorem_parts.append(f"\n{i}. {theorem['theorem_name']}")
            theorem_parts.append(f"   ID: {theorem['theorem_id']}")
            theorem_parts.append(f"   Description: {theorem.get('description', 'N/A')}")
            theorem_parts.append(f"   Mathematical Form: {theorem.get('mathematical_form', 'N/A')}")
            theorem_parts.append(f"   Conditions: {theorem.get('conditions', 'N/A')}")
            theorem_parts.append(f"   Conclusions: {theorem.get('conclusions', 'N/A')}")
            theorem_parts.append(f"   Applicable to: {', '.join(theorem.get('applicable_shapes', []))}")
            theorem_parts.append(f"   Relevance Score: {theorem.get('score', 'N/A')}")

        # Add basic geometric properties
        theorem_parts.append("\n\nBASIC GEOMETRIC PROPERTIES:")
        theorem_parts.append("- Distance formula: d = √((x₂-x₁)² + (y₂-y₁)²)")
        theorem_parts.append("- Triangle area: A = ½ × base × height")
        theorem_parts.append("- Triangle perimeter: P = sum of all sides")
        theorem_parts.append("- Angle sum in triangle: sum of angles = 180°")
        theorem_parts.append("- Pythagorean theorem: a² + b² = c² (for right triangles)")
        theorem_parts.append("- Similar triangles: corresponding sides proportional")
        theorem_parts.append("- Circle area: A = πr²")
        theorem_parts.append("- Circle circumference: C = 2πr")

        return "\n".join(theorem_parts)

    def _convert_to_solution_format(self, llm_result: Dict, problem_text: str, graph_id: str) -> Dict[str, Any]:
        """Convert LLM response to standard solution format with robust error handling"""

        try:
            final_answer = llm_result.get('final_answer', {})
            reasoning_chain = llm_result.get('reasoning_chain', [])

            # Extract found variables with safe defaults
            found_variables = {}
            target_vars = final_answer.get('target_variables', {})

            if isinstance(target_vars, dict):
                for var_name, var_info in target_vars.items():
                    try:
                        if isinstance(var_info, dict):
                            found_variables[var_name] = {
                                'value': var_info.get('value'),
                                'units': var_info.get('units', ''),
                                'confidence': final_answer.get('overall_confidence', 0.8),
                                'derivation_path': [step.get('theorem_or_property_used', '') for step in reasoning_chain if isinstance(step, dict)]
                            }
                    except Exception as e:
                        logger.warning(f"Error processing variable {var_name}: {e}")
                        continue

            # Format reasoning steps with safe extraction
            reasoning_steps = []
            if isinstance(reasoning_chain, list):
                for step in reasoning_chain:
                    try:
                        if isinstance(step, dict):
                            reasoning_steps.append({
                                'step_id': f"step_{step.get('step_number', 0)}",
                                'step_type': step.get('reasoning_type', 'unknown'),
                                'theorem_used': step.get('theorem_or_property_used', ''),
                                'reasoning': step.get('explanation', ''),
                                'mathematical_expression': step.get('mathematical_work', ''),
                                'inputs': step.get('inputs', {}),
                                'outputs': step.get('outputs', {}),
                                'confidence': step.get('confidence', 0.8)
                            })
                    except Exception as e:
                        logger.warning(f"Error processing reasoning step: {e}")
                        continue

            solution_format = {
                'success': final_answer.get('success', False) if isinstance(final_answer, dict) else False,
                'completion_rate': 1.0 if (isinstance(final_answer, dict) and final_answer.get('success', False)) else 0.0,
                'problem_text': problem_text,
                'graph_id': graph_id,
                'target_variables': list(target_vars.keys()) if isinstance(target_vars, dict) else [],
                'found_variables': found_variables,
                'missing_variables': [],
                'reasoning_steps': reasoning_steps,
                'total_steps': len(reasoning_steps),
                'confidence': final_answer.get('overall_confidence', 0.0) if isinstance(final_answer, dict) else 0.0,
                'explanation': llm_result.get('solution_explanation', 'Solution completed.'),
                'problem_analysis': llm_result.get('problem_analysis', {}),
                'approach': 'single_iteration_complete_kg'
            }

            # Add multiple choice information if present with safe extraction
            if isinstance(final_answer, dict):
                selected_choice = final_answer.get('selected_choice')
                if selected_choice:
                    solution_format['selected_choice'] = selected_choice
                    solution_format['selected_choice_value'] = final_answer.get('selected_choice_value', '')
                    solution_format['choice_explanation'] = final_answer.get('choice_explanation', '')

            return solution_format

        except Exception as e:
            logger.error(f"Error converting LLM result to solution format: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal valid solution format on error
            return {
                'success': False,
                'completion_rate': 0.0,
                'problem_text': problem_text,
                'graph_id': graph_id,
                'target_variables': [],
                'found_variables': {},
                'missing_variables': [],
                'reasoning_steps': [],
                'total_steps': 0,
                'confidence': 0.0,
                'explanation': f'Error processing solution: {str(e)}',
                'problem_analysis': {},
                'approach': 'single_iteration_complete_kg',
                'error': str(e)
            }


    def _load_graph_structure(self, graph_id: str) -> Dict:
        """Load complete graph structure information with simplified schema and parametric support"""

        with self.kb.driver.session() as session:
            # Get graph metadata including parameters
            graph_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})
                RETURN g.has_parameters as has_parameters, g.parameters as parameters
            """, {'graph_id': graph_id}).single()

            has_parameters = graph_result['has_parameters'] if graph_result else False
            parameters = []
            if graph_result and graph_result['parameters']:
                try:
                    parameters = json.loads(graph_result['parameters']) if isinstance(graph_result['parameters'], str) else graph_result['parameters']
                except:
                    pass

            # Get shapes
            shapes_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s:Shape)
                RETURN s.shape_id as shape_id, s.shape_type as shape_type,
                       s.properties as properties, s.confidence as confidence, s.label as label
            """, {'graph_id': graph_id})

            shapes_by_type = defaultdict(list)
            shapes_by_id = {}

            for record in shapes_result:
                shape_info = {
                    'shape_id': record['shape_id'],
                    'shape_type': record['shape_type'],
                    'properties': json.loads(record['properties'] or '{}'),
                    'confidence': record['confidence'],
                    'label': record['label']
                }
                shapes_by_type[record['shape_type']].append(shape_info)
                shapes_by_id[record['shape_id']] = shape_info

            # Get points with angles
            points_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(p:Point)
                RETURN p.point_id as point_id, p.x as x, p.y as y,
                       p.label as label, p.angles_at_vertex as angles_json
            """, {'graph_id': graph_id})

            points = []
            for record in points_result:
                point_data = {
                    'point_id': record['point_id'],
                    'x': record['x'],
                    'y': record['y'],
                    'label': record['label'],
                    'angles': []
                }
                # Parse angles at vertex
                if record['angles_json']:
                    try:
                        angles_data = json.loads(record['angles_json']) if isinstance(record['angles_json'], str) else record['angles_json']
                        point_data['angles'] = angles_data
                    except:
                        pass
                points.append(point_data)

            # Get edges (as relationships) with parametric support
            edges_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(p1:Point)
                MATCH (p1)-[e:EDGE]->(p2:Point)
                RETURN e.edge_id as edge_id, e.label as label, e.length as length,
                       e.is_parametric as is_parametric, e.length_expression as length_expression,
                       e.edge_type as edge_type, e.parallel_to as parallel_to,
                       e.perpendicular_to as perpendicular_to, e.equal_to as equal_to,
                       p1.point_id as start_point, p2.point_id as end_point
            """, {'graph_id': graph_id})

            edges = []
            for record in edges_result:
                edge_data = {
                    'edge_id': record['edge_id'],
                    'label': record['label'],
                    'length': record['length'],
                    'is_parametric': record.get('is_parametric', False),
                    'length_expression': record.get('length_expression'),
                    'edge_type': record['edge_type'],
                    'start_point': record['start_point'],
                    'end_point': record['end_point'],
                    'relationships': {}
                }
                # Parse relationship arrays
                if record['parallel_to']:
                    try:
                        edge_data['relationships']['parallel_to'] = json.loads(record['parallel_to']) if isinstance(record['parallel_to'], str) else record['parallel_to']
                    except:
                        pass
                if record['perpendicular_to']:
                    try:
                        edge_data['relationships']['perpendicular_to'] = json.loads(record['perpendicular_to']) if isinstance(record['perpendicular_to'], str) else record['perpendicular_to']
                    except:
                        pass
                if record['equal_to']:
                    try:
                        edge_data['relationships']['equal_to'] = json.loads(record['equal_to']) if isinstance(record['equal_to'], str) else record['equal_to']
                    except:
                        pass
                edges.append(edge_data)

            # Get shape relationships (containment)
            shape_relationships_result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s1:Shape)
                OPTIONAL MATCH (s1)-[r:CONTAINS_SHAPE|CONTAINED_IN]->(s2:Shape)
                RETURN s1.shape_id as shape1_id, s2.shape_id as shape2_id,
                       type(r) as relationship_type
            """, {'graph_id': graph_id})

            shape_relationships = []
            for record in shape_relationships_result:
                if record['shape2_id']:
                    shape_relationships.append({
                        'shape1_id': record['shape1_id'],
                        'shape2_id': record['shape2_id'],
                        'relationship_type': record['relationship_type']
                    })

        return {
            'shapes_by_type': dict(shapes_by_type),
            'shapes_by_id': shapes_by_id,
            'shape_relationships': shape_relationships,
            'points': points,
            'edges': edges,
            'has_parameters': has_parameters,
            'parameters': parameters
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

    def _print_solution_steps(self, result: Dict, problem_text: str):
        """Print step-by-step solution with intermediate results"""

        print("\n" + "="*80)
        print("GEOMETRIC PROBLEM SOLUTION")
        print("="*80)

        print(f"\nPROBLEM: {problem_text}\n")

        # Print problem analysis
        analysis = result.get('problem_analysis', {})
        if analysis:
            print("PROBLEM ANALYSIS:")
            print(f"   Given: {', '.join(analysis.get('given_information', []))}")
            print(f"   Find: {', '.join(analysis.get('target_variables', []))}")
            print(f"   Shapes: {', '.join(analysis.get('relevant_shapes', []))}")
            if analysis.get('key_relationships'):
                print(f"   Key Relationships: {', '.join(analysis.get('key_relationships', []))}")

        # Print reasoning steps
        print("\nSOLUTION STEPS:\n")
        reasoning_chain = result.get('reasoning_chain', [])

        for step in reasoning_chain:
            step_num = step.get('step_number', 0)
            theorem = step.get('theorem_or_property_used', 'Unknown')
            is_new = step.get('is_new_theorem', False)
            theorem_label = f"{theorem} {'[NEW]' if is_new else ''}"

            print(f"   Step {step_num}: {theorem_label}")
            print(f"   └─ {step.get('explanation', '')}")

            # Print inputs
            inputs = step.get('inputs', {})
            if inputs:
                inputs_str = ", ".join([f"{k}={v}" for k, v in inputs.items()])
                print(f"      Inputs: {inputs_str}")

            # Print mathematical work
            math_work = step.get('mathematical_work', '')
            if math_work:
                print(f"      Calculation: {math_work}")

            # Print intermediate result
            intermediate = step.get('intermediate_result', '')
            if intermediate:
                print(f"      ➜ Intermediate Result: {intermediate}")

            # Print outputs
            outputs = step.get('outputs', {})
            if outputs:
                outputs_str = ", ".join([f"{k}={v}" for k, v in outputs.items()])
                print(f"      ✓ Output: {outputs_str}")

            print(f"      Confidence: {step.get('confidence', 0):.2%}\n")

        # Print new theorems proposed
        new_theorems = result.get('new_theorems_proposed', [])
        if new_theorems:
            print("NEW THEOREMS PROPOSED:")
            for theorem in new_theorems:
                print(f"   • {theorem.get('name', 'Unnamed')}")
                print(f"     Description: {theorem.get('description', '')}")
                print(f"     Applies to: {', '.join(theorem.get('applicable_shapes', []))}")
                print(f"     Formula: {theorem.get('mathematical_form', '')}\n")

        # Print final answer
        print("="*80)
        print("FINAL ANSWER:")
        print("="*80)

        final_answer = result.get('final_answer', {})
        target_vars = final_answer.get('target_variables', {})

        if target_vars:
            for var_name, var_info in target_vars.items():
                value = var_info.get('value', 'N/A')
                units = var_info.get('units', '')
                derivation = var_info.get('derivation_summary', '')

                print(f"\n   {var_name} = {value} {units}")
                if derivation:
                    print(f"   (Derived by: {derivation})")

        # Print selected choice for multiple choice questions
        selected_choice = final_answer.get('selected_choice')
        if selected_choice:
            choice_value = final_answer.get('selected_choice_value', '')
            choice_explanation = final_answer.get('choice_explanation', '')
            print(f"\n   📝 Selected Answer: {selected_choice}")
            if choice_value:
                print(f"   Value: {choice_value}")
            if choice_explanation:
                print(f"   Reasoning: {choice_explanation}")

        success = final_answer.get('success', False)
        confidence = final_answer.get('overall_confidence', 0)

        print(f"\n   Success: {'✓ Yes' if success else '✗ No'}")
        print(f"   Overall Confidence: {confidence:.2%}")

        # Print explanation
        explanation = result.get('solution_explanation', '')
        if explanation:
            print(f"\nEXPLANATION:")
            print(f"   {explanation}")

        print("\n" + "="*80 + "\n")

    def _store_new_knowledge(self, result: Dict, graph_structure: Dict, graph_id: str):
        """Store new theorems and shapes in knowledge base if solution was successful"""

        try:
            # Extract shapes from graph structure if not already in KB
            shapes_in_problem = set()
            for shape_type in graph_structure.get('shapes_by_type', {}).keys():
                shapes_in_problem.add(shape_type.lower())

            # Store new shapes
            with self.kb.driver.session() as session:
                for shape_name in shapes_in_problem:
                    # Check if shape exists
                    existing_shape = session.run(
                        "MATCH (s:Shape {name: $name}) RETURN s",
                        {'name': shape_name}
                    ).single()

                    if not existing_shape:
                        session.run("""
                            CREATE (s:Shape {
                                name: $name,
                                created_at: datetime(),
                                source: 'dynamic_learning'
                            })
                        """, {'name': shape_name})
                        logger.info(f"✓ Added new shape to KB: {shape_name}")

            # Store new theorems
            new_theorems = result.get('new_theorems_proposed', [])
            if new_theorems:
                with self.kb.driver.session() as session:
                    for theorem_data in new_theorems:
                        theorem_name = theorem_data.get('name', '')
                        if not theorem_name:
                            continue

                        # Check if theorem exists
                        existing_theorem = session.run(
                            "MATCH (t:Theorem {name: $name}) RETURN t",
                            {'name': theorem_name}
                        ).single()

                        if not existing_theorem:
                            # Create new theorem
                            theorem_id = hashlib.md5(f"{theorem_name}_{theorem_data.get('description', '')}".encode()).hexdigest()[:12]

                            session.run("""
                                CREATE (t:Theorem {
                                    theorem_id: $theorem_id,
                                    name: $name,
                                    description: $description,
                                    conditions: $conditions,
                                    conclusions: $conclusions,
                                    mathematical_form: $mathematical_form,
                                    applicable_shapes: $applicable_shapes,
                                    confidence: 0.9,
                                    usage_count: 1,
                                    success_rate: 1.0,
                                    source_problems: [$graph_id],
                                    created_at: datetime(),
                                    source: 'dynamic_learning'
                                })
                            """, {
                                'theorem_id': theorem_id,
                                'name': theorem_name,
                                'description': theorem_data.get('description', ''),
                                'conditions': theorem_data.get('conditions', []),
                                'conclusions': theorem_data.get('conclusions', []),
                                'mathematical_form': theorem_data.get('mathematical_form', ''),
                                'applicable_shapes': theorem_data.get('applicable_shapes', []),
                                'graph_id': graph_id
                            })

                            # Link theorem to applicable shapes
                            for shape_name in theorem_data.get('applicable_shapes', []):
                                session.run("""
                                    MATCH (s:Shape {name: $shape_name})
                                    MATCH (t:Theorem {theorem_id: $theorem_id})
                                    MERGE (s)-[:APPLICABLE_TO]->(t)
                                """, {
                                    'shape_name': shape_name.lower(),
                                    'theorem_id': theorem_id
                                })

                            logger.info(f"✓ Added new theorem to KB: {theorem_name}")
                            print(f"   Stored new theorem in Knowledge Base: {theorem_name}")
                        else:
                            # Update usage count
                            session.run("""
                                MATCH (t:Theorem {name: $name})
                                SET t.usage_count = t.usage_count + 1,
                                    t.source_problems = t.source_problems + [$graph_id],
                                    t.updated_at = datetime()
                            """, {'name': theorem_name, 'graph_id': graph_id})
                            logger.info(f"✓ Updated theorem usage: {theorem_name}")

        except Exception as e:
            logger.error(f"Error storing new knowledge: {e}")

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from LLM response with robust error handling"""

        if not response_text:
            logger.error("Empty response text received")
            return {}

        try:
            # Remove any leading/trailing whitespace
            cleaned_text = response_text.strip()

            # Method 1: Try parsing the entire response as-is
            try:
                result = json.loads(cleaned_text)
                if isinstance(result, dict):
                    logger.info("Successfully parsed JSON directly")
                    return result
            except json.JSONDecodeError:
                pass

            # Method 2: Remove markdown code blocks if present (despite instructions)
            # Remove ```json and ``` markers
            if "```json" in cleaned_text or "```" in cleaned_text:
                logger.warning("Found markdown code blocks in response (should not happen)")
                cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
                cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                cleaned_text = cleaned_text.strip()

                try:
                    result = json.loads(cleaned_text)
                    if isinstance(result, dict):
                        logger.info("Successfully parsed JSON after removing markdown")
                        return result
                except json.JSONDecodeError:
                    pass

            # Method 3: Find the first { to last } and extract that
            first_brace = cleaned_text.find('{')
            last_brace = cleaned_text.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = cleaned_text[first_brace:last_brace + 1]

                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        logger.info("Successfully parsed JSON by extracting braces")
                        return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON extraction by braces failed: {e}")

                    # Method 4: Try to fix common JSON issues
                    # Remove trailing commas before } or ]
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)

                    # Replace single quotes with double quotes (common mistake)
                    # But be careful not to replace quotes inside strings
                    # This is a simple heuristic - may not work in all cases
                    if "'" in fixed_json and '"' not in fixed_json:
                        fixed_json = fixed_json.replace("'", '"')

                    # Replace Python-style boolean/null values
                    fixed_json = re.sub(r'\bTrue\b', 'true', fixed_json)
                    fixed_json = re.sub(r'\bFalse\b', 'false', fixed_json)
                    fixed_json = re.sub(r'\bNone\b', 'null', fixed_json)

                    try:
                        result = json.loads(fixed_json)
                        if isinstance(result, dict):
                            logger.info("Successfully parsed JSON after repairs")
                            return result
                    except json.JSONDecodeError:
                        pass

            # If all methods fail, log the error with a preview of the response
            logger.error("All JSON extraction methods failed")
            logger.error(f"Response preview (first 500 chars): {response_text[:500]}")
            logger.error(f"Response preview (last 500 chars): {response_text[-500:]}")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error in JSON extraction: {e}")
            logger.error(f"Response text length: {len(response_text)}")
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
            gemini_api_key=config['gemini_api_key']
        )

        self.image_processor = GeometricImageProcessor(config)
        self.reasoner = ScalableLLMReasoner(self.kb, self.image_processor)

    def solve_geometric_problem(self, image_path: str, problem_text: str,
                               problem_id: Optional[str] = None, choices: Optional[List[str]] = None,
                               include_image: bool = False) -> Dict[str, Any]:
        """Complete scalable pipeline for geometric problem solving

        Args:
            image_path: Path to the geometric diagram image
            problem_text: The problem statement
            problem_id: Optional identifier for the problem
            choices: Optional list of multiple choice options
            include_image: If True, includes the image along with knowledge graph in the prompt
        """

        try:
            logger.info(f"Solving geometric problem with scalable LLM approach: {problem_id or 'unnamed'}")
            logger.info(f"Include image in reasoning: {include_image}")

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
            solution = self.reasoner.solve_problem(
                problem_text,
                graph_id,
                choices=choices,
                include_image=include_image,
                image_path=image_path if include_image else None
            )

            # Step 3: Enhance solution with graph context
            solution.update({
                'image_path': image_path,
                'graph_summary': graph_result['summary'],
                'available_theorems': graph_result['applicable_theorems'],
                'shape_relationships': graph_result['shape_relationships'],
                'approach': 'scalable_llm_reasoning',
                'image_included_in_prompt': include_image
            })

            logger.info(f"Scalable problem solving completed. Success: {solution['success']}, "
                       f"Completion: {solution.get('completion_rate', 0):.2%}")

            return solution

        except Exception as e:
            import traceback
            traceback.print_exc(e)
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
    GEMINI_API_KEY = ""

    # Configuration
    config = {
        'neo4j_uri': NEO4J_URI,
        'neo4j_user': NEO4J_USER,
        'neo4j_password': NEO4J_PASSWORD,
        'gemini_api_key': GEMINI_API_KEY
    }

    # Initialize scalable solver
    solver = ScalableGeometricProblemSolver(config)

    try:
        # Example 1: Complex triangle problem with multiple choice
        complex_problem = {
            'image_path': "images/geo3k/train/1/img_diagram.png",
            'problem_text': "Use parallelogram MNPR to find y.",
            'problem_id': "complex_trapezoid",
            'choices': ["10", "15", "20", "25"]  # Multiple choice options
        }

        result = solver.solve_geometric_problem(**complex_problem)
        print("Complex Problem Result:")
        print(f"Success: {result['success']}")
        print(f"Completion Rate: {result.get('completion_rate', 0):.2%}")
        print(f"Steps taken: {result.get('total_steps', 0)}")
        if result.get('selected_choice'):
            print(f"Selected Answer: {result.get('selected_choice')}")
        print(f"Explanation: {result.get('explanation', 'N/A')[:200]}...")
        print()



    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        solver.close()

if __name__ == "__main__":
    main()
