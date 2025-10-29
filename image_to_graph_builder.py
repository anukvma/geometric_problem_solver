"""
Geometric Graph Constructor - Part 2 Implementation (Gemini Version)
Uses Google Gemini Vision API to process geometric images and creates Neo4j graph representations
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import json
from io import BytesIO
import os
import re

# Core dependencies
import neo4j
from neo4j import GraphDatabase
import google.generativeai as genai
from PIL import Image
import numpy as np
import traceback

# Import from Part 1 (Gemini version)
from knowledge_base_builder import GeometricKnowledgeBase, GeometricKnowledgeBaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeometricPoint:
    """Represents a point in the geometric image"""
    x: float
    y: float
    point_id: str
    label: Optional[str] = None
    confidence: float = 1.0

@dataclass
class GeometricLine:
    """Represents a line segment in the geometric image"""
    start_point_id: str
    end_point_id: str
    line_id: str
    length: Optional[float] = None
    label: Optional[str] = None
    confidence: float = 1.0

@dataclass
class GeometricAngle:
    """Represents an angle formed by two lines"""
    vertex_id: str
    arm1_id: str
    arm2_id: str
    angle_id: str
    measure: Optional[float] = None
    angle_type: Optional[str] = None  # acute, right, obtuse, straight
    label: Optional[str] = None
    confidence: float = 1.0

@dataclass
class DetectedShape:
    """Represents a detected geometric shape"""
    shape_id: str
    shape_type: str  # triangle, circle, rectangle, etc.
    vertex_ids: List[str]
    edge_ids: List[str]
    properties: Dict[str, Any]
    confidence: float = 1.0
    label: Optional[str] = None

class GPTVisionGraphExtractor:
    """Uses Google Gemini Vision to extract graph structure from geometric images"""

    def __init__(self, gemini_model, model_name: str = "gemini-2.5-flash"):
        self.gemini_model = gemini_model
        self.model_name = model_name
        logger.info(f"Initialized Gemini Vision Graph Extractor with model: {model_name}")

    def extract_graph_structure(self, image_path: str) -> Dict[str, Any]:
        """Extract complete graph structure from geometric image using Gemini Vision

        Handles large responses by increasing token limit and detecting truncation.
        """

        logger.info(f"Extracting graph structure from: {os.path.basename(image_path)}")

        # Load image as PIL Image
        pil_image = self._load_image(image_path)

        # Try extraction with progressively larger token limits
        max_attempts = 2
        token_limits = [32000, 65000]  # Progressive increase

        for attempt in range(max_attempts):
            try:
                logger.info(f"Extraction attempt {attempt + 1}/{max_attempts} with {token_limits[attempt]} tokens")

                # Extract geometric elements using Gemini Vision
                prompt = self._create_graph_extraction_prompt()

                # Generate content with Gemini
                response = self.gemini_model.generate_content(
                    [prompt, pil_image],
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': token_limits[attempt],
                    }
                )

                result_text = response.text.strip()

                # Check if response was truncated
                if self._is_response_truncated(result_text):
                    logger.warning(f"Response appears truncated at {len(result_text)} characters")

                    if attempt < max_attempts - 1:
                        logger.info("Retrying with higher token limit...")
                        continue
                    else:
                        logger.warning("Max token limit reached, attempting to extract partial graph...")
                        # Try to fix truncated JSON
                        result_text = self._fix_truncated_json(result_text)

                # Extract JSON from response
                graph_data = self._extract_json_from_response(result_text)

                if not graph_data or self._is_graph_incomplete(graph_data):
                    if attempt < max_attempts - 1:
                        logger.warning("Graph data incomplete or invalid, retrying...")
                        continue
                    else:
                        logger.error("Could not extract complete graph after all attempts")
                        return self._create_fallback_extraction(pil_image, image_path)

                # Validate and structure the graph data
                structured_graph = self._structure_graph_data(graph_data)

                logger.info(f"Successfully extracted graph structure:")
                logger.info(f"  - Points: {len(structured_graph.get('points', []))}")
                logger.info(f"  - Edges: {len(structured_graph.get('edges', []))}")
                logger.info(f"  - Shapes: {len(structured_graph.get('shapes', []))}")

                return structured_graph

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    continue
                else:
                    logger.error("All extraction attempts failed")
                    return self._create_fallback_extraction(pil_image, image_path)

            except Exception as e:
                logger.error(f"Error extracting graph structure on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    continue
                else:
                    return {
                        'points': [],
                        'edges': [],
                        'shapes': [],
                        'error': str(e)
                    }

        # Should not reach here, but return empty graph as fallback
        return {
            'points': [],
            'edges': [],
            'shapes': [],
            'error': 'Extraction failed after all attempts'
        }

    def _is_response_truncated(self, response_text: str) -> bool:
        """Detect if the response was truncated

        Checks for common signs of truncation:
        - Incomplete JSON (unclosed braces/brackets)
        - Ends mid-word or mid-sentence
        - Missing closing markers
        """
        # Check for balanced braces and brackets
        open_braces = response_text.count('{')
        close_braces = response_text.count('}')
        open_brackets = response_text.count('[')
        close_brackets = response_text.count(']')

        if open_braces != close_braces or open_brackets != close_brackets:
            return True

        # Check if ends with incomplete JSON structure
        truncation_indicators = [
            ',\n',  # Ends with comma (likely more items expected)
            '",',   # Ends with quoted value and comma
            ': "',  # Ends with key and opening quote
            ': [',  # Ends with key and opening bracket
            ': {',  # Ends with key and opening brace
        ]

        stripped = response_text.strip()
        for indicator in truncation_indicators:
            if stripped.endswith(indicator):
                return True

        return False

    def _fix_truncated_json(self, response_text: str) -> str:
        """Attempt to fix truncated JSON by closing open structures

        This is a best-effort approach to salvage partial data.
        """
        logger.info("Attempting to fix truncated JSON...")

        # Count unclosed structures
        open_braces = response_text.count('{')
        close_braces = response_text.count('}')
        open_brackets = response_text.count('[')
        close_brackets = response_text.count(']')

        # Remove trailing incomplete content (after last complete item)
        # Find the last complete object/array
        last_complete_pos = max(
            response_text.rfind('}'),
            response_text.rfind(']'),
            response_text.rfind('",')
        )

        if last_complete_pos > 0:
            # Truncate to last complete element
            response_text = response_text[:last_complete_pos + 1]

            # Recount after truncation
            open_braces = response_text.count('{')
            close_braces = response_text.count('}')
            open_brackets = response_text.count('[')
            close_brackets = response_text.count(']')

        # Close unclosed brackets and braces
        for _ in range(open_brackets - close_brackets):
            response_text += ']'
        for _ in range(open_braces - close_braces):
            response_text += '}'

        logger.info(f"Fixed JSON length: {len(response_text)} characters")
        return response_text

    def _is_graph_incomplete(self, graph_data: Dict) -> bool:
        """Check if extracted graph data is incomplete or invalid"""
        if not graph_data:
            return True

        # Check if essential fields are present
        has_points = 'points' in graph_data and len(graph_data['points']) > 0
        has_edges = 'edges' in graph_data
        has_shapes = 'shapes' in graph_data

        # At minimum, should have points
        if not has_points:
            logger.warning("Graph missing points")
            return True

        # Check if data structure is valid
        try:
            for point in graph_data.get('points', []):
                if 'point_id' not in point and 'label' not in point:
                    logger.warning("Point missing required fields")
                    return True

            for edge in graph_data.get('edges', []):
                if 'start_point' not in edge or 'end_point' not in edge:
                    logger.warning("Edge missing required fields")
                    return True

            for shape in graph_data.get('shapes', []):
                if 'shape_type' not in shape:
                    logger.warning("Shape missing required fields")
                    return True

        except (KeyError, TypeError) as e:
            logger.warning(f"Invalid graph structure: {e}")
            return True

        return False

    def _create_fallback_extraction(self, pil_image: Image.Image, image_path: str) -> Dict[str, Any]:
        """Create a simplified extraction when full extraction fails

        Uses a simpler prompt to extract basic geometric elements.
        """
        logger.warning("Using fallback extraction with simplified prompt")

        simplified_prompt = """
        Analyze this geometric diagram and extract ONLY the essential elements.

        Return a JSON object with:
        {
            "points": [{"point_id": "A", "label": "A", "approximate_position": {"x": 0.5, "y": 0.5}}],
            "edges": [{"edge_id": "AB", "start_point": "A", "end_point": "B", "label": "AB"}],
            "shapes": [{"shape_id": "shape_1", "shape_type": "triangle", "vertices": ["A", "B", "C"]}]
        }

        Focus on:
        1. Identifying all visible points (vertices)
        2. All line segments connecting points
        3. Main shapes (triangles, circles, rectangles)

        Keep it simple. Return ONLY the JSON, no extra text.
        """

        try:
            response = self.gemini_model.generate_content(
                [simplified_prompt, pil_image],
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 8000,
                }
            )

            result_text = response.text.strip()
            graph_data = self._extract_json_from_response(result_text)

            if graph_data:
                structured_graph = self._structure_graph_data(graph_data)
                logger.info("Fallback extraction succeeded")
                return structured_graph

        except Exception as e:
            logger.error(f"Fallback extraction also failed: {e}")

        # Return minimal graph structure
        return {
            'points': [],
            'edges': [],
            'shapes': [],
            'error': 'Both primary and fallback extraction failed'
        }

    def _create_graph_extraction_prompt(self) -> str:
        """Create comprehensive prompt for graph extraction"""

        return """
        Analyze this geometric diagram and extract its complete graph structure. Identify ALL geometric elements with precise details.

        Return a JSON object with the following structure:
        {
            "points": [
                {
                    "point_id": "A",
                    "label": "A",
                    "description": "vertex of triangle at top",
                    "approximate_position": {"x": 0.5, "y": 0.2},
                    "angles_at_vertex": [
                        {
                            "angle_id": "angle_ABC",
                            "adjacent_points": ["B", "C"],
                            "measure": 60.0,
                            "measure_expression": null,
                            "is_parametric": false,
                            "angle_type": "acute",
                            "label": "∠ABC"
                        }
                    ]
                }
            ],
            "edges": [
                {
                    "edge_id": "AB",
                    "start_point": "A",
                    "end_point": "B",
                    "label": "AB",
                    "length": 5.0,
                    "length_expression": null,
                    "is_parametric": false,
                    "description": "side of triangle",
                    "edge_type": "segment",
                    "relationships": {
                        "parallel_to": ["CD"],
                        "perpendicular_to": ["BC"],
                        "equal_to": ["DE"],
                        "bisects": null
                    }
                }
            ],
            "shapes": [
                {
                    "shape_id": "triangle_ABC",
                    "shape_type": "triangle",
                    "vertices": ["A", "B", "C"],
                    "edges": ["AB", "BC", "CA"],
                    "properties": {
                        "triangle_type": "right",
                        "side_lengths": {"AB": 3, "BC": 4, "CA": 5},
                        "angles": {"A": 90, "B": 60, "C": 30},
                        "special_properties": ["right_angled_at_A", "hypotenuse_CA"]
                    },
                    "label": "△ABC",
                    "contains_shapes": [],
                    "contained_in": null
                }
            ],
            "measurements": [
                {
                    "type": "length",
                    "element": "AB",
                    "value": 5.0,
                    "expression": null,
                    "is_parametric": false,
                    "unit": "units"
                },
                {
                    "type": "angle",
                    "element": "angle_ABC",
                    "value": 60.0,
                    "expression": null,
                    "is_parametric": false,
                    "unit": "degrees"
                }
            ],
            "parameters": [
                {
                    "parameter_name": "x",
                    "appears_in": ["AB", "BC"],
                    "constraints": ["x > 0"],
                    "description": "parameter representing variable length"
                }
            ]
        }

        IMPORTANT GUIDELINES:
        1. **Points**: Use clear labels (A, B, C). Include ALL angles at each vertex within the point data
        2. **Edges**: Represent ALL line segments as edges connecting points. Include:
           - Length if visible (numeric or parametric)
           - If length is parametric (e.g., "2x", "y+3", "a/2"), set is_parametric=true and store expression
           - If numeric (e.g., "5", "10.5"), set is_parametric=false and store numeric value
           - Edge type (segment, ray, line, arc for circles)
           - Relationships with OTHER edges (parallel, perpendicular, equal length, bisects)
        3. **Shapes**: Identify ALL shapes including:
           - Primary shapes (triangles, circles, rectangles, parallelograms, etc.)
           - Nested shapes (if a triangle is inside a circle, note both)
           - Use "contains_shapes" for shapes within this shape
           - Use "contained_in" for parent shape
        4. **Position coordinates**: Use normalized values (0.0 to 1.0)
        5. **Measurements**: Include ALL visible measurements (numeric or parametric)
           - For numeric values: set value field, is_parametric=false, expression=null
           - For parametric values: set expression field, is_parametric=true, value=null
        6. **Parameters**: If the diagram contains algebraic parameters or variables:
           - List all unique parameters (x, y, a, b, etc.)
           - Note where each parameter appears
           - Include any visible constraints or equations
        7. **Special properties**: Note midpoints, medians, altitudes, angle bisectors, centroids, etc.
        8. **Complex diagrams**: If multiple shapes overlap or one contains another:
           - List each shape separately
           - Use contains_shapes and contained_in to show hierarchy
        9. **Precision**:
           - Be precise with numerical values shown in the image
           - Accurately capture algebraic expressions exactly as shown (e.g., "2x", "x+5", "y/2")
        10. **Missing values**: If not visible, set to null

        EXAMPLES OF PARAMETRIC VALUES:
        - Edge with length "2x": {"length": null, "length_expression": "2*x", "is_parametric": true}
        - Edge with length "5": {"length": 5.0, "length_expression": null, "is_parametric": false}
        - Angle with measure "θ": {"measure": null, "measure_expression": "theta", "is_parametric": true}
        - Angle with measure "30°": {"measure": 30.0, "measure_expression": null, "is_parametric": false}

        Return ONLY the JSON object, no additional text.
        """

    def _structure_graph_data(self, raw_data: Dict) -> Dict[str, Any]:
        """Structure and validate the extracted graph data"""

        structured = {
            'points': [],
            'edges': [],
            'shapes': [],
            'measurements': raw_data.get('measurements', []),
            'parameters': raw_data.get('parameters', []),
            'has_parameters': len(raw_data.get('parameters', [])) > 0
        }

        # Process points with angles at vertex
        for point_data in raw_data.get('points', []):
            point = GeometricPoint(
                x=point_data.get('approximate_position', {}).get('x', 0.0),
                y=point_data.get('approximate_position', {}).get('y', 0.0),
                point_id=point_data.get('point_id', point_data.get('label', f"P_{len(structured['points'])}")),
                label=point_data.get('label'),
                confidence=0.9
            )
            # Store angles at this vertex in point metadata
            point.angles_at_vertex = point_data.get('angles_at_vertex', [])
            structured['points'].append(point)

        # Process edges (lines) with relationships and parametric support
        for edge_data in raw_data.get('edges', []):
            edge = GeometricLine(
                start_point_id=edge_data.get('start_point', ''),
                end_point_id=edge_data.get('end_point', ''),
                line_id=edge_data.get('edge_id', edge_data.get('label', f"E_{len(structured['edges'])}")),
                length=edge_data.get('length'),
                label=edge_data.get('label'),
                confidence=0.9
            )
            # Store edge type and relationships
            edge.edge_type = edge_data.get('edge_type', 'segment')
            edge.relationships = edge_data.get('relationships', {})
            # Store parametric information
            edge.is_parametric = edge_data.get('is_parametric', False)
            edge.length_expression = edge_data.get('length_expression')
            structured['edges'].append(edge)

        # Process shapes with containment hierarchy
        for shape_data in raw_data.get('shapes', []):
            shape = DetectedShape(
                shape_id=shape_data.get('shape_id', f"S_{len(structured['shapes'])}"),
                shape_type=shape_data.get('shape_type', 'unknown'),
                vertex_ids=shape_data.get('vertices', []),
                edge_ids=shape_data.get('edges', []),
                properties=shape_data.get('properties', {}),
                label=shape_data.get('label'),
                confidence=0.85
            )
            # Store containment information
            shape.contains_shapes = shape_data.get('contains_shapes', [])
            shape.contained_in = shape_data.get('contained_in')
            structured['shapes'].append(shape)

        return structured

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image as PIL Image and resize if necessary"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from Gemini response with improved handling for large responses"""

        # Log response length for debugging
        logger.info(f"Response length: {len(response_text)} characters")

        try:
            # First, try to parse the entire response as JSON
            try:
                return json.loads(response_text.strip())
            except json.JSONDecodeError:
                pass

            # If that fails, look for JSON within the response
            # Try to find the outermost JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.info(f"Extracted JSON length: {len(json_str)} characters")

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse extracted JSON: {e}")

                    # Try to fix common JSON issues
                    # Remove trailing commas
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)

                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        logger.error("Could not fix JSON even after cleanup")

            # If all else fails, try extracting with code block markers
            code_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Last resort: try to extract at least the points, edges, shapes arrays
            logger.warning("Attempting partial JSON extraction...")
            partial_data = self._extract_partial_json(response_text)
            if partial_data:
                return partial_data

            logger.error("All JSON extraction methods failed")
            return {}

        except Exception as e:
            logger.error(f"Unexpected error in JSON extraction: {e}")
            logger.error(f"Response text preview: {response_text[:500]}")
            return {}

    def _extract_partial_json(self, response_text: str) -> Dict:
        """Extract partial JSON data when full parsing fails

        Attempts to extract individual arrays (points, edges, shapes) separately.
        """
        partial_data = {}

        try:
            # Extract points array
            points_match = re.search(r'"points"\s*:\s*(\[.*?\])', response_text, re.DOTALL)
            if points_match:
                try:
                    partial_data['points'] = json.loads(points_match.group(1))
                    logger.info(f"Extracted {len(partial_data['points'])} points")
                except:
                    pass

            # Extract edges array
            edges_match = re.search(r'"edges"\s*:\s*(\[.*?\])', response_text, re.DOTALL)
            if edges_match:
                try:
                    partial_data['edges'] = json.loads(edges_match.group(1))
                    logger.info(f"Extracted {len(partial_data['edges'])} edges")
                except:
                    pass

            # Extract shapes array
            shapes_match = re.search(r'"shapes"\s*:\s*(\[.*?\])', response_text, re.DOTALL)
            if shapes_match:
                try:
                    partial_data['shapes'] = json.loads(shapes_match.group(1))
                    logger.info(f"Extracted {len(partial_data['shapes'])} shapes")
                except:
                    pass

            # Extract measurements array
            measurements_match = re.search(r'"measurements"\s*:\s*(\[.*?\])', response_text, re.DOTALL)
            if measurements_match:
                try:
                    partial_data['measurements'] = json.loads(measurements_match.group(1))
                except:
                    pass

            # Extract parameters array
            parameters_match = re.search(r'"parameters"\s*:\s*(\[.*?\])', response_text, re.DOTALL)
            if parameters_match:
                try:
                    partial_data['parameters'] = json.loads(parameters_match.group(1))
                except:
                    pass

        except Exception as e:
            logger.error(f"Error in partial extraction: {e}")

        return partial_data if partial_data else None

class GeometricGraphBuilder:
    """Builds Neo4j graph representation using Gemini Vision analysis"""

    def __init__(self, knowledge_base: GeometricKnowledgeBase):
        self.kb = knowledge_base
        self.vision_extractor = GPTVisionGraphExtractor(knowledge_base.gemini_model)

    def process_image_to_graph(self, image_path: str, problem_id: Optional[str] = None) -> str:
        """Main method to process image and create graph representation using Gemini Vision"""

        # Generate unique graph ID
        graph_id = problem_id or f"graph_{hash(image_path)}_{int(datetime.now().timestamp())}"

        # Check if graph already exists
        if self._graph_exists(graph_id):
            logger.info(f"Graph {graph_id} already exists in Neo4j. Skipping extraction.")
            return graph_id

        logger.info(f"Processing image: {image_path}")

        # Extract graph structure using Gemini Vision
        graph_structure = self.vision_extractor.extract_graph_structure(image_path)

        if not graph_structure or 'error' in graph_structure:
            logger.error(f"Failed to extract graph structure from image")
            return None

        # Create graph in Neo4j
        self._create_graph_in_neo4j(graph_id, graph_structure, image_path)

        # Attach relevant theorems from knowledge base
        self._attach_theorems_to_graph(graph_id, graph_structure)

        logger.info(f"Created graph representation: {graph_id}")
        return graph_id

    def _graph_exists(self, graph_id: str) -> bool:
        """Check if a graph with the given ID already exists in Neo4j"""
        with self.kb.driver.session() as session:
            result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})
                RETURN count(g) as count
            """, {'graph_id': graph_id})

            record = result.single()
            exists = record['count'] > 0 if record else False

            if exists:
                logger.info(f"Found existing graph: {graph_id}")

            return exists

    def _create_graph_in_neo4j(self, graph_id: str, graph_structure: Dict, image_path: str):
        """Create simplified graph structure in Neo4j - edges as relationships, angles in points, parameters supported

        All node IDs are namespaced with graph_id to ensure isolation between different problems.
        """

        with self.kb.driver.session() as session:
            # Create main graph node with parameter information
            has_parameters = graph_structure.get('has_parameters', False)
            parameters = json.dumps(graph_structure.get('parameters', []))

            session.run("""
                CREATE (g:GeometricGraph {
                    graph_id: $graph_id,
                    image_path: $image_path,
                    created_at: datetime(),
                    extraction_method: 'Gemini_Vision',
                    has_parameters: $has_parameters,
                    parameters: $parameters
                })
            """, {
                'graph_id': graph_id,
                'image_path': image_path,
                'has_parameters': has_parameters,
                'parameters': parameters
            })

            # Create points with angles as properties (IDs are namespaced with graph_id)
            for point in graph_structure.get('points', []):
                self._create_point_node(session, graph_id, point)

            # Create edges as relationships between points (IDs are namespaced with graph_id)
            for edge in graph_structure.get('edges', []):
                self._create_edge_relationship(session, graph_id, edge)

            # Create shapes (IDs are namespaced with graph_id)
            for shape in graph_structure.get('shapes', []):
                self._create_shape_node(session, graph_id, shape)

            # Create shape containment relationships
            self._create_shape_containment(session, graph_id, graph_structure.get('shapes', []))

    def _namespace_id(self, graph_id: str, local_id: str) -> str:
        """Namespace a local ID with graph_id to ensure global uniqueness"""
        return f"{graph_id}_{local_id}"

    def _create_point_node(self, session, graph_id: str, point: GeometricPoint):
        """Create a point node with angles as properties

        Point IDs are namespaced with graph_id to prevent conflicts between different problems.
        The original point_id is preserved in the 'label' field for readability.
        """
        # Namespace the point_id for global uniqueness
        namespaced_id = self._namespace_id(graph_id, point.point_id)
        original_label = point.label or point.point_id

        session.run("""
            MATCH (g:GeometricGraph {graph_id: $graph_id})
            CREATE (p:Point {
                point_id: $point_id,
                original_id: $original_id,
                x: $x,
                y: $y,
                label: $label,
                confidence: $confidence,
                angles_at_vertex: $angles
            })
            CREATE (g)-[:CONTAINS]->(p)
        """, {
            'graph_id': graph_id,
            'point_id': namespaced_id,
            'original_id': point.point_id,
            'x': point.x,
            'y': point.y,
            'label': original_label,
            'confidence': point.confidence,
            'angles': json.dumps(getattr(point, 'angles_at_vertex', []))
        })

    def _create_edge_relationship(self, session, graph_id: str, edge: GeometricLine):
        """Create edge as relationship between points with properties including parametric support

        Edge and point IDs are namespaced with graph_id to prevent conflicts.
        """
        if edge.start_point_id and edge.end_point_id:
            try:
                # Namespace all IDs
                namespaced_edge_id = self._namespace_id(graph_id, edge.line_id)
                namespaced_start_id = self._namespace_id(graph_id, edge.start_point_id)
                namespaced_end_id = self._namespace_id(graph_id, edge.end_point_id)

                # Get edge relationships with safe defaults
                relationships = getattr(edge, 'relationships', {}) or {}
                edge_type = getattr(edge, 'edge_type', 'segment') or 'segment'
                is_parametric = getattr(edge, 'is_parametric', False) or False
                length_expression = getattr(edge, 'length_expression', None)
                original_label = edge.label or edge.line_id

                # Safely extract relationship lists, handling None values
                parallel_to_raw = relationships.get('parallel_to') or []
                perpendicular_to_raw = relationships.get('perpendicular_to') or []
                equal_to_raw = relationships.get('equal_to') or []
                bisects_raw = relationships.get('bisects')

                # Ensure they are lists and namespace them
                parallel_to = [self._namespace_id(graph_id, ref) for ref in parallel_to_raw] if isinstance(parallel_to_raw, list) else []
                perpendicular_to = [self._namespace_id(graph_id, ref) for ref in perpendicular_to_raw] if isinstance(perpendicular_to_raw, list) else []
                equal_to = [self._namespace_id(graph_id, ref) for ref in equal_to_raw] if isinstance(equal_to_raw, list) else []
                bisects = self._namespace_id(graph_id, bisects_raw) if bisects_raw else None

                session.run("""
                    MATCH (p1:Point {point_id: $start_id})
                    MATCH (p2:Point {point_id: $end_id})
                    CREATE (p1)-[:EDGE {
                        edge_id: $edge_id,
                        original_id: $original_id,
                        label: $label,
                        length: $length,
                        is_parametric: $is_parametric,
                        length_expression: $length_expression,
                        edge_type: $edge_type,
                        parallel_to: $parallel_to,
                        perpendicular_to: $perpendicular_to,
                        equal_to: $equal_to,
                        bisects: $bisects,
                        confidence: $confidence
                    }]->(p2)
                """, {
                    'start_id': namespaced_start_id,
                    'end_id': namespaced_end_id,
                    'edge_id': namespaced_edge_id,
                    'original_id': edge.line_id,
                    'label': original_label,
                    'length': edge.length,
                    'is_parametric': is_parametric,
                    'length_expression': length_expression,
                    'edge_type': edge_type,
                    'parallel_to': json.dumps(parallel_to),
                    'perpendicular_to': json.dumps(perpendicular_to),
                    'equal_to': json.dumps(equal_to),
                    'bisects': bisects,
                    'confidence': edge.confidence
                })

            except Exception as e:
                logger.error(f"Error creating edge relationship {edge.line_id}: {e}")
                logger.error(f"Edge data: start={edge.start_point_id}, end={edge.end_point_id}, relationships={getattr(edge, 'relationships', None)}")
                # Continue processing other edges rather than failing completely
                pass

    def _create_shape_node(self, session, graph_id: str, shape: DetectedShape):
        """Create a shape node and connect to its components

        Shape IDs are namespaced with graph_id to prevent conflicts.
        """
        # Namespace the shape_id
        namespaced_shape_id = self._namespace_id(graph_id, shape.shape_id)
        original_label = shape.label or shape.shape_id

        session.run("""
            MATCH (g:GeometricGraph {graph_id: $graph_id})
            CREATE (s:Shape {
                shape_id: $shape_id,
                original_id: $original_id,
                shape_type: $shape_type,
                properties: $properties,
                label: $label,
                confidence: $confidence
            })
            CREATE (g)-[:CONTAINS]->(s)
        """, {
            'graph_id': graph_id,
            'shape_id': namespaced_shape_id,
            'original_id': shape.shape_id,
            'shape_type': shape.shape_type,
            'properties': json.dumps(shape.properties),
            'label': original_label,
            'confidence': shape.confidence
        })

        # Connect shape to vertices (with namespaced IDs)
        for vertex_id in shape.vertex_ids:
            namespaced_vertex_id = self._namespace_id(graph_id, vertex_id)
            session.run("""
                MATCH (s:Shape {shape_id: $shape_id})
                MATCH (p:Point {point_id: $vertex_id})
                MERGE (s)-[:HAS_VERTEX]->(p)
            """, {'shape_id': namespaced_shape_id, 'vertex_id': namespaced_vertex_id})

    def _create_shape_containment(self, session, graph_id: str, shapes: List[DetectedShape]):
        """Create containment relationships between shapes

        All shape IDs are namespaced to prevent conflicts.
        """
        for shape in shapes:
            namespaced_parent_id = self._namespace_id(graph_id, shape.shape_id)

            # Create CONTAINS relationships
            contains_shapes = getattr(shape, 'contains_shapes', [])
            for contained_shape_id in contains_shapes:
                namespaced_child_id = self._namespace_id(graph_id, contained_shape_id)
                session.run("""
                    MATCH (parent:Shape {shape_id: $parent_id})
                    MATCH (child:Shape {shape_id: $child_id})
                    MERGE (parent)-[:CONTAINS_SHAPE]->(child)
                """, {
                    'parent_id': namespaced_parent_id,
                    'child_id': namespaced_child_id
                })

            # Create CONTAINED_IN relationships
            contained_in = getattr(shape, 'contained_in', None)
            if contained_in:
                namespaced_container_id = self._namespace_id(graph_id, contained_in)
                session.run("""
                    MATCH (child:Shape {shape_id: $child_id})
                    MATCH (parent:Shape {shape_id: $parent_id})
                    MERGE (child)-[:CONTAINED_IN]->(parent)
                """, {
                    'child_id': namespaced_parent_id,
                    'parent_id': namespaced_container_id
                })

    def _attach_theorems_to_graph(self, graph_id: str, graph_structure: Dict):
        """Attach relevant theorems from knowledge base to graph elements

        Uses namespaced shape IDs to attach theorems.
        """
        logger.info(f"Attaching theorems from knowledge base to graph: {graph_id}")

        with self.kb.driver.session() as session:
            # Get all shapes in the graph
            shapes = graph_structure.get('shapes', [])

            for shape in shapes:
                shape_type = shape.shape_type
                shape_id = shape.shape_id
                namespaced_shape_id = self._namespace_id(graph_id, shape_id)

                # Query knowledge base for applicable theorems
                applicable_theorems = self._get_applicable_theorems_from_kb(shape_type)

                # Attach theorems to shape
                for theorem_info in applicable_theorems:
                    # Evaluate applicability based on shape properties
                    applicability_score, reasoning = self._evaluate_theorem_applicability(
                        theorem_info, shape.properties
                    )

                    if applicability_score > 0.3:  # Threshold for relevance
                        session.run("""
                            MATCH (s:Shape {shape_id: $shape_id})
                            MATCH (t:Theorem {theorem_id: $theorem_id})
                            CREATE (s)-[:CAN_APPLY {
                                applicability_score: $score,
                                reasoning: $reasoning,
                                attached_at: datetime()
                            }]->(t)
                        """, {
                            'shape_id': namespaced_shape_id,
                            'theorem_id': theorem_info['theorem_id'],
                            'score': applicability_score,
                            'reasoning': reasoning
                        })

                        logger.info(f"Attached theorem '{theorem_info['name']}' to {shape_type} {namespaced_shape_id} (score: {applicability_score:.2f})")

    def _get_applicable_theorems_from_kb(self, shape_type: str) -> List[Dict]:
        """Get applicable theorems for a shape type from knowledge base"""

        with self.kb.driver.session() as session:
            results = session.run("""
                MATCH (s:Shape {name: $shape_type})-[:APPLICABLE_TO]->(t:Theorem)
                RETURN t.theorem_id as theorem_id, t.name as name,
                       t.description as description, t.conditions as conditions,
                       t.mathematical_form as mathematical_form,
                       t.usage_count as usage_count
                ORDER BY t.usage_count DESC
            """, {'shape_type': shape_type.lower()})

            return [dict(record) for record in results]

    def _evaluate_theorem_applicability(self, theorem_info: Dict, shape_properties: Dict) -> Tuple[float, str]:
        """Evaluate how applicable a theorem is to a specific shape instance"""

        score = 0.5  # Base score
        reasoning_parts = []

        conditions = theorem_info.get('conditions', [])
        theorem_name = theorem_info.get('name', '')

        # Check specific conditions
        if 'right' in theorem_name.lower() or any('right' in str(c).lower() for c in conditions):
            if shape_properties.get('triangle_type') == 'right':
                score += 0.4
                reasoning_parts.append("Triangle is right-angled")
            else:
                score -= 0.2

        # Check for available measurements
        if shape_properties.get('side_lengths') or shape_properties.get('angles'):
            score += 0.2
            reasoning_parts.append("Measurements available")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "General applicability"

        return min(score, 1.0), reasoning

class GeometricGraphQuery:
    """Query interface for geometric graphs"""

    def __init__(self, knowledge_base: GeometricKnowledgeBase):
        self.kb = knowledge_base

    def get_graph_summary(self, graph_id: str) -> Dict:
        """Get a summary of the geometric graph"""
        with self.kb.driver.session() as session:
            result = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})
                OPTIONAL MATCH (g)-[:CONTAINS]->(p:Point)
                OPTIONAL MATCH (g)-[:CONTAINS]->(s:Shape)
                OPTIONAL MATCH (p1:Point)-[e:EDGE]->(p2:Point)
                WHERE (g)-[:CONTAINS]->(p1)
                RETURN g.image_path as image_path,
                       g.created_at as created_at,
                       g.extraction_method as extraction_method,
                       count(DISTINCT p) as point_count,
                       count(DISTINCT e) as edge_count,
                       count(DISTINCT s) as shape_count
            """, {'graph_id': graph_id}).single()

            return dict(result) if result else {}

    def get_applicable_theorems(self, graph_id: str) -> List[Dict]:
        """Get all theorems applicable to shapes in the graph"""
        with self.kb.driver.session() as session:
            results = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s:Shape)
                MATCH (s)-[r:CAN_APPLY]->(t:Theorem)
                RETURN s.shape_id as shape_id,
                       s.shape_type as shape_type,
                       s.label as shape_label,
                       t.theorem_id as theorem_id,
                       t.name as theorem_name,
                       t.description as theorem_description,
                       t.mathematical_form as mathematical_form,
                       r.applicability_score as score,
                       r.reasoning as reasoning
                ORDER BY r.applicability_score DESC
            """, {'graph_id': graph_id})

            return [dict(record) for record in results]

    def find_shapes_by_type(self, graph_id: str, shape_type: str) -> List[Dict]:
        """Find all shapes of a specific type in the graph"""
        with self.kb.driver.session() as session:
            results = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s:Shape {shape_type: $shape_type})
                OPTIONAL MATCH (s)-[:HAS_VERTEX]->(p:Point)
                RETURN s.shape_id as shape_id,
                       s.label as label,
                       s.properties as properties,
                       s.confidence as confidence,
                       collect(DISTINCT p.point_id) as vertex_ids
            """, {'graph_id': graph_id, 'shape_type': shape_type})

            return [dict(record) for record in results]

    def get_measurements(self, graph_id: str) -> Dict[str, Any]:
        """Get all measurements from the graph"""
        with self.kb.driver.session() as session:
            # Get edge lengths from EDGE relationships
            edges = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(p1:Point)
                MATCH (p1)-[e:EDGE]->(p2:Point)
                WHERE e.length IS NOT NULL
                RETURN e.edge_id as edge_id, e.label as label, e.length as length
            """, {'graph_id': graph_id}).values()

            # Get angle measures from Point properties
            points_with_angles = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(p:Point)
                WHERE p.angles_at_vertex IS NOT NULL
                RETURN p.point_id as point_id, p.label as label,
                       p.angles_at_vertex as angles_json
            """, {'graph_id': graph_id}).values()

            angles = []
            for point_id, label, angles_json in points_with_angles:
                if angles_json:
                    try:
                        angles_data = json.loads(angles_json) if isinstance(angles_json, str) else angles_json
                        for angle in angles_data:
                            angles.append({
                                'id': angle.get('angle_id'),
                                'label': angle.get('label'),
                                'measure': angle.get('measure'),
                                'type': angle.get('angle_type'),
                                'vertex': point_id
                            })
                    except:
                        pass

            return {
                'edge_lengths': [{'id': e[0], 'label': e[1], 'length': e[2]} for e in edges],
                'angle_measures': angles
            }

    def get_shape_relationships(self, graph_id: str) -> List[Dict]:
        """Get relationships between shapes in the graph"""
        with self.kb.driver.session() as session:
            results = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(s1:Shape)
                MATCH (s1)-[r]->(s2:Shape)
                WHERE type(r) <> 'CONTAINS'
                RETURN s1.shape_id as shape1_id,
                       s1.shape_type as shape1_type,
                       s2.shape_id as shape2_id,
                       s2.shape_type as shape2_type,
                       type(r) as relationship_type
            """, {'graph_id': graph_id})

            return [dict(record) for record in results]

# Main processing class that combines all components
class GeometricImageProcessor:
    """Main class for processing geometric images into Neo4j graphs"""

    def __init__(self, config: Dict[str, str]):
        # Initialize knowledge base
        self.kb_manager = GeometricKnowledgeBaseManager(config)

        # Initialize graph builder
        self.graph_builder = GeometricGraphBuilder(self.kb_manager.kb)

        # Initialize query interface
        self.query_interface = GeometricGraphQuery(self.kb_manager.kb)

    def process_image(self, image_path: str, problem_id: Optional[str] = None) -> Dict:
        """Process a geometric image and return graph information"""
        try:
            # Create graph representation
            graph_id = self.graph_builder.process_image_to_graph(image_path, problem_id)

            # Get graph summary
            summary = self.query_interface.get_graph_summary(graph_id)

            # Get applicable theorems
            theorems = self.query_interface.get_applicable_theorems(graph_id)

            # Get shape relationships
            relationships = self.query_interface.get_shape_relationships(graph_id)

            return {
                'graph_id': graph_id,
                'summary': summary,
                'applicable_theorems': theorems,
                'shape_relationships': relationships,
                'success': True
            }

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def batch_process_images(self, image_paths: List[str], problem_ids: Optional[List[str]] = None) -> List[Dict]:
        """Process multiple images in batch"""
        results = []

        for i, image_path in enumerate(image_paths):
            problem_id = problem_ids[i] if problem_ids and i < len(problem_ids) else None
            result = self.process_image(image_path, problem_id)
            result['image_path'] = image_path
            results.append(result)

            logger.info(f"Processed image {i+1}/{len(image_paths)}: {image_path}")

        return results

    def close(self):
        """Clean up resources"""
        self.kb_manager.close()

# Example usage and testing
def main():
    """Example usage of the geometric graph constructor"""

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

    # Initialize processor
    processor = GeometricImageProcessor(config)

    try:
        # Process single image
        image_path = "images/geo3k/train/1/img_diagram.png"
        result = processor.process_image(image_path, "problem_001")

        print("Processing Result:")
        print(f"Graph ID: {result['graph_id']}")
        print(f"Summary: {result['summary']}")
        print(f"Applicable Theorems: {len(result['applicable_theorems'])}")
        print(f"Shape Relationships: {len(result['shape_relationships'])}")


    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        processor.close()

if __name__ == "__main__":
    main()
