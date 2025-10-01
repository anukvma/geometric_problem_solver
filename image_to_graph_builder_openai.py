"""
Geometric Graph Constructor - Part 2 Implementation
Uses GPT-4 Vision API to process geometric images and creates Neo4j graph representations
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import json
import base64
from io import BytesIO
import os
import re

# Core dependencies
import neo4j
from neo4j import GraphDatabase
import openai
from PIL import Image
import numpy as np

# Import from Part 1
from knowledge_base_builder_openai1 import GeometricKnowledgeBase, GeometricKnowledgeBaseManager

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
    """Uses GPT-4 Vision to extract graph structure from geometric images"""
    
    def __init__(self, openai_client, model_name: str = "gpt-4o"):
        self.openai_client = openai_client
        self.model_name = model_name
        logger.info(f"Initialized GPT Vision Graph Extractor with model: {model_name}")
    
    def extract_graph_structure(self, image_path: str) -> Dict[str, Any]:
        """Extract complete graph structure from geometric image using GPT Vision"""
        
        logger.info(f"Extracting graph structure from: {os.path.basename(image_path)}")
        
        # Convert image to base64
        image_base64 = self._image_to_base64(image_path)
        
        # Extract geometric elements using GPT Vision
        prompt = self._create_graph_extraction_prompt()
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            graph_data = self._extract_json_from_response(result_text)
            
            # Validate and structure the graph data
            structured_graph = self._structure_graph_data(graph_data)
            
            logger.info(f"Successfully extracted graph structure:")
            logger.info(f"  - Points: {len(structured_graph.get('points', []))}")
            logger.info(f"  - Lines: {len(structured_graph.get('lines', []))}")
            logger.info(f"  - Shapes: {len(structured_graph.get('shapes', []))}")
            logger.info(f"  - Angles: {len(structured_graph.get('angles', []))}")
            
            return structured_graph
            
        except Exception as e:
            logger.error(f"Error extracting graph structure: {e}")
            return {
                'points': [],
                'lines': [],
                'shapes': [],
                'angles': [],
                'error': str(e)
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
                    "approximate_position": {"x": 0.5, "y": 0.2}
                }
            ],
            "lines": [
                {
                    "line_id": "AB",
                    "start_point": "A",
                    "end_point": "B",
                    "label": "AB",
                    "length": 5.0,
                    "description": "side of triangle"
                }
            ],
            "angles": [
                {
                    "angle_id": "angle_ABC",
                    "vertex": "B",
                    "arm1": "BA",
                    "arm2": "BC",
                    "measure": 60.0,
                    "angle_type": "acute",
                    "label": "∠ABC"
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
                        "side_lengths": [3, 4, 5],
                        "angles": [90, 60, 30]
                    },
                    "label": "△ABC"
                }
            ],
            "measurements": [
                {
                    "type": "length",
                    "element": "AB",
                    "value": 5.0,
                    "unit": "units"
                }
            ],
            "relationships": [
                {
                    "type": "parallel",
                    "element1": "AB",
                    "element2": "CD"
                },
                {
                    "type": "perpendicular",
                    "element1": "AB",
                    "element2": "BC"
                }
            ]
        }

        IMPORTANT GUIDELINES:
        1. Use clear, consistent labels (A, B, C for points; AB, BC for lines)
        2. For position coordinates, use normalized values (0.0 to 1.0) representing relative positions
        3. Include ALL visible measurements (lengths, angles)
        4. Identify shape types: triangle, circle, rectangle, square, parallelogram, etc.
        5. Note special properties: right angles, equal sides, parallel lines, middle points, median etc.
        6. Be precise with numerical values shown in the image
        7. If a value is not visible, omit it or set to null
        8. Include relationships between elements (parallel, perpendicular, congruent, etc.)
        9. If there are multiple shapes in the image or shape contains other shape within it, then provide all the associated shapes

        Return ONLY the JSON object, no additional text.
        """
    
    def _structure_graph_data(self, raw_data: Dict) -> Dict[str, Any]:
        """Structure and validate the extracted graph data"""
        
        structured = {
            'points': [],
            'lines': [],
            'angles': [],
            'shapes': [],
            'measurements': raw_data.get('measurements', []),
            'relationships': raw_data.get('relationships', [])
        }
        
        # Process points
        for point_data in raw_data.get('points', []):
            point = GeometricPoint(
                x=point_data.get('approximate_position', {}).get('x', 0.0),
                y=point_data.get('approximate_position', {}).get('y', 0.0),
                point_id=point_data.get('point_id', point_data.get('label', f"P_{len(structured['points'])}")),
                label=point_data.get('label'),
                confidence=0.9
            )
            structured['points'].append(point)
        
        # Process lines
        for line_data in raw_data.get('lines', []):
            line = GeometricLine(
                start_point_id=line_data.get('start_point', ''),
                end_point_id=line_data.get('end_point', ''),
                line_id=line_data.get('line_id', line_data.get('label', f"L_{len(structured['lines'])}")),
                length=line_data.get('length'),
                label=line_data.get('label'),
                confidence=0.9
            )
            structured['lines'].append(line)
        
        # Process angles
        for angle_data in raw_data.get('angles', []):
            angle = GeometricAngle(
                vertex_id=angle_data.get('vertex', ''),
                arm1_id=angle_data.get('arm1', ''),
                arm2_id=angle_data.get('arm2', ''),
                angle_id=angle_data.get('angle_id', f"A_{len(structured['angles'])}"),
                measure=angle_data.get('measure'),
                angle_type=angle_data.get('angle_type'),
                label=angle_data.get('label'),
                confidence=0.9
            )
            structured['angles'].append(angle)
        
        # Process shapes
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
            structured['shapes'].append(shape)
        
        return structured
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize if too large
                max_size = 2048
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=85, optimize=True)
                
                return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from GPT response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            return {}
    
class GeometricGraphBuilder:
    """Builds Neo4j graph representation using GPT Vision analysis"""
    
    def __init__(self, knowledge_base: GeometricKnowledgeBase):
        self.kb = knowledge_base
        self.vision_extractor = GPTVisionGraphExtractor(knowledge_base.openai_client)
    
    def process_image_to_graph(self, image_path: str, problem_id: Optional[str] = None) -> str:
        """Main method to process image and create graph representation using GPT Vision"""
        logger.info(f"Processing image: {image_path}")
        
        # Generate unique graph ID
        graph_id = problem_id or f"graph_{hash(image_path)}_{int(datetime.now().timestamp())}"
        
        # Extract graph structure using GPT Vision
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
    
    def _create_graph_in_neo4j(self, graph_id: str, graph_structure: Dict, image_path: str):
        """Create the graph structure in Neo4j from GPT Vision extraction"""
        
        with self.kb.driver.session() as session:
            # Create main graph node
            session.run("""
                CREATE (g:GeometricGraph {
                    graph_id: $graph_id,
                    image_path: $image_path,
                    created_at: datetime(),
                    extraction_method: 'GPT_Vision'
                })
            """, {
                'graph_id': graph_id,
                'image_path': image_path
            })
            
            # Create points
            for point in graph_structure.get('points', []):
                self._create_point_node(session, graph_id, point)
            
            # Create lines
            for line in graph_structure.get('lines', []):
                self._create_line_node(session, graph_id, line)
            
            # Create angles
            for angle in graph_structure.get('angles', []):
                self._create_angle_node(session, graph_id, angle)
            
            # Create shapes
            for shape in graph_structure.get('shapes', []):
                self._create_shape_node(session, graph_id, shape)
            
            # Create relationships from extracted data
            self._create_relationships_from_structure(session, graph_id, graph_structure)
    
    def _create_point_node(self, session, graph_id: str, point: GeometricPoint):
        """Create a point node in the graph"""
        session.run("""
            MATCH (g:GeometricGraph {graph_id: $graph_id})
            CREATE (p:Point {
                point_id: $point_id,
                x: $x,
                y: $y,
                label: $label,
                confidence: $confidence
            })
            CREATE (g)-[:CONTAINS]->(p)
        """, {
            'graph_id': graph_id,
            'point_id': point.point_id,
            'x': point.x,
            'y': point.y,
            'label': point.label,
            'confidence': point.confidence
        })
    
    def _create_line_node(self, session, graph_id: str, line: GeometricLine):
        """Create a line node and connect to its endpoints"""
        session.run("""
            MATCH (g:GeometricGraph {graph_id: $graph_id})
            CREATE (l:Line {
                line_id: $line_id,
                length: $length,
                label: $label,
                confidence: $confidence
            })
            CREATE (g)-[:CONTAINS]->(l)
        """, {
            'graph_id': graph_id,
            'line_id': line.line_id,
            'length': line.length,
            'label': line.label,
            'confidence': line.confidence
        })
        
        # Connect to endpoints
        if line.start_point_id and line.end_point_id:
            session.run("""
                MATCH (l:Line {line_id: $line_id})
                MATCH (p1:Point {point_id: $start_id})
                MATCH (p2:Point {point_id: $end_id})
                CREATE (l)-[:STARTS_AT]->(p1)
                CREATE (l)-[:ENDS_AT]->(p2)
                CREATE (p1)-[:CONNECTED_TO]->(p2)
            """, {
                'line_id': line.line_id,
                'start_id': line.start_point_id,
                'end_id': line.end_point_id
            })
    
    def _create_angle_node(self, session, graph_id: str, angle: GeometricAngle):
        """Create an angle node"""
        session.run("""
            MATCH (g:GeometricGraph {graph_id: $graph_id})
            CREATE (a:Angle {
                angle_id: $angle_id,
                measure: $measure,
                angle_type: $angle_type,
                label: $label,
                confidence: $confidence
            })
            CREATE (g)-[:CONTAINS]->(a)
        """, {
            'graph_id': graph_id,
            'angle_id': angle.angle_id,
            'measure': angle.measure,
            'angle_type': angle.angle_type,
            'label': angle.label,
            'confidence': angle.confidence
        })
        
        # Connect angle to vertex and arms
        if angle.vertex_id:
            session.run("""
                MATCH (a:Angle {angle_id: $angle_id})
                MATCH (p:Point {point_id: $vertex_id})
                CREATE (a)-[:HAS_VERTEX]->(p)
            """, {'angle_id': angle.angle_id, 'vertex_id': angle.vertex_id})
    
    def _create_shape_node(self, session, graph_id: str, shape: DetectedShape):
        """Create a shape node and connect to its components"""
        session.run("""
            MATCH (g:GeometricGraph {graph_id: $graph_id})
            CREATE (s:Shape {
                shape_id: $shape_id,
                shape_type: $shape_type,
                properties: $properties,
                label: $label,
                confidence: $confidence
            })
            CREATE (g)-[:CONTAINS]->(s)
        """, {
            'graph_id': graph_id,
            'shape_id': shape.shape_id,
            'shape_type': shape.shape_type,
            'properties': json.dumps(shape.properties),
            'label': shape.label,
            'confidence': shape.confidence
        })
        
        # Connect shape to vertices
        for vertex_id in shape.vertex_ids:
            session.run("""
                MATCH (s:Shape {shape_id: $shape_id})
                MATCH (p:Point {point_id: $vertex_id})
                MERGE (s)-[:HAS_VERTEX]->(p)
            """, {'shape_id': shape.shape_id, 'vertex_id': vertex_id})
        
        # Connect shape to edges
        for edge_id in shape.edge_ids:
            session.run("""
                MATCH (s:Shape {shape_id: $shape_id})
                MATCH (l:Line {line_id: $edge_id})
                MERGE (s)-[:HAS_EDGE]->(l)
            """, {'shape_id': shape.shape_id, 'edge_id': edge_id})
    
    def _create_relationships_from_structure(self, session, graph_id: str, graph_structure: Dict):
        """Create additional relationships from extracted structure"""
        
        # Process explicit relationships from GPT Vision
        for relationship in graph_structure.get('relationships', []):
            rel_type = relationship.get('type', 'RELATES_TO').upper()
            element1 = relationship.get('element1')
            element2 = relationship.get('element2')
            
            if element1 and element2:
                # Try to find elements (could be lines, shapes, etc.)
                session.run(f"""
                    MATCH (e1) WHERE e1.line_id = $elem1 OR e1.shape_id = $elem1
                    MATCH (e2) WHERE e2.line_id = $elem2 OR e2.shape_id = $elem2
                    MERGE (e1)-[r:{rel_type}]->(e2)
                """, {'elem1': element1, 'elem2': element2})
    
    def _attach_theorems_to_graph(self, graph_id: str, graph_structure: Dict):
        """Attach relevant theorems from knowledge base to graph elements"""
        logger.info(f"Attaching theorems from knowledge base to graph: {graph_id}")
        
        with self.kb.driver.session() as session:
            # Get all shapes in the graph
            shapes = graph_structure.get('shapes', [])
            
            for shape in shapes:
                shape_type = shape.shape_type
                shape_id = shape.shape_id
                
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
                            'shape_id': shape_id,
                            'theorem_id': theorem_info['theorem_id'],
                            'score': applicability_score,
                            'reasoning': reasoning
                        })
                        
                        logger.info(f"Attached theorem '{theorem_info['name']}' to {shape_type} {shape_id} (score: {applicability_score:.2f})")
    
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
                OPTIONAL MATCH (g)-[:CONTAINS]->(l:Line)
                OPTIONAL MATCH (g)-[:CONTAINS]->(s:Shape)
                OPTIONAL MATCH (g)-[:CONTAINS]->(a:Angle)
                RETURN g.image_path as image_path,
                       g.created_at as created_at,
                       g.extraction_method as extraction_method,
                       count(DISTINCT p) as point_count,
                       count(DISTINCT l) as line_count,
                       count(DISTINCT s) as shape_count,
                       count(DISTINCT a) as angle_count
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
                OPTIONAL MATCH (s)-[:HAS_EDGE]->(l:Line)
                RETURN s.shape_id as shape_id,
                       s.label as label,
                       s.properties as properties,
                       s.confidence as confidence,
                       collect(DISTINCT p.point_id) as vertex_ids,
                       collect(DISTINCT l.line_id) as edge_ids
            """, {'graph_id': graph_id, 'shape_type': shape_type})
            
            return [dict(record) for record in results]
    
    def get_measurements(self, graph_id: str) -> Dict[str, Any]:
        """Get all measurements from the graph"""
        with self.kb.driver.session() as session:
            # Get line lengths
            lines = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(l:Line)
                WHERE l.length IS NOT NULL
                RETURN l.line_id as line_id, l.label as label, l.length as length
            """, {'graph_id': graph_id}).values()
            
            # Get angle measures
            angles = session.run("""
                MATCH (g:GeometricGraph {graph_id: $graph_id})-[:CONTAINS]->(a:Angle)
                WHERE a.measure IS NOT NULL
                RETURN a.angle_id as angle_id, a.label as label, 
                       a.measure as measure, a.angle_type as angle_type
            """, {'graph_id': graph_id}).values()
            
            return {
                'line_lengths': [{'id': l[0], 'label': l[1], 'length': l[2]} for l in lines],
                'angle_measures': [{'id': a[0], 'label': a[1], 'measure': a[2], 'type': a[3]} for a in angles]
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
    
    # Configuration (same as Part 1)
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
            