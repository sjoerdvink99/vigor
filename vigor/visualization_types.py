from enum import Enum

class VisualizationType(Enum):
  NODELINK = "Node-Link"
  MATRIX = "Matrix"
  NODETRIX = "NodeTrix"
  NODELINK_MAP = "Node-Link-Map"
  PAOHVIS = "PaohVis"
  CHORD_DIAGRAM = "Chord Diagram"
  TREEMAP = "Treemap"

class NobreVisualizations(Enum):
  NODELINK = "Node-Link"
  NODELINK_FACETING = "Node-Link Attribute-driven Faceting"
  NODELINK_POSITIONING = "Node-Link Attribute-driven Positioning"
  MATRIX = "Matrix"
  QUILTS = "Quilts"
  BIOFABRIC = "BioFabric"
  TREEMAP = "Treemap"
  SUNBURST = "Sunburst"