from langgraph.graph import StateGraph, END
from agents.concept_expander import ConceptExpanderAgent
from agents.visual_designer import VisualDesignerAgent
from agents.layout_planner import LayoutPlannerAgent
from agents.quality_refiner import QualityRefinerAgent
from api.models.schemas import PosterState

concept_agent = ConceptExpanderAgent()
visual_agent = VisualDesignerAgent()
layout_agent = LayoutPlannerAgent()
quality_agent = QualityRefinerAgent()

def build_workflow() -> StateGraph:
    workflow = StateGraph(PosterState)

    # Define nodes in graph and call run() for each class
    workflow.add_node("ConceptExpander", concept_agent.run)
    workflow.add_node("VisualDesigner", visual_agent.run)
    workflow.add_node("LayoutPlanner", layout_agent.run)
    workflow.add_node("QualityRefiner", quality_agent.run)

    # Define edges in graph
    workflow.add_edge("ConceptExpander", "VisualDesigner")
    workflow.add_edge("VisualDesigner", "LayoutPlanner")
    workflow.add_edge("LayoutPlanner", "QualityRefiner")

    # Define cycle, QualityRefiner -> Visual Designer
    workflow.add_conditional_edges(
        "QualityRefiner",
        lambda state: state["next_node"],
        {
            "VisualDesigner": "VisualDesigner",
            END: END
        }
    )

    workflow.set_entry_point("ConceptExpander")
    return workflow

if __name__ == "__main__":
    poster_graph = build_workflow().compile()

    init_state = {
        "user_input": "Seafood noodle poster.",
        "iteration_count": 0,
        "max_iterations": 3,
    }

    result = poster_graph.invoke(init_state)
    print("Generation Successful!")
