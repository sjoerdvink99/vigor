# Visualization Recommendation for

## Authors

Sjoerd Vink

## Description

This repository is part of the thesis project of the Applied Data Science Masters at Utrecht University. The system is designed to recommend visualizations based on input statistics derived from a knowledge graph (KG).

## Usage

```
from recommendation_agent import RecommendationAgent

# Initialize the agent
agent = RecommendationAgent(
    alpha=0.1,
    epsilon=0.1,
    n_statistics=4,
    domains=['domain1', 'domain2'],
    visualizations=['viz1', 'viz2', 'viz3'],
    q_table_path='path/to/q_table.json',
    batch_size=10,
    decay_rate=0.99,
    min_alpha=0.05
)

# Load existing scores
agent.load_scores()

# Generate state ID from statistics
statistics = {'stat1': 1.0, 'stat2': 2.0, 'stat3': 3.0, 'stat4': 4.0}
state_id = agent.state_id(statistics)

# Get a visualization recommendation
domain = 'domain1'
recommended_action, is_exploratory = agent.recommend_visualization(domain, state_id)
print(f"Recommended action: {recommended_action}, Exploratory: {is_exploratory}")

# Update Q-value based on user feedback
agent.update_q_value(domain, state_id, action=recommended_action, reward=4.5, require_feedback=True)

# Add a new domain
agent.add_domain('new_domain')

# Add a new visualization
agent.add_visualization('new_viz')

# Export indices
agent.export_index(folder='path/to/folder/')
```
