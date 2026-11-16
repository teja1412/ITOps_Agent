from graph import compiled_graph,config

saved_state = compiled_graph.get_state(config)

print("Full saved values:", saved_state.values)
print("Saved user_id:", saved_state.values.get("user_id"))
print("Saved messages:", saved_state.values.get("messages"))