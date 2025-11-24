#veriflow/structural/schema.py
N8N_MINIMAL_SCHEMA = {
    "type": "object",
    "required": ["nodes", "connections"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "type"],
                "properties": {
                    "id": {"type": ["string", "number"]},
                    "name": {"type": "string"},
                    "type": {"type": "string"},  # e.g., n8n-nodes-base.emailSend
                    "parameters": {"type": "object"}
                }
            },
            "minItems": 1
        },
        "connections": {"type": "object"}
    }
}