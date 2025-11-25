#veriflow/structural/schema.py
N8N_MINIMAL_SCHEMA = {
    "type": "object",
    "required": ["nodes", "connections"],
    "properties": {
        "nodes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "name", "type"],
                "properties": {
                    "id": {"type": ["string", "number"]},

                    "name": {"type": "string"},

                    "type": {
                        "type": "string",
                        "pattern": ".+\\..+"    # n8n-like type format
                    },

                    "parameters": {
                        "type": "object",
                        "additionalProperties": True,
                        "default": {}
                    }
                }
            }
        },

        "connections": {
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "patternProperties": {
                        "^main$": {
                            "type": "array"
                        }
                    },
                    "additionalProperties": True
                }
            }
        }
    }
}