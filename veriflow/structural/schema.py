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
                    "id": {
                        "type": ["string", "number"]
                    },
                    "name": {
                        "type": "string",
                        "minLength": 1
                    },
                    "type": {
                        "type": "string",
                        # n8n-like pattern: prevents invalid non-string / malformed node types
                        "pattern": "^[A-Za-z0-9_-]+\\.[A-Za-z0-9_.-]+$"

                    },

                    # Optional: parameters must be an object when present
                    "parameters": {
                        "type": "object"
                    },

                    # Optional field commonly found in real n8n workflows
                    "typeVersion": {
                        "type": ["integer", "number"]
                    },

                    # Optional: typical layout information in n8n export
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["x", "y"],
                        "additionalProperties": True
                    }
                },
                "additionalProperties": True
            },
            "minItems": 1
        },

        "connections": {
            "type": "object",

            # Top-level keys: source node names (non-empty strings)
            "patternProperties": {
                "^.+$": {
                    "type": "object",

                    # Inner keys: output streams (e.g., "main"), also non-empty strings
                    "patternProperties": {
                        "^.+$": {
                            "type": "array",
                            "items": {

                                # Either:
                                #   (1) a path array: [ {hop}, {hop}, ... ]
                                #   (2) a single hop object: {hop}
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["node"],
                                            "properties": {
                                                "node": {"type": "string"},
                                                # Optional hop type (usually "main")
                                                "type": {"type": "string"},
                                                # Optional hop index: non-negative integer
                                                "index": {
                                                    "type": "integer",
                                                    "minimum": 0
                                                }
                                            },
                                            "additionalProperties": True
                                        },
                                        "minItems": 1
                                    },
                                    {
                                        "type": "object",
                                        "required": ["node"],
                                        "properties": {
                                            "node": {"type": "string"},
                                            # Optional hop type
                                            "type": {"type": "string"},
                                            # Optional hop index
                                            "index": {
                                                "type": "integer",
                                                "minimum": 0
                                            }
                                        },
                                        "additionalProperties": True
                                    }
                                ]
                            }
                        }
                    },

                    "additionalProperties": False
                }
            },

            "additionalProperties": False
        }
    }
}