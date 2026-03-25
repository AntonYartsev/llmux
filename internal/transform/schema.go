package transform

import "strings"

// converts lowercase JSON Schema types to the uppercase strings that the Gemini API requires
var schemaTypes = map[string]string{
	"string":  "STRING",
	"number":  "NUMBER",
	"integer": "INTEGER",
	"boolean": "BOOLEAN",
	"array":   "ARRAY",
	"object":  "OBJECT",
}

// lists JSON Schema keys that Gemini does not understand and must be stripped before forwarding
var unsupportedKeys = map[string]bool{
	"additionalProperties": true,
	"$schema":              true,
	"$defs":                true,
	"definitions":          true,
	"$ref":                 true,
}

// recursively converts an OpenAI-style JSON Schema map to a Gemini-compatible one: uppercase types, stripped unsupported keys
func NormalizeSchema(schema map[string]any) map[string]any {
	result := make(map[string]any, len(schema))
	for k, v := range schema {
		if unsupportedKeys[k] {
			continue
		}
		switch k {
		case "type":
			if s, ok := v.(string); ok {
				if upper, ok := schemaTypes[strings.ToLower(s)]; ok {
					result[k] = upper
				} else {
					result[k] = strings.ToUpper(s)
				}
			} else {
				result[k] = v
			}
		case "properties":
			if props, ok := v.(map[string]any); ok {
				normalised := make(map[string]any, len(props))
				for pk, pv := range props {
					if pm, ok := pv.(map[string]any); ok {
						normalised[pk] = NormalizeSchema(pm)
					} else {
						normalised[pk] = pv
					}
				}
				result[k] = normalised
			} else {
				result[k] = v
			}
		case "items":
			if im, ok := v.(map[string]any); ok {
				result[k] = NormalizeSchema(im)
			} else if ia, ok := v.([]any); ok && len(ia) > 0 {
				// Gemini only supports a single item schema
				if im, ok := ia[0].(map[string]any); ok {
					result[k] = NormalizeSchema(im)
				}
			}
		case "anyOf":
			if arr, ok := v.([]any); ok {
				normalised := make([]any, 0, len(arr))
				for _, item := range arr {
					if im, ok := item.(map[string]any); ok {
						normalised = append(normalised, NormalizeSchema(im))
					} else {
						normalised = append(normalised, item)
					}
				}
				result[k] = normalised
			} else {
				result[k] = v
			}
		default:
			result[k] = v
		}
	}
	return result
}
