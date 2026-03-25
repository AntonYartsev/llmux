package transform

import (
	"reflect"
	"testing"
)

func TestNormalizeSchema_TypeConversion(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"string", "STRING"},
		{"integer", "INTEGER"},
		{"number", "NUMBER"},
		{"boolean", "BOOLEAN"},
		{"array", "ARRAY"},
		{"object", "OBJECT"},
		{"String", "STRING"},
	}
	for _, tt := range tests {
		result := NormalizeSchema(map[string]any{"type": tt.input})
		got, ok := result["type"].(string)
		if !ok {
			t.Errorf("type=%q: result[\"type\"] is not a string: %v", tt.input, result["type"])
			continue
		}
		if got != tt.expected {
			t.Errorf("type=%q: got %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestNormalizeSchema_RemovesUnsupportedKeys(t *testing.T) {
	input := map[string]any{
		"type":                 "object",
		"$ref":                 "#/definitions/Foo",
		"$defs":                map[string]any{"Foo": map[string]any{"type": "string"}},
		"additionalProperties": false,
		"$schema":              "http://json-schema.org/draft-07/schema#",
		"definitions":          map[string]any{},
		"title":                "MySchema",
	}
	result := NormalizeSchema(input)

	for _, key := range []string{"$ref", "$defs", "additionalProperties", "$schema", "definitions"} {
		if _, exists := result[key]; exists {
			t.Errorf("unsupported key %q should have been removed but is present", key)
		}
	}
	if result["title"] != "MySchema" {
		t.Errorf("expected title=MySchema, got %v", result["title"])
	}
	if result["type"] != "OBJECT" {
		t.Errorf("expected type=OBJECT, got %v", result["type"])
	}
}

func TestNormalizeSchema_RecursiveProperties(t *testing.T) {
	input := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":                 "string",
				"additionalProperties": true,
			},
			"age": map[string]any{
				"type": "integer",
			},
		},
	}
	result := NormalizeSchema(input)

	props, ok := result["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties should be map[string]any, got %T", result["properties"])
	}

	nameProp, ok := props["name"].(map[string]any)
	if !ok {
		t.Fatalf("name property should be map[string]any, got %T", props["name"])
	}
	if nameProp["type"] != "STRING" {
		t.Errorf("name.type: got %v, want STRING", nameProp["type"])
	}
	if _, exists := nameProp["additionalProperties"]; exists {
		t.Error("name.additionalProperties should have been stripped")
	}

	ageProp, ok := props["age"].(map[string]any)
	if !ok {
		t.Fatalf("age property should be map[string]any, got %T", props["age"])
	}
	if ageProp["type"] != "INTEGER" {
		t.Errorf("age.type: got %v, want INTEGER", ageProp["type"])
	}
}

func TestNormalizeSchema_RecursiveItems(t *testing.T) {
	input := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type":  "string",
			"$defs": map[string]any{},
		},
	}
	result := NormalizeSchema(input)

	if result["type"] != "ARRAY" {
		t.Errorf("type: got %v, want ARRAY", result["type"])
	}
	items, ok := result["items"].(map[string]any)
	if !ok {
		t.Fatalf("items should be map[string]any, got %T", result["items"])
	}
	if items["type"] != "STRING" {
		t.Errorf("items.type: got %v, want STRING", items["type"])
	}
	if _, exists := items["$defs"]; exists {
		t.Error("items.$defs should have been stripped")
	}
}

func TestNormalizeSchema_RecursiveAnyOf(t *testing.T) {
	input := map[string]any{
		"anyOf": []any{
			map[string]any{"type": "string", "$ref": "#/foo"},
			map[string]any{"type": "integer"},
		},
	}
	result := NormalizeSchema(input)

	anyOf, ok := result["anyOf"].([]any)
	if !ok {
		t.Fatalf("anyOf should be []any, got %T", result["anyOf"])
	}
	if len(anyOf) != 2 {
		t.Fatalf("anyOf should have 2 elements, got %d", len(anyOf))
	}

	first, ok := anyOf[0].(map[string]any)
	if !ok {
		t.Fatalf("anyOf[0] should be map[string]any, got %T", anyOf[0])
	}
	if first["type"] != "STRING" {
		t.Errorf("anyOf[0].type: got %v, want STRING", first["type"])
	}
	if _, exists := first["$ref"]; exists {
		t.Error("anyOf[0].$ref should have been stripped")
	}

	second, ok := anyOf[1].(map[string]any)
	if !ok {
		t.Fatalf("anyOf[1] should be map[string]any, got %T", anyOf[1])
	}
	if second["type"] != "INTEGER" {
		t.Errorf("anyOf[1].type: got %v, want INTEGER", second["type"])
	}
}

func TestNormalizeSchema_PassthroughUnknownKeys(t *testing.T) {
	input := map[string]any{
		"title":       "My title",
		"description": "Some description",
		"enum":        []any{"foo", "bar"},
		"required":    []any{"name"},
	}
	result := NormalizeSchema(input)

	if !reflect.DeepEqual(result, input) {
		t.Errorf("unknown keys should pass through unchanged\ngot:  %v\nwant: %v", result, input)
	}
}
