#!/usr/bin/env python3
"""
Shader Source Synchronization Script
====================================

This script reads GLSL shader files (Vertex.glsl and Fragment.glsl) and updates
the corresponding shader source code in main.js, replacing the embedded shader strings.

Usage: python syncshadersource.py
"""

import re
import os
import sys

def read_file(filepath):
    """Read a file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        return None

def write_file(filepath, content):
    """Write content to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing to '{filepath}': {e}")
        return False

def replace_shader_source(js_content, shader_content, shader_type):
    """
    Replace shader source in JavaScript content.
    
    Args:
        js_content: The JavaScript file content
        shader_content: The new shader source code
        shader_type: Either 'vertex' or 'fragment'
    
    Returns:
        Updated JavaScript content
    """
    # Define the variable name to look for
    if shader_type == 'vertex':
        var_name = 'vertexShaderSource'
    elif shader_type == 'fragment':
        var_name = 'fragmentShaderSource'
    else:
        raise ValueError("shader_type must be 'vertex' or 'fragment'")
    
    # Pattern to match the shader source assignment
    # Matches: const vertexShaderSource = `...`.trim();
    # or: const fragmentShaderSource = `...`.trim();
    pattern = rf'(const {var_name} = `)(.*?)(`\.trim\(\);)'
    
    # Replace the content between the backticks
    def replacement(match):
        return f"{match.group(1)}{shader_content}{match.group(3)}"
    
    # Perform the replacement using DOTALL flag to match across newlines
    updated_content = re.sub(pattern, replacement, js_content, flags=re.DOTALL)
    
    # Check if replacement was made
    if updated_content == js_content:
        print(f"Warning: No {shader_type} shader source found to replace in main.js")
        return js_content
    
    return updated_content

def main():
    """Main function to sync shader sources."""
    print("Syncing shader sources...")
    
    # File paths
    vertex_shader_file = 'Vertex.glsl'
    fragment_shader_file = 'Fragment.glsl'
    main_js_file = 'main.js'
    
    # Check if all required files exist
    required_files = [vertex_shader_file, fragment_shader_file, main_js_file]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"Error: Required file '{filepath}' not found.")
            print("Make sure you're running this script in the directory containing:")
            print("  - Vertex.glsl")
            print("  - Fragment.glsl") 
            print("  - main.js")
            sys.exit(1)
    
    # Read shader files
    print("Reading shader files...")
    vertex_content = read_file(vertex_shader_file)
    fragment_content = read_file(fragment_shader_file)
    
    if vertex_content is None or fragment_content is None:
        sys.exit(1)
    
    # Read main.js
    print("Reading main.js...")
    js_content = read_file(main_js_file)
    if js_content is None:
        sys.exit(1)
    
    # Create backup
    backup_file = 'main.js.backup'
    print(f"Creating backup: {backup_file}")
    if not write_file(backup_file, js_content):
        print("Failed to create backup. Aborting.")
        sys.exit(1)
    
    # Replace vertex shader source
    print("Updating vertex shader source...")
    updated_js = replace_shader_source(js_content, vertex_content, 'vertex')
    
    # Replace fragment shader source
    print("Updating fragment shader source...")
    updated_js = replace_shader_source(updated_js, fragment_content, 'fragment')
    
    # Write updated main.js
    print("Writing updated main.js...")
    if write_file(main_js_file, updated_js):
        print("Successfully updated shader sources in main.js!")
        print(f"Backup saved as: {backup_file}")
        
        # Show summary
        vertex_lines = len(vertex_content.splitlines())
        fragment_lines = len(fragment_content.splitlines())
        print(f"Summary:")
        print(f"   - Vertex shader: {vertex_lines} lines")
        print(f"   - Fragment shader: {fragment_lines} lines")
        
    else:
        print("Failed to write updated main.js")
        sys.exit(1)

if __name__ == "__main__":
    main()
