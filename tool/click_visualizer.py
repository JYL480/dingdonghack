"""
Click Visualizer for UI Element Actions
Draws click markers on labeled images to show where actions are performed
"""

import os
import json
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Dict, Any


def draw_click_marker(image_path: str, click_position: Tuple[int, int], 
                      output_path: Optional[str] = None, 
                      marker_color: str = 'red', 
                      marker_size: int = 20) -> str:
    """
    Draw a click marker on an image at the specified position.
    
    Args:
        image_path: Path to the input image
        click_position: (x, y) coordinates where the click occurred
        output_path: Optional path for the output image. If None, overwrites the input
        marker_color: Color of the click marker
        marker_size: Size of the click marker
        
    Returns:
        Path to the saved image with click marker
    """
    try:
        # Simple path handling
        import os
        
        print(f"[DEBUG draw_click_marker] image_path: {image_path}")
        print(f"[DEBUG draw_click_marker] output_path: {output_path}")
        
        # Check if input image exists first
        if not os.path.exists(image_path):
            print(f"[DEBUG draw_click_marker] Input image doesn't exist: {image_path}")
            return image_path  # Return original if file doesn't exist
        
        print(f"[DEBUG draw_click_marker] Opening image...")
        # Open the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        x, y = click_position
        
        # Draw a cross marker at the click position
        # Horizontal line
        draw.line(
            [(x - marker_size, y), (x + marker_size, y)], 
            fill=marker_color, 
            width=3
        )
        # Vertical line
        draw.line(
            [(x, y - marker_size), (x, y + marker_size)], 
            fill=marker_color, 
            width=3
        )
        
        # Draw a circle around the click point
        draw.ellipse(
            [(x - marker_size//2, y - marker_size//2), 
             (x + marker_size//2, y + marker_size//2)],
            outline=marker_color,
            width=3
        )
        
        # Add a small filled circle at the exact click point
        draw.ellipse(
            [(x - 3, y - 3), (x + 3, y + 3)],
            fill=marker_color,
            outline=marker_color
        )
        
        # Save the image
        if output_path is None:
            output_path = image_path
        
        # Ensure directory exists for output
        output_dir = os.path.dirname(output_path)
        print(f"[DEBUG draw_click_marker] output_dir: {output_dir}")
        if output_dir and not os.path.exists(output_dir):
            print(f"[DEBUG draw_click_marker] Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert to absolute path for PIL save on Windows
        abs_output_path = os.path.abspath(output_path)
        print(f"[DEBUG draw_click_marker] Saving to absolute path: {abs_output_path}")
        img.save(abs_output_path)
        print(f"[DEBUG draw_click_marker] Successfully saved!")
        return output_path  # Return the original relative path
        
    except Exception as e:
        print(f"Error drawing click marker: {str(e)}")
        return image_path


def save_action_visualization(labeled_image_path: str, 
                             click_position: Tuple[int, int],
                             action_info: Dict[str, Any],
                             save_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Save a visualization of an action with click marker on the labeled image.
    
    Args:
        labeled_image_path: Path to the labeled bounding box image
        click_position: (x, y) coordinates where the action occurred
        action_info: Dictionary containing action details (type, element_id, etc.)
        save_dir: Directory to save the visualization. If None, uses same dir as labeled image
        
    Returns:
        Dictionary with paths to saved files
    """
    try:
        # Check if the labeled image exists
        if not os.path.exists(labeled_image_path):
            # Silently skip if source image doesn't exist
            return {
                "clicked_image": labeled_image_path,
                "action_json": None,
                "original_labeled": labeled_image_path
            }
        
        # Determine save directory - keep forward slashes consistent
        if save_dir is None:
            save_dir = os.path.dirname(labeled_image_path)
        
        # Don't create subdirectory - save directly in the task folder
        action_viz_dir = save_dir
        os.makedirs(action_viz_dir, exist_ok=True)
        
        # Generate shorter filename to avoid Windows path length issues
        action_type = action_info.get('action', 'unknown')
        element_id = action_info.get('element_id', 'unknown')
        step = action_info.get('step', 0)
        timestamp = action_info.get('timestamp', '')
        
        # Create shorter output filename
        output_filename = f"step{step}_{action_type}_elem{element_id}_clicked.png"
        # Use forward slashes consistently
        output_path = f"{action_viz_dir}/{output_filename}"
        
        # Debug: print paths
        print(f"[DEBUG] labeled_image_path: {labeled_image_path}")
        print(f"[DEBUG] action_viz_dir: {action_viz_dir}")
        print(f"[DEBUG] output_path: {output_path}")
        
        # Draw the click marker on the labeled image
        result_path = draw_click_marker(
            labeled_image_path, 
            click_position, 
            output_path,
            marker_color='red' if action_type == 'tap' else 'blue',
            marker_size=25
        )
        
        # Save action info as JSON - build path from output_path to ensure consistency
        action_json_path = os.path.splitext(output_path)[0] + "_action.json"
        abs_json_path = os.path.abspath(action_json_path)
        action_data = {
            "labeled_image": labeled_image_path,
            "clicked_image": result_path,
            "click_position": {"x": click_position[0], "y": click_position[1]},
            "action_info": action_info,
            "timestamp": action_info.get('timestamp', None)
        }
        
        with open(abs_json_path, 'w', encoding='utf-8') as f:
            json.dump(action_data, f, ensure_ascii=False, indent=2)
        
        return {
            "clicked_image": result_path,
            "action_json": action_json_path,
            "original_labeled": labeled_image_path
        }
        
    except Exception as e:
        print(f"Error saving action visualization: {str(e)}")
        return {
            "clicked_image": labeled_image_path,
            "action_json": None,
            "original_labeled": labeled_image_path
        }