"""
Coordinate Converter for UI Element Click Accuracy
Handles both OmniParser JSON formats to ensure accurate click positioning
"""

def convert_element_to_click_coordinates(element, device_size):
    """
    Convert element data to accurate click coordinates.
    Handles multiple OmniParser formats:
    - coordinates: Direct [x, y] coordinates
    - shape: absolute pixels {x, y, width, height}
    - bbox dict: {x, y, width, height}
    - bbox list: [x1, y1, x2, y2] (relative or absolute)
    
    Args:
        element: Element dict from JSON
        device_size: Dict with 'width' and 'height' keys
        
    Returns:
        tuple: (x, y) coordinates for click position in pixels
    """
    
    # Note: 'coordinates' field seems to be top-left, not center
    # We should prefer calculating center from bbox for accuracy
    
    # Check if we have the new format with 'shape' field
    if 'shape' in element:
        shape = element['shape']
        # Shape format is {x, y, width, height} in absolute pixels
        # Calculate center point
        center_x = shape['x'] + shape['width'] // 2
        center_y = shape['y'] + shape['height'] // 2
        return (center_x, center_y)
    
    # Check if we have the old format with 'bbox' field
    elif 'bbox' in element:
        bbox = element['bbox']
        
        # Check if bbox is a dict with x, y, width, height (similar to shape)
        if isinstance(bbox, dict) and 'x' in bbox and 'y' in bbox:
            # Dict format with x, y, width, height
            center_x = bbox['x'] + bbox.get('width', 0) // 2
            center_y = bbox['y'] + bbox.get('height', 0) // 2
            return (center_x, center_y)
        
        # Check if bbox is in relative format (values between 0 and 1)
        elif isinstance(bbox, list) and len(bbox) >= 4:
            if all(0 <= val <= 1 for val in bbox):
                # Relative coordinates [x1, y1, x2, y2]
                center_x = int((bbox[0] + bbox[2]) / 2 * device_size['width'])
                center_y = int((bbox[1] + bbox[3]) / 2 * device_size['height'])
            else:
                # Absolute coordinates [x1, y1, x2, y2]
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
            return (center_x, center_y)
    
    # Fallback to default if no valid format found
    print(f"Warning: Could not extract coordinates from element: {element}")
    return (0, 0)


def get_element_bounds(element, device_size):
    """
    Get element boundaries in absolute pixel coordinates.
    
    Args:
        element: Element dict from JSON
        device_size: Dict with 'width' and 'height' keys
        
    Returns:
        tuple: (x1, y1, x2, y2) absolute pixel coordinates
    """
    
    if 'shape' in element:
        shape = element['shape']
        x1 = shape['x']
        y1 = shape['y']
        x2 = shape['x'] + shape['width']
        y2 = shape['y'] + shape['height']
        return (x1, y1, x2, y2)
    
    elif 'bbox' in element:
        bbox = element['bbox']
        if isinstance(bbox, list) and len(bbox) >= 4:
            if all(0 <= val <= 1 for val in bbox):
                # Relative coordinates
                x1 = int(bbox[0] * device_size['width'])
                y1 = int(bbox[1] * device_size['height'])
                x2 = int(bbox[2] * device_size['width'])
                y2 = int(bbox[3] * device_size['height'])
            else:
                # Absolute coordinates
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            return (x1, y1, x2, y2)
    
    return (0, 0, 0, 0)