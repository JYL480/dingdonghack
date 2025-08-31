from dotenv import load_dotenv
from config import ADB_PATH
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
import os
import base64
import datetime
from time import sleep
import subprocess
from ppadb.client import Client as AdbClient


# Load environment variables
load_dotenv()

# Initialize LangChain OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("CHATGPT_KEY"),
    streaming=True
)

async def return_llm_output(user_input: str) -> str:
    messages = [
    SystemMessage(content=(
        "You are an expert UI testing agent specializing in evaluating the usability, design, and functionality of user interfaces. "
        "Your goal is to analyze UI components, identify usability issues, and suggest improvements. "
        "Communicate your observations clearly, offering actionable insights without unnecessary verbosity. "
        "When rephrasing, incorporate all user feedback while ensuring clarity and precision."
    )),
    HumanMessage(content=f"{user_input}")
    ]
    
    response = await llm.ainvoke(messages)
    return response.content

# Helper function to encode a local image file to a base64 string
def encode_image(image_path: str) -> str:
    """Encodes a local image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def return_llm_image_cap_output(user_input: str, image_path: str):
    """Pass the image to the LLM for processing and return a 'yes' or 'no' response."""
    
    # Encode the image to base64
    base64_image = encode_image(image_path)
    
    # Prepare the system message to instruct the model to respond with 'yes' or 'no'
    messages = [
        SystemMessage(
            content="""You are an AI assistant that analyzes UI elements in images. 
                        Based on the question, please provide a clear and concise answer with only 'yes' or 'no'.
                        Do not include additional explanations or commentary.
                    """
        ),
        HumanMessage(
            content=[ 
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ]
        )
    ]
    
    # Invoke the LLM with the multi-modal message (image + text)
    response = await llm.ainvoke(messages)
    
    # Return the content of the response, which should be 'yes' or 'no'
    return response.content.strip().lower()

# Define a function to execute ADB commands
def execute_adb(adb_command):
    adb_command = adb_command.replace("adb", ADB_PATH, 1)
    result = subprocess.run(
        adb_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"Command execution failed: {adb_command}")
    print(result.stderr)
    return "ERROR"

def take_screenshot_ui(
    device: str = "emulator",
    save_dir: str = "./log/screenshots",  # Default save directory
    app_name: str = None,
    step: int = 0,
) -> str:
    """
    Take a screenshot of the specified mobile device (Android emulator or real device) and save it to a directory organized by application.

    Parameters:
        - device (str): Specify the target device ID, default is "emulator". You can view connected devices using the `list_all_devices` tool.
        - save_dir (str): Directory path to save the screenshot locally, default is "./screenshots" in the current directory.
        - app_name (str): Name of the current application, used to organize subdirectories for saving screenshots.
        - step (int): Step number of the current operation, used to generate the filename.

    Returns:
        - Success: Returns the specific path string where the screenshot is saved.
        - Failure: Returns an error message string, such as "Screenshot failed, please check device connection or permissions".
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if app_name is None:
        app_name = "unknown_app"

    # Create a subdirectory organized by application
    app_dir = os.path.join(save_dir, app_name)
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a filename including the application name, step number, and timestamp
    if step is not None:
        filename = f"{app_name}_step{step}_{timestamp}.png"
    else:
        filename = f"{app_name}_{timestamp}.png"

    screenshot_file = os.path.join(app_dir, filename)
    remote_file = f"/sdcard/{filename}"

    # Construct ADB commands
    cap_command = f"adb -s {device} shell screencap -p {remote_file}"
    pull_command = f"adb -s {device} pull {remote_file} {screenshot_file}"
    delete_command = f"adb -s {device} shell rm {remote_file}"

    sleep(3)
    # Execute screenshot command
    try:
        if execute_adb(cap_command) != "ERROR":
            if execute_adb(pull_command) != "ERROR":
                execute_adb(
                    delete_command
                )  # Delete temporary screenshot file from device
                return f"{screenshot_file}"
    except Exception as e:
        return f"Screenshot failed, error information: {str(e)}"

    return "Screenshot failed. Please check device connection or permissions."

def record_video(
    device: str = "emulator",
    save_dir: str = "./log/videos",  # Default save directory
    app_name: str = None,
    step: int = 0,
    duration: int = 3  # Duration of the video in seconds (default 30 seconds)
) -> str:
    """
    Record a video of the specified mobile device (Android emulator or real device) and save it to a directory organized by application.

    Parameters:
        - device (str): Specify the target device ID, default is "emulator". You can view connected devices using the `list_all_devices` tool.
        - save_dir (str): Directory path to save the video locally, default is "./videos" in the current directory.
        - app_name (str): Name of the current application, used to organize subdirectories for saving videos.
        - step (int): Step number of the current operation, used to generate the filename.
        - duration (int): Duration of the video recording in seconds (default is 30 seconds).

    Returns:
        - Success: Returns the specific path string where the video is saved.
        - Failure: Returns an error message string, such as "Video recording failed, please check device connection or permissions".
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if app_name is None:
        app_name = "unknown_app"

    # Create a subdirectory organized by application
    app_dir = os.path.join(save_dir, app_name)
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a filename including the application name, step number, and timestamp
    if step is not None:
        filename = f"{app_name}_step{step}_{timestamp}.mp4"
    else:
        filename = f"{app_name}_{timestamp}.mp4"

    video_file = os.path.join(app_dir, filename)
    remote_file = f"/sdcard/{filename}"

    # Construct ADB commands
    record_command = f"adb -s {device} shell screenrecord --time-limit {duration} {remote_file}"
    pull_command = f"adb -s {device} pull {remote_file} {video_file}"
    delete_command = f"adb -s {device} shell rm {remote_file}"

    sleep(2)  # Wait for the command to be ready
    try:
        # Start screen recording
        if execute_adb(record_command) != "ERROR":
            sleep(duration + 2)  # Wait for the recording to finish and buffer to settle
            if execute_adb(pull_command) != "ERROR":
                execute_adb(delete_command)  # Delete temporary video file from device
                return f"{video_file}"
    except Exception as e:
        return f"Video recording failed, error information: {str(e)}"

    return "Video recording failed. Please check device connection or permissions."

def get_connected_devices():
    """
    Get the list of connected devices/emulators using the AdbClient from pure-python-adb.
    """
    # Initialize the ADB client
    client = AdbClient(host="127.0.0.1", port=5037)  # Default ADB server location

    # Get a list of devices connected to ADB
    devices = client.devices()

    if devices:
        # Return a list of device IDs
        return [device.serial for device in devices]
    else:
        return []

# Test the function
if __name__ == "__main__":
    device = "emulator-5554"
    record_video(device=device, app_name="example_app", step=1, duration=5)