- Python 3.8+ 
- Android Studio 

Please contact us for if you require API key and endpoint access, we will provide you with it for judging purposes! 

1) VENV SETUP
python -m venv venv

.\venv\Scripts\Activate OR source venv/bin/activate

pip install -r requirements.txt

# these have already been added to config.py and are the backend uris, note that if restarted the uri will change!
# Omni_URI = "https://your-omniparser-tunnel-url.trycloudflare.com"
# ImageEmbedding_URI = "https://your-embedding-tunnel-url.trycloudflare.com"


2) ANDROID STUDIO
install https://developer.android.com/studio
NOTE THE PATH YOU INSTALL TO YOU NEED TO LOCATE THE ADB.EXE
for me example dir "C:\Users\nicho\AppData\Local\Android\Sdk\platform-tools\adb.exe"
Directory: C:\Users\nicho\AppData\Local\Android\Sdk\platform-tools


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         28/8/2025   4:09 pm        6641760 adb.exe

take the adb.exe path
add it to Path either 
- through your settings 
- $env:Path += ";C:\Users\nicho\AppData\Local\Android\Sdk\platform-tools" (this is only for the session)
- echo "alias adb='/mnt/c/Users/tytan/AppData/Local/Android/Sdk/platform-tools/adb.exe'" >> ~/.bashrc (for linux/wsl)

then open Android Studio and on the "Welcome to Android Studio" window, you should see
- New Project
- Open
- Clone Repository
> More Actions

Click > More Actions and then "Virtual Device Manager" 

Start the device by pressing the play button. Return back to your IDE and check if it's up (or just check the window)

adb devices  

cd pipeline/

python .\nich_pipeline.py "insert task here"
e,g

python .\nich_pipeline.py "go to the messaging app, identify that there is a star button above the start chat icon and that it is blue/purplish in a white box vs the blue tinted box for the start chat icon, then click the start check icon and check that the create group button is there, then click it"

or a simpler one 

python .\nich_pipeline.py "go to the app store and search for tiktok"
