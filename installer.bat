pyinstaller -y -D service.py ^
-p .\interface\python\interface.py ^
-p .\modules\*.py

XCOPY .\cmake-build-release .\dist\service\cmake-build-release /E /Y /I &
XCOPY .\models .\dist\service\models /E /Y /I
XCOPY .\setting.json .\dist\service\setting.json /E /Y /I