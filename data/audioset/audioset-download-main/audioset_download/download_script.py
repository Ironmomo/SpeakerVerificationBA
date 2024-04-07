import os
current_working_directory = os.getcwd()
print(current_working_directory)
current_python_interpreter = os.sys.executable
print(current_python_interpreter)
current_python_version = os.sys.version
print(current_python_version)
current_operating_system = os.sys.platform
print(current_operating_system)

from audioset_download import Downloader
d = Downloader(root_path='unbalanced', labels=None, n_jobs=128, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format = 'mp3')
