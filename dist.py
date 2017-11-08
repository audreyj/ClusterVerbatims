"""
file = wx2exeZ.py
Py2Exe (version 6.6 and higher) setup file for wxPython GUI programs.
Creates an exe file and a library.zip file.
It's easiest to save this wx2exeZ.py file into the same folder
folder with the source file and needed image files.
Give your correct source file name and run wx2exeZ.py ...
Two subfolders are created called dist and build.
The build folder is temporary and for info and can be deleted.
Distribute whatever is in the dist folder.
The dist folder contains your .exe file, library.zip, MSVCR71.dll
and w9xpopen.exe
Your library.zip file contains your optimized byte code, all needed
modules, the Python interpreter (eg. Python25.dll).  Note that with
Python/wxPython programs you might be able to share the large zip
library file with other similar exe files.
MSVCR71.dll can be distributed and is often already in the
Windows/system32 folder.  Python26+ uses a higher version MSVCR90.dll
w9xpopen.exe is needed for os.popen() only, can be deleted otherwise.
"""
from distutils.core import setup
import py2exe
import sys
# important >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# enter the filename of your wxPython code file to compile
filename = "VerbatimSorter.py"
# this creates the filename of your .exe file in the dist folder
if filename.endswith(".py"):
    distribution = filename[:-3]
elif filename.endswith(".pyw"):
    distribution = filename[:-4]
# if run without args, build executables in quiet mode
if len(sys.argv) == 1:
    sys.argv.append("py2exe")
    sys.argv.append("-q")
class Target:
    """ edit the version/name info as needed """
    def __init__(self, **kw):
        self.__dict__.update(kw)
        # for the versioninfo resources, edit to your needs
        self.version = "0.0.1"
        self.company_name = "AJC"
        self.copyright = "no copyright"
        # fill this in with your own program description
        self.name = "VC"
# start of manifest for custom icon and appearance ...
#
# This XML manifest will be inserted as resource into your .exe file
# It gives the controls a Windows XP appearance (if run on XP)
#
manifest_template = '''
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
<assemblyIdentity
    version="5.0.0.0"
    processorArchitecture="x86"
    name="%(prog)s"
    type="win32"
/>
<description>%(prog)s Program</description>
<dependency>
    <dependentAssembly>
        <assemblyIdentity
            type="win32"
            name="Microsoft.Windows.Common-Controls"
            version="6.0.0.0"
            processorArchitecture="X86"
            publicKeyToken="6595b64144ccf1df"
            language="*"
        />
    </dependentAssembly>
</dependency>
</assembly>
'''
RT_MANIFEST = 24
# description is the versioninfo resource
# script is the wxPython code file
# manifest_template is the above XML code
# distribution will be the exe filename
# icon_resource is optional, remove any comment and give it
# an iconfile you have, otherwise a default icon is used
# dest_base will be the exe filename
test_wx = Target(
    description = "Verbatim Categorizer",
    script = filename,
    other_resources = [(RT_MANIFEST, 1, manifest_template % dict(prog=distribution))],
    # remove comment below if you want to use your own icon
    #icon_resources = [(1, "icon.ico")],
    dest_base = distribution)
# end of manifest
setup(
    options = {"py2exe": {"compressed": 1,
                          "optimize": 2,
                          "packages": ['wx'],
                          "bundle_files": 1}},
    # remove comment below to put all data into single exe file
    # zipfile = None,
    # to add .dll or image files use list of filnames
    # these files are added to the dir containing the exe file
    #data_files = ['./images/red.jpg', './images/'blue.jpg'],
    windows = [test_wx]
)