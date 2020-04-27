import os
import sys
from pathlib import Path

# Where is this file?
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

__parent__ = Path(__location__).parent.parent

sys.path.append(__location__)
sys.path.append(__parent__)
sys.path.append(os.path.join(__parent__, 'scripts'))

# Import allows .pkl files to be imported without importing from 
# Session script in every notebook
from Session import Session, SessionLite
