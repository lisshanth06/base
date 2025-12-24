# check_django.py
import os
import sys
import django
from django.conf import settings

# Setup Django
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Check if settings are configured
try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    django.setup()
    print("✅ Django setup successful")
    
    # Check installed apps
    from django.conf import settings
    print(f"\n📋 Installed Apps:")
    for app in settings.INSTALLED_APPS:
        print(f"  - {app}")
    
    # Check if writer app is properly configured
    from django.apps import apps
    try:
        writer_config = apps.get_app_config('writer')
        print(f"\n✅ Writer app found: {writer_config}")
    except LookupError:
        print("\n❌ Writer app not found in Django apps")
        
except Exception as e:
    print(f"❌ Error: {e}")