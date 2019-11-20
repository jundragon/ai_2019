###############
# Django ORM Standalone
import os
from django.core.wsgi import get_wsgi_application

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "data.settings")
# Ensure settings are read
application = get_wsgi_application()
#################

# Your application specific imports

if __name__ == "__main__":

    # Application logic
    # first = job.objects.all()[0]

    # print(first.mode)
    print("DB connect")