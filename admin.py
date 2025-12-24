from django.contrib import admin

# Register your models here.
from .models import Project, Source

admin.site.register(Project)
admin.site.register(Source)
