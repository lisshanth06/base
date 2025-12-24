from django.urls import path
from . import views

urlpatterns = [
    path("", views.project_list, name="project_list"),
    path("create/", views.create_project, name="create_project"),
    path("project/<int:project_id>/", views.project_detail, name="project_detail"),

    path("project/<int:project_id>/edit/", views.edit_project, name="edit_project"),
    path("project/<int:project_id>/delete/", views.delete_project, name="delete_project"),

    path("project/<int:project_id>/add-text/", views.add_text_source, name="add_text"),
    path("project/<int:project_id>/web-search/", views.web_search_source, name="web_search"),
    path("project/<int:project_id>/add-audio/", views.add_audio_source, name="add_audio"),

    path("source/<int:source_id>/edit/", views.edit_source, name="edit_source"),
    path("source/<int:source_id>/delete/", views.delete_source, name="delete_source"),
]