from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=200)
    content = models.TextField(blank=True)

    def __str__(self):
        return self.name


class Source(models.Model):
    SOURCE_TYPES = [
        ("text", "Text"),
        ("web", "Web"),
        ("pdf", "PDF"),
        ("audio", "Audio"),
    ]

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    text = models.TextField()
    source_type = models.CharField(max_length=10, choices=SOURCE_TYPES)

    def __str__(self):
        return self.title