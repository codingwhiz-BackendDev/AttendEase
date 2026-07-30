with open("templates/millialms_lecturer_dashboard.html", "r", encoding="utf-8") as f:
    content = f.read()

# Fix line 1: split extends and block
content = content.replace(
    "{% extends 'base.html' %} {% block content %}",
    "{% extends 'base.html' %}\n{% block content %}",
    1,
)

# Fix the split for loop
content = content.replace(
    "{% for course in\n                                assessment.courses.all|slice:':4' %}",
    "{% for course in assessment.courses.all|slice:':4' %}",
)

# Fix the split if statement
content = content.replace(
    "{% endfor %} {% if assessment.courses.all|length\n                                > 4 %}",
    "{% endfor %}\n                                {% if assessment.courses.all|length > 4 %}",
)

with open("templates/millialms_lecturer_dashboard.html", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed!")
