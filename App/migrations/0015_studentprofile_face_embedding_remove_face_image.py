from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("App", "0014_alter_lecturerprofile_academic_rank_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="studentprofile",
            name="face_embedding",
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.RemoveField(
            model_name="studentprofile",
            name="face_image",
        ),
    ]
