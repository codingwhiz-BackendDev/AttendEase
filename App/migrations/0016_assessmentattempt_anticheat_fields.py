from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("App", "0015_studentprofile_face_embedding_remove_face_image"),
    ]

    operations = [
        migrations.AddField(
            model_name="assessmentattempt",
            name="violation_count",
            field=models.PositiveIntegerField(
                default=0,
                help_text="Total number of anti-cheat violations detected.",
            ),
        ),
        migrations.AddField(
            model_name="assessmentattempt",
            name="violation_log",
            field=models.JSONField(
                blank=True,
                default=list,
                help_text="Timestamped log of each violation event.",
            ),
        ),
        migrations.AddField(
            model_name="assessmentattempt",
            name="flagged",
            field=models.BooleanField(
                default=False,
                help_text="Set to True when violation threshold is exceeded.",
            ),
        ),
        migrations.AddIndex(
            model_name="assessmentattempt",
            index=models.Index(fields=["flagged"], name="app_assessm_flagged_idx"),
        ),
    ]
