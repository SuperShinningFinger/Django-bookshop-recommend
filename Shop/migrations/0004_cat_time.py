# Generated by Django 2.1.1 on 2019-02-25 14:42

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('Shop', '0003_cat'),
    ]

    operations = [
        migrations.AddField(
            model_name='cat',
            name='time',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]