# Generated by Django 2.1.1 on 2019-02-25 14:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Shop', '0004_cat_time'),
    ]

    operations = [
        migrations.RenameField(
            model_name='cat',
            old_name='standard',
            new_name='numprice',
        ),
    ]
