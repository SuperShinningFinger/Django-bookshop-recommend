# Generated by Django 2.1.1 on 2019-02-16 09:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Category',
            fields=[
                ('cid', models.AutoField(primary_key=True, serialize=False)),
                ('kind', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='Product',
            fields=[
                ('pid', models.AutoField(primary_key=True, serialize=False)),
                ('pdname', models.CharField(max_length=128)),
                ('pdprice', models.FloatField()),
                ('discount', models.FloatField()),
                ('pdImage', models.TextField()),
                ('pdInfo', models.TextField(default='')),
                ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='category_product', to='Shop.Category')),
            ],
        ),
    ]
