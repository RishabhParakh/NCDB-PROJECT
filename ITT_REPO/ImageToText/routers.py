
#from django.conf import settings

class DatabaseAppsRouter:
    """
    A router to control all database operations on models for different
    applications.
    """

    def db_for_read(self, model, **hints):
        """
        Attempts to read models go to the appropriate database.
        """
        if model._meta.app_label == 'ImageToText':
            return 'ImageToText'
        return 'default'

    def db_for_write(self, model, **hints):
        """
        Attempts to write models go to the appropriate database.
        """
        if model._meta.app_label == 'ImageToText':
            return 'ImageToText'
        return 'default'

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Ensures that the ImageToText app's models get created on the right database.
        """
        if app_label == 'ImageToText':
            return db == 'ImageToText'
        # Ensure that all other apps' models go to the default database.
        return db == 'default'

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the ImageToText app is involved.
        """
        if obj1._meta.app_label == 'ImageToText' or obj2._meta.app_label == 'ImageToText':
            return True
        return None

