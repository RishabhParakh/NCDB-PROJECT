from django.urls import path
from Chatbot import views

urlpatterns = [
    path('', views.chat_page),
    path('query/', views.chat),
    path('admin/', views.admin),
    path('admin/answer/', views.answer),
    path('admin/question', views.get_new_question),
]
