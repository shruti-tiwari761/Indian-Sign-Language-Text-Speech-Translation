from django.urls import path
from . import views  # Import views from the same app

urlpatterns = [
    path("", views.index, name="home"),
    path("learning/", views.learning, name="learning"),
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup, name="signup"),
    path("profile/", views.profile, name="profile"),
    path("logout/", views.logout_view, name="logout"),
    path("aboutus/", views.aboutus, name="aboutus"),
    path("contact/", views.contact, name="contact"),
    path("prediction/", views.prediction, name="prediction"),
    path("predict/", views.predict, name="predict"),
    path("feedback/", views.feedback, name="feedback"),
]
