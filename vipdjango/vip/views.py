from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.http import HttpResponse


@login_required
def home(request):
    if request.user.groups.filter(name="Professor").exists():
        return redirect("vip:professor_dashboard")
    elif request.user.groups.filter(name="Student").exists():
        return redirect("vip:student_dashboard")
    else:
        return redirect("vip:dashboard")

@login_required
def professor_dashboard(request):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("home")
    return render(request, "vip/professor_dashboard.html")
@login_required
def student_dashboard(request):
    if not request.user.groups.filter(name="Student").exists():
        return redirect("home")
    return render(request, "vip/student_dashboard.html")

@login_required
def dashboard(request):
    return render(request, "vip/dashboard.html")