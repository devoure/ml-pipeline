from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html', {"number":[1,2,3,4,5,6]})
