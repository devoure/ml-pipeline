from django.shortcuts import render
from .utils import get_data


# Create your views here.
def home(request):
    context = get_data()

    return render(request, 'home.html', context)

def predict(request):
    pass
