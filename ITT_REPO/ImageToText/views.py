import os
import sys
from django.shortcuts import render, redirect
from django.conf import settings
from django.utils import timezone
from .forms import UploadImageForm
from .models import ProcessedImageDetail

sys.path.append('/home/ncdbproj/NCDBContent/ImageToText')
# Import the process_image function from ml_processing
try:
    from ml_processing import process_image
    print("ML processing module import successful")
except Exception as e:
    print("ML processing module import failed:", e)

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            
            # Save the image to the specified directory
            image_path = os.path.join('/home/ncdbproj/NCDBContent/ImageToText/uploaded_images', image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Handle the case where the user is not logged in
            uploader = request.user if request.user.is_authenticated else None

            # Save image details to the database
            image_instance = ProcessedImageDetail.objects.create(
                imageurl=image_path,
                uploader=uploader,
                uploaded_date=timezone.now(),
            )
            image_instance.save()

            # Call the ML script to process the image and get the car name
            car_model_name = process_image(image_path)
            image_instance.car_model = car_model_name
            image_instance.save()

            return redirect('image_detail', pk=image_instance.pk)
    else:
        form = UploadImageForm()
    return render(request, 'ImageToText/upload_image.html', {'form': form})

def image_detail(request, pk):
    car_image = ProcessedImageDetail.objects.get(pk=pk)
    return render(request, 'ImageToText/image_detail.html', {'car_image': car_image})

