from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import SignUpForm, LoginForm
from django.contrib.auth.forms import UserCreationForm
from django.views.decorators.csrf import csrf_exempt
import torch
import cv2
import numpy as np
import os
from PIL import Image
import io
from ultralytics import YOLO
import ollama
import tempfile
import json
from pathlib import Path
from .models import Feedback

# Load models once at module level
MODEL = None
MODEL_PATH = None
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_models():
    global MODEL, MODEL_PATH
    try:
        # Get the model paths
        MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best (5).pt')
        
        print(f"Attempting to load model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: YOLO model file not found at {MODEL_PATH}")
            return False
            
        print("Loading YOLO model...")
        MODEL = YOLO(MODEL_PATH)
        
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def index(request):
    return render(request, "home/home.html")

def learning(request):
    return render(request, "home/learning.html")

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                return redirect('profile')
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    return render(request, "home/login.html", {'form': form})

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('profile')
    else:
        form = SignUpForm()
    return render(request, "home/signup.html", {'form': form})

@login_required
def profile(request):
    user = request.user
    feedbacks = Feedback.objects.filter(user=user).order_by('-created_at')
    return render(request, "home/profile.html", {'user': user, 'feedbacks': feedbacks})

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')

def aboutus(request):
    return render(request, "home/aboutus.html")

def contact(request):
    return render(request, "home/contact.html")

@login_required(login_url='/login/')
def prediction(request):
    # Try to load the model if it's not already loaded
    if MODEL is None and not load_models():
        messages.error(request, "Failed to load detection model. Please contact administrator.")
    return render(request, 'home/prediction.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # Load model if not already loaded
        global MODEL
        if MODEL is None:
            print("Model not loaded, attempting to load...")
            if not load_models():
                print("Failed to load model")
                return JsonResponse({'error': 'Model failed to load'}, status=500)
        
        try:
            # Get the image from the request
            image_file = request.FILES.get('image')
            if not image_file:
                print("No image file received in request")
                return JsonResponse({'error': 'No image provided'}, status=400)

            print(f"Received image file: {image_file.name}, size: {image_file.size} bytes")
            
            # Process the image
            try:
                image = Image.open(image_file).convert('RGB')
                print(f"Image opened successfully, size: {image.size}, mode: {image.mode}")
                
                # Convert to numpy array for OpenCV processing
                img_np = np.array(image)
                print(f"Converted to numpy array, shape: {img_np.shape}, dtype: {img_np.dtype}")
                
                # Run YOLO inference with verbose output
                print("Running model inference...")
                results = MODEL.predict(image, verbose=True, conf=0.5)  # Increased confidence threshold
                print("Inference completed")
                
                # Process results
                filtered_predictions = []
                CONFIDENCE_THRESHOLD = 0.5  # Increased threshold for better accuracy
                
                if not results:
                    print("No results returned from model prediction")
                    return JsonResponse({'predictions': [], 'message': 'No results from model prediction'})
                
                for result in results:
                    if not hasattr(result, 'boxes'):
                        print("Result object has no 'boxes' attribute")
                        continue
                        
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        print("No boxes detected in result")
                        continue
                        
                    print(f"Found {len(boxes)} boxes in result")
                    print(f"Raw predictions: {result.boxes.cls.tolist()}")  # Added logging of raw predictions
                    
                    for box in boxes:
                        try:
                            confidence = float(box.conf[0])
                            print(f"Box confidence: {confidence}")
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                class_id = int(box.cls[0])
                                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"
                                print(f"Detected class: {class_name} (ID: {class_id}) with confidence: {confidence}")
                                
                                filtered_predictions.append({
                                    'sign': class_name,
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'type': 'number' if class_name.isdigit() else 'letter'
                                })
                        except Exception as box_error:
                            print(f"Error processing box: {str(box_error)}")
                            continue
                
                print(f"Final filtered predictions: {filtered_predictions}")
                
                if filtered_predictions:
                    return JsonResponse({'predictions': filtered_predictions})
                else:
                    print("No predictions above confidence threshold")
                    return JsonResponse({'predictions': [], 'message': 'No hand gesture detected'})
                    
            except Exception as image_error:
                print(f"Error processing image: {str(image_error)}")
                import traceback
                print(traceback.format_exc())
                return JsonResponse({'error': f'Image processing error: {str(image_error)}'}, status=500)
                
        except Exception as e:
            import traceback
            print(f"ERROR during prediction: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'error': f'Error during prediction: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
def feedback(request):
    if request.method == 'POST':
        rating = request.POST.get('rating')
        comment = request.POST.get('comment')
        
        if rating and comment:
            Feedback.objects.create(
                user=request.user,
                rating=rating,
                comment=comment
            )
            return JsonResponse({'status': 'success'})
        return JsonResponse({'status': 'error', 'message': 'Please provide both rating and comment'})
    
    return render(request, 'home/feedback.html') 