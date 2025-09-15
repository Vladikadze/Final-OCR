import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
from pytesseract import Output
import re
import string
import os
import pandas as pd
import openai
import base64
import logging

# Configure OpenAI API Key (ensure you have set this securely)
# openai.api_key = os.environ.get("OPENAI_API_KEY") # Recommended to use environment variables or Colab secrets

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the heuristic check function (will be replaced by Tesseract confidence)
def is_satisfactory(text, probability_min=75):
    """
    Heuristic check to see if the text is 'real' based on structure and readability.
    """
    try:
        if not text or len(text) < 10:
            return False, 0

        punctuation_ratio = sum(c in string.punctuation for c in text) / len(text)
        vowel_count = len(re.findall(r"[aeiouAEIOU]", text))
        word_count = len(text.split())

        score = 0
        score += 30 if vowel_count >= word_count else 0
        score += 30 if word_count >= 3 else 0
        score += 20 if punctuation_ratio < 0.2 else 0
        score += 20 if re.search(r"[A-Z][a-z]+", text) else 0

        return score >= probability_min, score
    except Exception as e:
        logging.error(f"Heuristic check failed: {e}")
        return False, 0

# Define image processing and OCR functions
def get_image_metadata(image_pil):
    """Collects metadata from a Pillow Image object."""
    try:
        gray_image = image_pil.convert("L")
        arr = np.array(gray_image)
        metadata = {
            "width": image_pil.width,
            "height": image_pil.height,
            "brightness": arr.mean(),
            "contrast": arr.std(),
            # "is_blank": arr.mean() > 245, # Old blank check
        }
        # Add the new blank check based on tiles
        metadata["is_blank"] = is_image_blank_tiled(image_pil)
        return metadata
    except Exception as e:
        logging.error(f"Failed to get image metadata: {e}")
        return {"width": 0, "height": 0, "brightness": 0, "contrast": 0, "is_blank": True} # Return default on error

# New function for tiled blank check
def is_image_blank_tiled(image_pil, tile_size=100, brightness_threshold=245, blank_tile_ratio_threshold=0.95):
    """
    Checks if an image is blank by tiling and analyzing brightness.
    An image is considered blank if a high percentage of tiles exceed a brightness threshold.
    """
    try:
        gray_image = image_pil.convert("L")
        arr = np.array(gray_image)
        height, width = arr.shape
        blank_tiles_count = 0
        total_tiles = 0

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                total_tiles += 1
                tile = arr[y:min(y + tile_size, height), x:min(x + tile_size, width)]
                if np.mean(tile) > brightness_threshold:
                    blank_tiles_count += 1

        if total_tiles == 0: # Avoid division by zero for very small images
            return True if np.mean(arr) > brightness_threshold else False

        blank_tile_ratio = blank_tiles_count / total_tiles
        logging.info(f"Blank tile ratio: {blank_tile_ratio:.2f}")
        return blank_tile_ratio > blank_tile_ratio_threshold

    except Exception as e:
        logging.error(f"Tiled blank check failed: {e}")
        return True # Assume blank on error


def is_image_blank(image_metadata):
    """Checks if the image is blank based on metadata."""
    return image_metadata.get("is_blank", False)

def apply_lightweight_preprocessing(image_pil):
    """Applies initial lightweight preprocessing steps to a Pillow Image."""
    try:
        # Convert Pillow to OpenCV
        image_cv = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh_cv = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

        # Apply denoising
        denoised_cv = cv2.medianBlur(thresh_cv, 3)

        # Convert back to Pillow Image
        preprocessed_image_pil = Image.fromarray(denoised_cv)

        # Apply brightness and contrast adjustments (adapting logic from old fix_image)
        gray_processed = preprocessed_image_pil.convert("L")
        brightness_processed = np.mean(np.array(gray_processed))
        contrast_processed = np.array(gray_processed).std()

        # Simple adjustment logic based on processed image characteristics
        enhancer = ImageEnhance.Brightness(preprocessed_image_pil)
        if brightness_processed < 100: # Lower threshold for processed image
             preprocessed_image_pil = enhancer.enhance(1.2)
        elif brightness_processed > 200: # Higher threshold for processed image
             preprocessed_image_pil = enhancer.enhance(0.8)

        enhancer = ImageEnhance.Contrast(preprocessed_image_pil)
        if contrast_processed < 50: # Lower threshold for processed image
            preprocessed_image_pil = enhancer.enhance(1.5)


        return preprocessed_image_pil
    except Exception as e:
        logging.error(f"Lightweight preprocessing failed: {e}")
        return None # Indicate failure

def apply_heavy_enhancement(image_pil):
    """Applies heavier enhancement steps (placeholder for ESRGAN/Real-ESRGAN)."""
    try:
        # Placeholder: Apply sharpening and OTSU thresholding as a heavier step
        # In a real scenario, this would involve calling an ESRGAN model
        image_cv = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Apply sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_cv = cv2.filter2D(gray_cv, -1, kernel)

        # Apply OTSU's thresholding
        _, enhanced_cv = cv2.threshold(sharpened_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        enhanced_image_pil = Image.fromarray(enhanced_cv)
        return enhanced_image_pil
    except Exception as e:
        logging.error(f"Heavy enhancement failed: {e}")
        return None # Indicate failure

# Modified run_tesseract_ocr to use image_to_data for confidence
def run_tesseract_ocr(image_pil, confidence_threshold=75):
    """
    Runs Tesseract OCR on a Pillow Image, extracts text and confidence,
    and validates the result based on mean confidence.
    """
    try:
        # Use image_to_data to get detailed box and confidence information
        data = pytesseract.image_to_data(image_pil, output_type=Output.DICT)
        text = data.get('text', [])
        confidences = data.get('conf', [])

        # Filter out invalid confidence values (-1) and convert to integers
        valid_confidences = [int(c) for c in confidences if c != -1]

        # Calculate mean confidence
        mean_confidence = np.mean(valid_confidences) if valid_confidences else 0

        # Reconstruct the full text from the words
        full_text = " ".join([word for word, conf in zip(text, confidences) if conf != -1]).strip()

        is_good = mean_confidence >= confidence_threshold

        return full_text, is_good, mean_confidence
    except Exception as e:
        logging.error(f"Tesseract OCR failed: {e}")
        return "", False, 0 # Indicate failure


def image_to_base64(image_path):
    """Converts an image file to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to convert image to base64: {e}")
        return ""


def use_openai_vision(image_path):
    """Runs OpenAI Vision on an image file and extracts text."""
    try:
        logging.info("Using OpenAI Vision API...")
        base64_image = image_to_base64(image_path)

        if not base64_image:
            return ""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image."}, # Focused prompt
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1000
        )

        result = response.choices[0].message.content
        return result
    except Exception as e:
        logging.error(f"OpenAI Vision error: {e}")
        return ""

def run_openai_vision_ocr(image_pil, image_path_for_b64):
    """Runs OpenAI Vision on a Pillow Image (via base64 from path) and validates."""
    # OpenAI Vision typically requires a path for base64 encoding
    try:
        # Save the Pillow image to a temporary file to get a path for base64
        temp_path = "temp_openai_input.png"
        image_pil.save(temp_path)

        text = use_openai_vision(temp_path)

        # Clean up the temporary file
        os.remove(temp_path)

        is_good, confidence = is_satisfactory(text) # Re-using the heuristic for OpenAI validation
        return text, is_good, confidence
    except Exception as e:
        logging.error(f"OpenAI Vision OCR failed: {e}")
        return "", False, 0 # Indicate failure

# --- Main OCR Pipeline Logic ---

# List of image paths to test the pipeline
# Replace with your actual image paths
image_paths_to_test = [
    "/Users/vladislavbogomazov/Desktop/work/page_1.png", 
    "/Users/vladislavbogomazov/Desktop/work/page_2.png",
    "/Users/vladislavbogomazov/Desktop/work/page_3.png",
    "/Users/vladislavbogomazov/Desktop/work/page_4.png",
]

all_log_entries = []
failed_images_dir = "failed_images"
os.makedirs(failed_images_dir, exist_ok=True)
log_file_path = "ocr_pipeline_log.csv"


for image_path in image_paths_to_test:
    print(f"\n--- Processing image: {image_path} ---")
    current_image_log_entries = []
    final_status = "Failed - Manual Review Needed" # Default status
    final_extracted_text = ""
    final_confidence = 0
    successful_attempt_made = False
    image_metadata = {} # Initialize image_metadata

    try:
        # 1. Load the image using Pillow.
        try:
            image_pil_original = Image.open(image_path)
            image_loaded = True
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            image_loaded = False
            current_image_log_entries.append({
                "file": image_path,
                "status": "Error - File Not Found",
                "method": "N/A",
                "confidence": 0,
                "conditions": {},
                "extracted_text": ""
            })

        if image_loaded:
            # 2. Collect image metadata using get_image_metadata.
            image_metadata = get_image_metadata(image_pil_original)
            image_metadata['path'] = image_path # Add path to metadata for logging

            current_image_log_entries.append({
                "file": image_metadata['path'],
                "status": "Metadata Collected",
                "method": "N/A",
                "confidence": None,
                "conditions": image_metadata,
                "extracted_text": ""
            })

            # 3. Check if the image is blank using is_image_blank. If blank, log and skip.
            if is_image_blank(image_metadata):
                print("Image is blank and will be skipped.")
                current_image_log_entries.append({
                    "file": image_metadata['path'],
                    "status": "Skipped - Blank Image",
                    "method": "N/A",
                    "confidence": 0,
                    "conditions": image_metadata,
                    "extracted_text": ""
                })
                final_status = "Skipped - Blank Image"
                successful_attempt_made = True # Consider skipping a success for this pipeline logic
            else:
                print("Image is not blank, proceeding with OCR.")
                current_image_log_entries.append({
                    "file": image_metadata['path'],
                    "status": "Proceeding with OCR",
                    "method": "N/A",
                    "confidence": None,
                    "conditions": image_metadata,
                    "extracted_text": ""
                })

                # 4. Apply lightweight preprocessing using apply_lightweight_preprocessing.
                preprocessed_image_pil = apply_lightweight_preprocessing(image_pil_original.copy()) # Use a copy

                if preprocessed_image_pil:
                    print("Lightweight preprocessing applied.")
                    current_image_log_entries.append({
                        "file": image_metadata['path'],
                        "status": "Lightweight Preprocessing Applied",
                        "method": "N/A",
                        "confidence": None,
                        "conditions": image_metadata,
                        "extracted_text": ""
                    })

                    # 5. Run Tesseract OCR on the preprocessed image using run_tesseract_ocr. Log the result.
                    extracted_text_tesseract_preprocessed, is_satisfactory_tesseract_preprocessed, confidence_tesseract_preprocessed = run_tesseract_ocr(preprocessed_image_pil)

                    print(f"Tesseract OCR (Preprocessed): Satisfactory={is_satisfactory_tesseract_preprocessed}, Confidence={confidence_tesseract_preprocessed:.2f}")

                    current_image_log_entries.append({
                        "file": image_metadata['path'],
                        "status": "Tesseract Preprocessed",
                        "method": "Tesseract",
                        "confidence": confidence_tesseract_preprocessed,
                        "conditions": image_metadata,
                        "extracted_text": extracted_text_tesseract_preprocessed[:500] if extracted_text_tesseract_preprocessed else "" # Truncate text
                    })

                    if is_satisfactory_tesseract_preprocessed:
                        final_status = "Tesseract Preprocessed (Satisfactory)"
                        final_extracted_text = extracted_text_tesseract_preprocessed
                        final_confidence = confidence_tesseract_preprocessed
                        successful_attempt_made = True
                    else:
                         # 6. If the first Tesseract attempt is not satisfactory, apply heavy enhancement.
                        print("First Tesseract attempt not satisfactory, applying heavy enhancement.")
                        enhanced_image_pil = apply_heavy_enhancement(image_pil_original.copy()) # Use a copy

                        if enhanced_image_pil:
                            print("Heavy enhancement applied.")
                            current_image_log_entries.append({
                                "file": image_metadata['path'],
                                "status": "Heavy Enhancement Applied",
                                "method": "N/A",
                                "confidence": None,
                                "conditions": image_metadata,
                                "extracted_text": ""
                            })

                            # 7. Run Tesseract OCR on the enhanced image. Log the result.
                            extracted_text_tesseract_enhanced, is_satisfactory_tesseract_enhanced, confidence_tesseract_enhanced = run_tesseract_ocr(enhanced_image_pil)

                            print(f"Tesseract OCR (Enhanced): Satisfactory={is_satisfactory_tesseract_enhanced}, Confidence={confidence_tesseract_enhanced:.2f}")

                            current_image_log_entries.append({
                                "file": image_metadata['path'],
                                "status": "Tesseract Enhanced",
                                "method": "Tesseract",
                                "confidence": confidence_tesseract_enhanced,
                                "conditions": image_metadata,
                                "extracted_text": extracted_text_tesseract_enhanced[:500] if extracted_text_tesseract_enhanced else "" # Truncate text
                            })

                            if is_satisfactory_tesseract_enhanced:
                                final_status = "Tesseract Enhanced (Satisfactory)"
                                final_extracted_text = extracted_text_tesseract_enhanced
                                final_confidence = confidence_tesseract_enhanced
                                successful_attempt_made = True
                            else:
                                # 8. If both Tesseract attempts not satisfactory, run OpenAI Vision on the original image.
                                print("Both Tesseract attempts not satisfactory, trying OpenAI Vision on original image.")
                                extracted_text_openai_original, is_satisfactory_openai_original, confidence_openai_original = run_openai_vision_ocr(image_pil_original.copy(), image_path) # Pass original PIL and its path

                                print(f"OpenAI Vision (Original): Satisfactory={is_satisfactory_openai_original}, Confidence={confidence_openai_original:.2f}")

                                current_image_log_entries.append({
                                    "file": image_metadata['path'],
                                    "status": "OpenAI Original",
                                    "method": "OpenAI Vision",
                                    "confidence": confidence_openai_original,
                                    "conditions": image_metadata,
                                    "extracted_text": extracted_text_openai_original[:500] if extracted_text_openai_original else "" # Truncate text
                                })

                                if is_satisfactory_openai_original:
                                    final_status = "OpenAI Original (Satisfactory)"
                                    final_extracted_text = extracted_text_openai_original
                                    final_confidence = confidence_openai_original
                                    successful_attempt_made = True
                                else:
                                    # 9. If OpenAI Vision on original not satisfactory, run on enhanced image.
                                    print("OpenAI Vision on original not satisfactory, trying on enhanced image.")
                                    # Ensure enhanced_image_pil exists, regenerate if necessary (though should exist if heavy enhancement ran)
                                    if 'enhanced_image_pil' not in locals() or enhanced_image_pil is None:
                                         print("Enhanced image object not found for final OpenAI, regenerating.")
                                         enhanced_image_pil_for_openai = apply_heavy_enhancement(image_pil_original.copy())
                                    else:
                                         enhanced_image_pil_for_openai = enhanced_image_pil.copy() # Use the enhanced image from step 6


                                    if enhanced_image_pil_for_openai:
                                        extracted_text_openai_enhanced, is_satisfactory_openai_enhanced, confidence_openai_enhanced = run_openai_vision_ocr(enhanced_image_pil_for_openai, image_path) # Pass enhanced PIL and original path for context

                                        print(f"OpenAI Vision (Enhanced): Satisfactory={is_satisfactory_openai_enhanced}, Confidence={confidence_openai_enhanced:.2f}")

                                        current_image_log_entries.append({
                                            "file": image_metadata['path'],
                                            "status": "OpenAI Enhanced",
                                            "method": "OpenAI Vision",
                                            "confidence": confidence_openai_enhanced,
                                            "conditions": image_metadata,
                                            "extracted_text": extracted_text_openai_enhanced[:500] if extracted_text_openai_enhanced else "" # Truncate text
                                        })

                                        if is_satisfactory_openai_enhanced:
                                            final_status = "OpenAI Enhanced (Satisfactory)"
                                            final_extracted_text = extracted_text_openai_enhanced
                                            final_confidence = confidence_openai_enhanced
                                            successful_attempt_made = True

                                    else:
                                         print("Enhanced image could not be generated for the final OpenAI attempt.")
                                         # This case is implicitly handled by successful_attempt_made being False


                # 10. If all attempts fail, log the failure and save the original image.
                if not successful_attempt_made:
                     print("All OCR attempts failed.")
                     # Failure log entry was already added as the default status

                     original_filename = os.path.basename(image_metadata['path'])
                     failed_image_path_full = os.path.join(failed_images_dir, f"failed_{original_filename}")

                     try:
                         image_pil_original.save(failed_image_path_full)
                         print(f"Original image saved to: {failed_image_path_full}")
                         current_image_log_entries.append({
                            "file": image_metadata['path'],
                            "status": "Image Saved for Manual Review",
                            "method": "N/A",
                            "confidence": None,
                            "conditions": image_metadata,
                            "extracted_text": ""
                         })
                     except Exception as e:
                         print(f"Error saving failed image: {e}")
                         current_image_log_entries.append({
                            "file": image_metadata['path'],
                            "status": f"Error Saving Failed Image: {e}",
                            "method": "N/A",
                            "confidence": None,
                            "conditions": image_metadata,
                            "extracted_text": ""
                         })

                else:
                     print(f"OCR successful for {image_path} with status: {final_status}")
                     # A final success log entry could be added here if desired,
                     # but the satisfactory log entry from the successful step is already recorded.


    except Exception as e:
        print(f"An unexpected error occurred processing {image_path}: {e}")
        current_image_log_entries.append({
            "file": image_path,
            "status": f"Error - Unexpected: {e}",
            "method": "N/A",
            "confidence": 0,
            "conditions": image_metadata if image_metadata else {},
            "extracted_text": ""
        })

    # Append logs for the current image to the overall list
    all_log_entries.extend(current_image_log_entries)


# 11. Save the complete list of log entries to the "ocr_pipeline_log.csv" file.
if all_log_entries:
    # Read existing log file if it exists
    if os.path.exists(log_file_path):
        try:
            existing_log_df = pd.read_csv(log_file_path)
            # Ensure consistent columns before concatenating
            combined_log_df = pd.concat([existing_log_df, pd.DataFrame(all_log_entries)], ignore_index=True)
        except Exception as e:
            logging.warning(f"Error reading existing log file, creating new one: {e}")
            combined_log_df = pd.DataFrame(all_log_entries)
    else:
        combined_log_df = pd.DataFrame(all_log_entries)

    # Save the combined log entries
    combined_log_df.to_csv(log_file_path, index=False)
    print(f"\nLog entries saved to {log_file_path}")
    display(combined_log_df)
else:
    print("\nNo log entries were generated.")

# 12. Review the generated "ocr_pipeline_log.csv" (Manual step, noted here for completeness).
print("\nReview 'ocr_pipeline_log.csv' manually to verify pipeline execution and logging.")
