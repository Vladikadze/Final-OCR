# Final-OCR

The OCR pipeline follows a cost-optimized decision flow to extract text from images. Here's a breakdown of the steps and decisions:

Load Image and Collect Metadata: The pipeline starts by loading the image once and collecting essential metadata, including dimensions, brightness, contrast, and a tiled-based assessment of blankness.
Check for Blank Image: It first checks if the image is considered blank based on the metadata.
If Blank: The image is skipped, logged as "Skipped - Blank Image", and no further processing occurs for this image.
If Not Blank: The pipeline proceeds with OCR attempts.
Apply Lightweight Preprocessing: Lightweight preprocessing (like adaptive thresholding and denoising) is applied to the image.
First Tesseract Attempt: Tesseract OCR is run on the preprocessed image. The result is validated using the mean confidence score provided by Tesseract's image_to_data output.
If Satisfactory (Confidence >= Threshold): The extracted text and confidence from this attempt are considered the final result. The image is logged as "Tesseract Preprocessed (Satisfactory)", and the pipeline moves to the logging stage for this image.
If Not Satisfactory: The pipeline moves to a heavier enhancement step.
Apply Heavy Enhancement: A heavier enhancement step (currently a placeholder for ESRGAN/Real-ESRGAN, implemented with sharpening and OTSU thresholding) is applied to the original image.
Second Tesseract Attempt: Tesseract OCR is run again, this time on the enhanced image. The result is again validated using the mean Tesseract confidence.
If Satisfactory (Confidence >= Threshold): The extracted text and confidence from this attempt are considered the final result. The image is logged as "Tesseract Enhanced (Satisfactory)", and the pipeline moves to the logging stage for this image.
If Not Satisfactory: The pipeline falls back to using the more expensive OpenAI Vision API.
OpenAI Vision (Original Image): OpenAI Vision is called to extract text from the original image. The result is validated using the heuristic is_satisfactory function (as OpenAI Vision doesn't provide per-word confidence in a readily parsable format in this implementation).
If Satisfactory (Heuristic Score >= Threshold): The extracted text and confidence from this attempt are considered the final result. The image is logged as "OpenAI Original (Satisfactory)", and the pipeline moves to the logging stage for this image.
If Not Satisfactory: The pipeline makes a final attempt using OpenAI Vision on the enhanced image.
OpenAI Vision (Enhanced Image): OpenAI Vision is called on the enhanced image. The result is again validated using the heuristic is_satisfactory function.
If Satisfactory (Heuristic Score >= Threshold): The extracted text and confidence from this attempt are considered the final result. The image is logged as "OpenAI Enhanced (Satisfactory)", and the pipeline moves to the logging stage for this image.
If Not Satisfactory: All automated attempts have failed.
Handle Failure:
The image is logged as "Failed - Manual Review Needed".
The original image is saved to a designated directory (failed_images) for manual inspection.
Logging: Throughout the process, details of each step attempted for an image are recorded and appended to a list of log entries. Finally, all log entries for the processed images are saved to a CSV file (ocr_pipeline_log.csv).
This flow ensures that the cheapest methods (local Tesseract) are attempted first, and the more expensive method (OpenAI Vision) is only used as a last resort, and even then, tried on both original and enhanced images to maximize the chance of a successful extraction before resorting to manual review.
