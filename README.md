# Cost-Optimized OCR Pipeline

This project implements a **cost-optimized OCR (Optical Character Recognition) pipeline** designed to balance accuracy, efficiency, and expense. The pipeline intelligently decides between lightweight preprocessing, open-source OCR (Tesseract), image enhancement, and API-based OCR (OpenAI Vision) based on confidence scores and heuristic validation.

---

## 🚀 Key Features
- **Cost Optimization:** Prioritizes local OCR (Tesseract) before using expensive API calls.  
- **Adaptive Decision Flow:** Escalates from simple preprocessing to advanced enhancement and API OCR only if needed.  
- **Confidence Validation:** Uses Tesseract confidence scores and heuristic checks for OpenAI Vision results.  
- **Logging & Transparency:** Logs every step of the decision process with clear labels (`Skipped`, `Satisfactory`, `Failed`).  
- **Fallback Handling:** Saves unprocessed images for manual review when all automated attempts fail.  

---

## 📋 Pipeline Workflow

1. **Load Image & Collect Metadata**  
   - Reads the image once.  
   - Collects metadata: dimensions, brightness, contrast, blankness (tile-based).  

2. **Check for Blank Image**  
   - If blank → skip and log as `Skipped - Blank Image`.  
   - If not blank → proceed to OCR attempts.  

3. **Lightweight Preprocessing**  
   - Applies adaptive thresholding & denoising.  

4. **First Tesseract Attempt**  
   - Runs Tesseract OCR on preprocessed image.  
   - Validates using mean confidence score.  
   - If **confidence ≥ threshold** → log `Tesseract Preprocessed (Satisfactory)` and finish.  
   - Else → proceed to heavy enhancement.  

5. **Heavy Enhancement**  
   - Applies sharpening + OTSU thresholding (placeholder for ESRGAN/Real-ESRGAN).  

6. **Second Tesseract Attempt**  
   - Runs Tesseract OCR again on enhanced image.  
   - If **confidence ≥ threshold** → log `Tesseract Enhanced (Satisfactory)` and finish.  
   - Else → fallback to API-based OCR.  

7. **OpenAI Vision (Original Image)**  
   - Calls OpenAI Vision OCR on the original image.  
   - Validates using heuristic scoring function.  
   - If **score ≥ threshold** → log `OpenAI Original (Satisfactory)` and finish.  
   - Else → try enhanced image.  

8. **OpenAI Vision (Enhanced Image)**  
   - Calls OpenAI Vision on the enhanced image.  
   - If **score ≥ threshold** → log `OpenAI Enhanced (Satisfactory)` and finish.  
   - Else → mark as failure.  

9. **Failure Handling**  
   - Log as `Failed - Manual Review Needed`.  
   - Save image in `failed_images/` for manual inspection.  

10. **Logging**  
    - Every step and decision is logged.  
    - Results are stored in `ocr_pipeline_log.csv`.  

---

## 📊 Decision Flow Summary
1. **Skip blank images early.**  
2. **Prefer Tesseract (cheap).**  
3. **Use enhancements if needed.**  
4. **Fallback to OpenAI Vision (expensive).**  
5. **Fail gracefully with manual review if all else fails.**  

---

## 📂 Project Structure
