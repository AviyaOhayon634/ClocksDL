from time_extractor import TimeExtractor

digital_clock_model = "runs/detect/digit_detector4/weights/best.pt"
digital_clock_img = "digital_clock_notebooks/3.jpg"

extractor_instance = TimeExtractor(digital_clock_model)
result = extractor_instance.extract_time(digital_clock_img)

print("-- Extracted Time --")
print(result)