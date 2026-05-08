from ultralytics import YOLO

CONF_ORIG = 0.4
CONF_CROP = 0.5

class TimeExtractor:
    def __init__(self, model_path,verbose=False):
        self.model = YOLO(model_path,verbose=verbose)

    def original_detections(self, image_path) -> list:

        result = self.model(image_path, verbose=False)[0]

        original_detections = []

        for box in result.boxes:
            conf = float(box.conf[0])

            if conf < CONF_ORIG:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            x1,y1,x2,y2 = box.xyxy[0].tolist()
            xc = (x1+x2)/2
            yc = (y1+y2)/2

            original_detections.append((label, conf, [x1,y1,x2,y2], xc, yc, "orig"))

        return original_detections
    
    def sliding_collect(self, image, window_size=416, stride=200):

        h, w = image.shape[:2]
        detections = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):

                crop = image[y:y+window_size, x:x+window_size]

                if crop.shape[0] < 50 or crop.shape[1] < 50:
                    continue

                results = self.model(crop, verbose=False)[0]

                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_CROP:
                        continue

                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]

                    x1,y1,x2,y2 = box.xyxy[0].tolist()

                    x1 += x
                    x2 += x
                    y1 += y
                    y2 += y

                    xc = (x1 + x2)/2
                    yc = (y1 + y2)/2

                    detections.append((label, conf, [x1,y1,x2,y2], xc, yc, "crop"))

        return detections

    @staticmethod
    def compute_iou(box1, box2):
        x1,y1,x2,y2 = box1
        x1b,y1b,x2b,y2b = box2

        inter_x1 = max(x1,x1b)
        inter_y1 = max(y1,y1b)
        inter_x2 = min(x2,x2b)
        inter_y2 = min(y2,y2b)

        inter = max(0, inter_x2-inter_x1)*max(0, inter_y2-inter_y1)

        a1 = (x2-x1)*(y2-y1)
        a2 = (x2b-x1b)*(y2b-y1b)

        union = a1 + a2 - inter
        return inter/union if union>0 else 0
        
    def group_boxes(self,items, iou_thresh=0.1):
        groups = []

        for item in items:
            placed = False

            for group in groups:
                for g in group:
                    if self.compute_iou(item[2], g[2]) > iou_thresh:
                        group.append(item)
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                groups.append([item])

        return groups


    def merge_groups(self,groups, iou_thresh=0.1):
        merged = True

        while merged:
            merged = False
            new_groups = []

            while groups:
                g1 = groups.pop(0)

                i = 0
                while i < len(groups):
                    g2 = groups[i]

                    if any(self.compute_iou(a[2], b[2]) > iou_thresh for a in g1 for b in g2):
                        g1.extend(g2)
                        groups.pop(i)
                        merged = True
                    else:
                        i += 1

                new_groups.append(g1)

            groups = new_groups

        return groups
    
    @staticmethod
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])


    def select_best(self, groups):
        final = []

        for group in groups:

            # ===== הפרדה =====
            orig_items = [g for g in group if g[5] == "orig"]
            crop_items = [g for g in group if g[5] == "crop"]

            # =========================
            # 🔵 מקרה 1: יש ORIG
            # =========================
            if orig_items:
                best_orig = max(orig_items, key=lambda x: x[1])

                # האם crop מנצח משמעותית?
                if crop_items:
                    best_crop = max(crop_items, key=lambda x: x[1])

                    if best_crop[1] > best_orig[1] + 0.5:
                        best = best_crop
                    else:
                        best = best_orig
                else:
                    best = best_orig

            # =========================
            # 🟢 מקרה 2: אין ORIG
            # =========================
            else:
                # הבוקס הכי גדול
                largest = max(group, key=lambda x: self.box_area(x[2]))
                largest_area = self.box_area(largest[2])
                largest_conf = largest[1]

                best = largest

                for item in group:
                    area = self.box_area(item[2])
                    conf = item[1]

                    # אם קטן יותר אבל הרבה יותר בטוח
                    if area < largest_area:
                        if conf > largest_conf + 0.25:
                            best = item

            final.append(best)

        return final

    @staticmethod
    def extract_time_bulletproof(ocr_results):
        # 1. Look for AM/PM
        is_pm = False
        is_am = False
        for item in ocr_results:
            text = str(item[0]).upper().strip() 
            if "PM" in text: is_pm = True
            elif "AM" in text: is_am = True

        # Get only the numeric results
        numbers = [item for item in ocr_results if str(item[0]).isdigit()]
        if not numbers:
            return "No time detected"

        # 2. Find the tallest height AND the main bottom baseline
        max_height = max([item[2][3] - item[2][1] for item in numbers])
        
        tall_digits = [item for item in numbers if (item[2][3] - item[2][1]) > max_height * 0.8]
        main_bottom_baseline = max([item[2][3] for item in tall_digits])

        # 3. Filter by Height and Baseline
        candidate_digits = []
        for item in numbers:
            height = item[2][3] - item[2][1]
            y_max = item[2][3] 
            
            is_tall = height > max_height * 0.7
            on_baseline = abs(y_max - main_bottom_baseline) < (max_height * 0.2)

            if is_tall or on_baseline:
                candidate_digits.append(item)

        candidate_digits.sort(key=lambda x: x[3])

        if not candidate_digits:
            return "No time detected"

        # 3.5 THE "CLEAR SKIES" FILTER (Structural Stacking Check)
        main_digits = []
        for candidate in candidate_digits:
            cand_x_min = candidate[2][0]
            cand_x_max = candidate[2][2]
            cand_y_min = candidate[2][1] # Top edge
            
            has_roof = False
            # Look through ALL numbers to see if anything sits above this candidate
            for other in numbers:
                if other == candidate: continue
                    
                other_x_min = other[2][0]
                other_x_max = other[2][2]
                other_y_max = other[2][3] # Bottom edge
                
                # Do they share horizontal space?
                horizontal_overlap = (cand_x_max > other_x_min) and (cand_x_min < other_x_max)
                
                # Is the 'other' number physically above our candidate?
                # (We add a small 15px buffer to account for slanted fonts/bounding boxes)
                is_above = other_y_max <= (cand_y_min + 15) 
                
                if horizontal_overlap and is_above:
                    has_roof = True
                    break
                    
            if not has_roof:
                main_digits.append(candidate)
            else:
                # We hit a stacked info block! We drop it and everything after it.
                break

        # 4. Extract the string
        time_str = "".join([item[0] for item in main_digits])

        # 5. Dynamic Parsing
        if len(time_str) in [6, 5]: 
            hour_length = 2 if len(time_str) == 6 else 1
            hour = int(time_str[:hour_length])
            remainder = time_str[hour_length:] 
        elif len(time_str) in [4, 3]: 
            hour_length = 2 if len(time_str) == 4 else 1
            hour = int(time_str[:hour_length])
            remainder = time_str[hour_length:] 
        else:
            return time_str 

        # 6. Apply 24-Hour Conversion (Only for PM)
        if is_pm and hour < 12:
            hour += 12

        # 7. Reconstruct
        hour_str = f"{hour:02d}"
        chunks = [hour_str]
        for i in range(0, len(remainder), 2):
            chunks.append(remainder[i:i+2])

        return ":".join(chunks)

    def extract_time(self, image_path):

        result = self.model(image_path, verbose=False)[0]

        img_original = result.orig_img.copy()

        original_detections = self.original_detections(img_original)
        sliding_detections = self.sliding_collect(img_original)

        all_detections = original_detections + sliding_detections
        
        groups = self.group_boxes(all_detections)
        groups = self.merge_groups(groups)

        filtered = self.select_best(groups)

        return self.extract_time_bulletproof(filtered)