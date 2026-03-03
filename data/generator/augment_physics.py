import cv2
import numpy as np
import os

class PhysicsEngine:
    
    def var_01_defocus(self, img): return cv2.GaussianBlur(img, (15, 15), 7)
    
    def var_02_flash(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (w//2, h//3), 300, 255, -1)
        mask = cv2.GaussianBlur(mask, (401, 401), 150) / 255.0
        img = img.astype(np.float32)
        for i in range(3): img[:,:,i] += mask * 160
        return np.clip(img, 0, 255).astype(np.uint8)

    def var_03_lowlight(self, img): return cv2.convertScaleAbs(img, alpha=0.5, beta=-40)

    def var_04_keystone(self, img):
        h, w = img.shape[:2]
        src = np.float32([[0,0], [w,0], [0,h], [w,h]])
        dst = np.float32([[w*0.12, h*0.05], [w*0.88, 0], [0, h], [w, h*0.9]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderValue=(255,255,255))

    def var_05_motionblur(self, img):
        kernel = np.zeros((11, 11)); kernel[5, :] = 1.0/11.0
        return cv2.filter2D(img, -1, kernel)

    def var_06_warping(self, img):
        rows, cols = img.shape[:2]
        out = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset = int(8.0 * np.sin(2 * np.pi * i / 150))
                if j+offset < cols: out[i,j] = img[i,(j+offset)%cols]
                else: out[i,j] = 255
        return out

    def var_07_stain(self, img):
        overlay = img.copy()
        cv2.ellipse(overlay, (250, 450), (120, 180), 45, 0, 360, (180, 220, 240), -1)
        return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    def var_08_annotation(self, img):
        cv2.circle(img, (410, 800), 50, (0, 0, 255), 3) # Mimic doctor's ink
        return img

    def var_09_all_in(self, img):
        img = self.var_01_defocus(img)
        img = self.var_04_keystone(img)
        noise = np.random.normal(0, 30, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    def var_10_jpeg(self, img):
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 25])
        return cv2.imdecode(enc, 1)