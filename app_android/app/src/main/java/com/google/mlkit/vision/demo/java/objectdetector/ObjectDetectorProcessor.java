/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.objectdetector;

import android.content.Context;
import android.graphics.Rect;
import android.os.Build;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.ObjectDetectorOptionsBase;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** A processor to run object detector. */
public class ObjectDetectorProcessor extends VisionProcessorBase<List<DetectedObject>> {

    private static final String TAG = "ObjectDetectorProcessor";

    private static final float MIN_THRESHOLD = 0.5f;

    private final ObjectDetector detector;
    private TextToSpeech tts;

    private ArrayList<Integer> ids;
    private Utils util;

    private Distance m_distance;

    public ObjectDetectorProcessor(Context context, ObjectDetectorOptionsBase options) {
        super(context);
        detector = ObjectDetection.getClient(options);
        m_distance = new Distance();

        tts = new TextToSpeech(context, new TextToSpeech.OnInitListener() {
            @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
            @Override
            public void onInit(int i) {
                tts.setLanguage(Locale.forLanguageTag("vi-VN"));
            }
        });
        ids = new ArrayList<>();
        util = new Utils();
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<DetectedObject>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(
            @NonNull List<DetectedObject> results, @NonNull GraphicOverlay graphicOverlay) {
        for (DetectedObject object : results) {

            int start_x = object.getBoundingBox().left;
            int start_y = object.getBoundingBox().top;
            int end_x   = object.getBoundingBox().right;
            int end_y   = object.getBoundingBox().bottom;

            String lb = "";
            float score = 0;
            double distance = 0;

            for (DetectedObject.Label label : object.getLabels()) {
                lb = label.getText();
                score = label.getConfidence() * 100;

                distance = m_distance.distanceCalculator2(start_x, end_x, lb);
                // distance = m_distance.distanceCalculator(start_x, start_y, end_x, end_y, end_x - start_x, end_y - start_y, 1920, 1080, lb);
            }

            if (score >= (MIN_THRESHOLD * 100)) {

                // distance = Distance.distanceCalculator(start_x, start_y, end_x, end_y, end_x - start_x, end_y - start_y, 1920, 1080);


                graphicOverlay.add(new ObjectGraphic(graphicOverlay, object, distance));

                int tid = object.getTrackingId();
                try {
                    if (!ids.contains(tid)) {
                        lb = util.labelProcessing(lb);
                        Log.d("TTS", lb + ", " + object.getTrackingId().toString());

                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                            tts.speak("Phia trước có " + lb + " ở khoảng cách " + String.format("%.2f", distance) + " mét", TextToSpeech.QUEUE_FLUSH, null, null);
                        } else {
                            tts.speak("Phia trước có " + lb + " ở khoảng cách " + String.format("%.2f", distance) + " mét", TextToSpeech.QUEUE_FLUSH, null);
                        }
                        ids.add(tid);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Object detection failed!", e);
    }
}
