package com.google.mlkit.vision.demo.java.objectdetector;

import java.lang.Math;
import java.util.Map;
import java.util.HashMap;

public class Distance {

    private Map<String, Double> known_distance;
    private Map<String, Double> known_width;

    public Distance() {}

    public double distanceCalculator(int start_x, int start_y, int end_x, int end_y, int box_width, int box_height, int img_width, int img_height, String de_class) {

        initiate_info();

        double x_3 = start_x;
        double y_3 = end_y - (box_height / 7);

        double x_1 = img_width / 2;
        double y_1 = 0.9 * img_height;

        double x_2 = end_x;
        double y_2 = end_y - (box_height / 7);

        double angle_x1_x2 = Math.toDegrees(Math.atan2(x_1 - x_2, y_1 - y_2));
        double angle_x1_x3 = Math.toDegrees(Math.atan2(x_1 - x_3, y_1 - y_3));

        double angle_right = 90 + angle_x1_x2;
        double angle_left  = 90 - angle_x1_x3;

        double total_angle = angle_right - angle_left;

        double de_kw = 0;
        if (known_width.containsKey(de_class)) {
            de_kw = known_width.get(de_class);
        }

        double distance = (de_kw * (1 / total_angle) * 57) / 1000;

        return distance;
    }
    public double focal_length_finder(double measured_distance, double real_width, double width_in_rf) {
        double focal_length = (width_in_rf * measured_distance) / real_width;
        return focal_length;
    }

    public double distance_finder(double focal_length, double real_object_width, double width_in_frame) {
        double distance = (real_object_width * focal_length) / width_in_frame;
        return distance;
    }

    private void initiate_info() {
        known_distance = new HashMap<String, Double>();
        known_width = new HashMap<String, Double>();

        known_distance.put("Person", 22.83464567);
        known_width.put("Person", 17.71653543);

        known_distance.put("cup", 11.81102362);
        known_width.put("cup", 3.543307087);

        known_distance.put("bottle", 15.7480315);
        known_width.put("bottle", 2.559055118);

        known_distance.put("Mouse", 19.68503937);
        known_width.put("Mouse", 3.937007874);


    }

    public double distanceCalculator2(int start_x, int end_x, String de_class) {

        initiate_info();
        if (known_distance.containsKey(de_class)) {
            double focal_length = focal_length_finder(known_distance.get(de_class), known_width.get(de_class), 13);
            double distance = distance_finder(focal_length, known_width.get(de_class), end_x - start_x);

            return distance;
        }
        return 0;
    }


}
