package com.google.mlkit.vision.demo.java.objectdetector;

import java.util.HashMap;
import java.util.Map;

public class Utils {

    private Map<String, String> m_trans;

    public Utils() {
        m_trans = new HashMap<String, String>();

        m_trans.put("Person", "Người");
        m_trans.put("Electronic device", "Thiết bị điện tử");
        m_trans.put("Mouse", "Chuột");
        m_trans.put("Food", "Đồ ăn");
        m_trans.put("Bottle", "Chai nước");
        m_trans.put("Mechanical fan", "Quạt máy");
        m_trans.put("Computer keyboard", "Bàn phím");
        m_trans.put("Clothing", "Đồ");
        m_trans.put("Wardrobe", "Tủ đồ");
        m_trans.put("Bed", "Giường");
        m_trans.put("Book", "Sách");
        m_trans.put("Hair Dryer", "Máy sấy");
        m_trans.put("Television", "Màn hình");

        m_trans.put("Home appliance", "Dụng cụ gia đình");
        m_trans.put("Tableware", "Đồ dùng để bàn");

        m_trans.put("cup", "Ly");
        m_trans.put("Whisk", "Đồ đánh trứng");

        m_trans.put("Container", "Hộp");
        m_trans.put("Toy",  "Đồ chơi");

        m_trans.put("Box",  "Hộp");
        m_trans.put("Pillow",  "Gối");

        m_trans.put("Luggage & bags ",  " Ba lô");
        m_trans.put("Furniture",  "Đồ nội thất");
        m_trans.put("Tie",  "Cà vạt");
    }

    public String labelProcessing(String label) {

        if (m_trans.containsKey(label)) {
            return m_trans.get(label);
        }

        return label;
    }
}
