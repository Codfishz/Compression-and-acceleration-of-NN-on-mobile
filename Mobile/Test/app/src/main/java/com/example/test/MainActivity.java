package com.example.test;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bitmap bitmap = null;
        Bitmap[] bitmaps=new Bitmap[500];
        Module module = null;
        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            // app/src/main/assets/image.jpg
            bitmap = BitmapFactory.decodeStream(getAssets().open("0000.jpg"));

            for(int i=0;i<500;i++){
                String temp=String.format("%04d",i);
                bitmaps[i]=BitmapFactory.decodeStream(getAssets().open(temp+".jpg"));
            }
            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            //module = LiteModuleLoader.load(assetFilePath(this, "qat.ptl"));
            module = LiteModuleLoader.load(assetFilePath(this, "ori.ptl"));
        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(bitmap);
//        final Button button = findViewById(R.id.button);
//
//        button.setOnClickListener(new View.OnClickListener() {
//            public void onClick(View v) {
//                // Code here executes on main thread after user presses button
//            }
//        });
        long time1 = System.currentTimeMillis();
        // preparing input tensor
        // running the model
        for(int i=0;i<100;i++){

            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmaps[i],
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
            final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
            final float[] scores = outputTensor.getDataAsFloatArray();
            float maxScore = -Float.MAX_VALUE;
            int maxScoreIdx = -1;
            for (int j = 0; j < scores.length; j++) {
                if (scores[j] > maxScore) {
                    maxScore = scores[j];
                    maxScoreIdx = j;
                }
            }
            String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
        }
        //final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats


        // searching for the index with maximum score
        //



        long time2 = System.currentTimeMillis();
        String time_stamp = String.valueOf(time2-time1);
        // showing className on UI
        //TextView textView = findViewById(R.id.text);
        TextView timeView = findViewById(R.id.textView2);
        //textView.setText(maxScoreIdx);
        timeView.setText(time_stamp);
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}