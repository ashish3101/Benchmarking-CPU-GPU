#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

void euclidian_distance_transform(unsigned char* img, float* dist, int w, int h) {
    
    const int N = w*h;
    
    for (int i = 0 ; i < N ; i++)
    {
        printf ("i=%d\n", i);
        int cx = i % w;
        int cy = i / w;
  
        float minv = INFINITY;
  
        if (img[i] > 0)
        {
            minv = 0.0f;
        }
        else
        {
            for (int j = 0; j < N; j++)  
            {
                if (img[j] > 0)
                {
                    int x = j % w;
                    int y = j / w;
                    float d = sqrtf( powf(float(x-cx), 2.0f) + powf(float(y-cy), 2.0f) );
                    if (d < minv) minv = d;
                }
            }
        }
        dist[i] = minv;
    }
}

int main()
{
    char line[256];
    int w,h;
    int i;
    int v;
    
    FILE* f = fopen("input.pgm", "r");
    fgets(line, sizeof(line), f);
    fgets(line, sizeof(line), f);
    fgets(line, sizeof(line), f);
    sscanf(line, "%d %d", &w, &h);
    fgets(line, sizeof(line), f);
    
    printf("%d %d\n", w, h);
    
    unsigned char* img = (unsigned char*) malloc (sizeof(unsigned char) * w * h);
    float* dist = (float*) malloc (sizeof (float) * w * h);
    
    for (i=0; i<w*h; i++)
    {
        fgets(line, sizeof(line), f);
        sscanf(line, "%d", &v);
        img[i] = (v > 0)? 255 : 0;
        //if (img[i]==255) printf("wp: %d %d\n", i%w, i/w);
    }
    
    fclose(f);
    
    printf("start\n");

    float time=0;
    euclidian_distance_transform(img, dist, w, h);

    printf("end\n");
    printf("time: %f\n", time);
    
    FILE* f2 = fopen("output.pgm", "w");
    fprintf(f2, "P2\n");
    fprintf(f2, "#\n");
    fprintf(f2, "%d %d\n", w, h);
    fprintf(f2, "255\n");
    
    float max = 0.0f;
    for (i=0; i<w*h; i++)
    {
       max = (dist[i] > max)? dist[i] : max;
    }
    printf("max: %f\n", max);
    
    for (i=0; i<w*h; i++)
    {
        fprintf(f2, "%d\n", ((int)floor((255.0f*dist[i])/max)));
    }
    
    fclose(f2);
       
    free(img);
    free(dist);
    
    return 0;
}
