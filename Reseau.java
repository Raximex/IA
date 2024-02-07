import java.util.ArrayList;
import java.util.Vector;

class Reseau{
    ArrayList <Couche> reseau = new ArrayList<>();
    private double coeffApprentissage;
    Vector vectorx = new Vector<>();
    float[][] W_1;
    float b;
    public Reseau()
    {
        for(int i=0; i<4; i++)
        reseau.add(new Couche());
        reseau.get(0).Initialisation();
    }


    public float Calcul()
    {
        return 
    }

    public float sigmoid(Vector vectorx)
    {
        return 1/(1+Math.exp(-vectorx))
    }

    // Vecteur X matrice de poids W_1 b: biais
    public float hidden(Vector vectorx, float[][] W_1,float b)
    {
        return sigmoid(Math.)
    }
}