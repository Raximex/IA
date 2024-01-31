import java.util.ArrayList;

class Reseau{
    ArrayList <Couche> reseau = new ArrayList<>();
    private double coeffApprentissage;
    public Reseau()
    {
        for(int i=0; i<4; i++)
        reseau.add(new Couche());
        reseau.get(0).Initialisation();
    }



}