import java.util.ArrayList;

public class Couche {
    ArrayList<Neurone> neurones = new ArrayList<>();
    private int i=0;


    public Couche()
    {
        for(int i=0; i<5; i++)
        neurones.add(new Neurone());
    }


    
    public void Initialisation()
    {   
        for(i=0; i<5; i++)
        {
            neurones.get(i).setPoids(Math.random());
        }
    }

    public void Calcul()
    {
        
    }
}