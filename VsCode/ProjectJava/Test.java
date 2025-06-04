import java.util.Arrays;


public class Test 
{
    public static int[] plusOne(int[] digits)
    {
        String str = "";
        for (int i = 0; i < digits.length; i++) {
            str += digits[i];
        }
        Arrays.binarySearch(null, str);
        //str转int
        int intStr = Integer.parseInt(str);
        intStr += 1;
        String str2 = String.valueOf(intStr);
        int numberArr[] = new int[str2.length()];
        for (int i = 0; i < str2.length(); i++) {
            int c = str2.charAt(i) - '0';
            numberArr[i] = c;
        }
        return numberArr;
    }
     public static void main(String[] args) 
     {
         int s[] = { 1, 2, 3 };
         System.out.println("运行结果："+Arrays.toString(plusOne(s)));
     }
    
}
    
