// Function: sub_2BDBC30
// Address: 0x2bdbc30
//
char *__fastcall sub_2BDBC30(char *a1, char *a2)
{
  char v2; // cl
  char *v3; // rax
  char *v5; // rdx

  if ( a2 != a1 )
  {
    while ( 1 )
    {
      v3 = a1++;
      if ( a2 == a1 )
        break;
      v2 = *(a1 - 1);
      if ( v2 == v3[1] )
      {
        if ( a2 == v3 )
          return a2;
        v5 = v3 + 2;
        if ( a2 != v3 + 2 )
        {
          while ( 1 )
          {
            if ( v2 != *v5 )
              *++v3 = *v5;
            if ( ++v5 == a2 )
              break;
            v2 = *v3;
          }
          return v3 + 1;
        }
        return a1;
      }
    }
  }
  return a2;
}
