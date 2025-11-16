// Function: sub_E182C0
// Address: 0xe182c0
//
char *__fastcall sub_E182C0(char *a1, char *a2)
{
  char *result; // rax
  int v3; // ecx
  char *v4; // rdx
  int v5; // ecx

  result = a1;
  if ( a1 != a2 )
  {
    if ( *a1 == 95 )
    {
      if ( a2 != a1 + 1 )
      {
        v3 = a1[1];
        if ( (unsigned int)(v3 - 48) > 9 )
        {
          if ( (_BYTE)v3 == 95 )
          {
            v4 = a1 + 2;
            if ( a2 != a1 + 2 )
            {
              while ( 1 )
              {
                v5 = *v4;
                if ( (unsigned int)(v5 - 48) > 9 )
                  break;
                if ( a2 == ++v4 )
                  return a1;
              }
              if ( a2 == v4 )
                return a1;
              result = v4 + 1;
              if ( (_BYTE)v5 != 95 )
                return a1;
            }
          }
        }
        else
        {
          return a1 + 2;
        }
      }
    }
    else if ( (unsigned int)(*a1 - 48) <= 9 )
    {
      result = a1 + 1;
      if ( a2 != a1 + 1 )
      {
        while ( (unsigned int)(*result - 48) <= 9 )
        {
          if ( a2 == ++result )
            return result;
        }
        if ( result != a2 )
          return a1;
      }
    }
  }
  return result;
}
