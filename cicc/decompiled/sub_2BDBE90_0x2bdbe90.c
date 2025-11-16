// Function: sub_2BDBE90
// Address: 0x2bdbe90
//
void __fastcall sub_2BDBE90(char *src, char *a2)
{
  char *i; // r13
  char v4; // bl
  char *v5; // rdi
  char v6; // dl
  char *j; // rax

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; *src = v4 )
    {
      while ( 1 )
      {
        v4 = *i;
        v5 = i++;
        if ( v4 < *src )
          break;
        v6 = *(i - 2);
        for ( j = i - 2; v4 < v6; --j )
        {
          j[1] = v6;
          v5 = j;
          v6 = *(j - 1);
        }
        *v5 = v4;
        if ( a2 == i )
          return;
      }
      if ( v5 != src )
        memmove(src + 1, src, v5 - src);
    }
  }
}
