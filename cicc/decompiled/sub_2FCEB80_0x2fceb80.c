// Function: sub_2FCEB80
// Address: 0x2fceb80
//
void __fastcall sub_2FCEB80(int **src, int **a2)
{
  int **i; // rbx
  int *v4; // r12
  int **v5; // rcx
  int v6; // esi
  int *v7; // rdx
  int **v8; // rax

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; *src = v4 )
    {
      while ( 1 )
      {
        v4 = *i;
        v5 = i;
        v6 = **i;
        if ( v6 < **src )
          break;
        v7 = *(i - 1);
        v8 = i - 1;
        if ( v6 < *v7 )
        {
          do
          {
            v8[1] = v7;
            v5 = v8;
            v7 = *--v8;
          }
          while ( *v4 < *v7 );
        }
        ++i;
        *v5 = v4;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      ++i;
    }
  }
}
