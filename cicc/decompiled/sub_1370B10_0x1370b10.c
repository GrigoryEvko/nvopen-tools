// Function: sub_1370B10
// Address: 0x1370b10
//
void __fastcall sub_1370B10(unsigned int *src, unsigned int *a2)
{
  unsigned int *i; // rbx
  unsigned int v3; // r12d
  unsigned int *v4; // rdx
  unsigned int *j; // rax

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; *src = v3 )
    {
      while ( 1 )
      {
        v3 = *i;
        v4 = i;
        if ( *i < *src )
          break;
        for ( j = i - 1; v3 < *j; --j )
        {
          j[1] = *j;
          v4 = j;
        }
        ++i;
        *v4 = v3;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      ++i;
    }
  }
}
