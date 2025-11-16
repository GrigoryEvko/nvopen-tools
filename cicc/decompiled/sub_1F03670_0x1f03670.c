// Function: sub_1F03670
// Address: 0x1f03670
//
void __fastcall sub_1F03670(unsigned int *src, unsigned int *a2)
{
  unsigned int *i; // rbx
  unsigned int v3; // r12d
  unsigned int *v4; // rcx
  unsigned int v5; // edx
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
        v5 = *(i - 1);
        for ( j = i - 1; v3 < v5; --j )
        {
          j[1] = v5;
          v4 = j;
          v5 = *(j - 1);
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
