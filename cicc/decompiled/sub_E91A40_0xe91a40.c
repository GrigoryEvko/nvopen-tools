// Function: sub_E91A40
// Address: 0xe91a40
//
void __fastcall sub_E91A40(unsigned __int16 *src, unsigned __int16 *a2)
{
  unsigned __int16 *i; // rbx
  unsigned __int16 v3; // r12
  unsigned __int16 *v4; // rcx
  unsigned __int16 v5; // dx
  unsigned __int16 *j; // rax

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
