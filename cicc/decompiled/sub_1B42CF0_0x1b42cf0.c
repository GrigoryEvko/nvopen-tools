// Function: sub_1B42CF0
// Address: 0x1b42cf0
//
void __fastcall sub_1B42CF0(_QWORD *src, _QWORD *a2)
{
  _QWORD *i; // rbx
  __int64 v3; // r12
  _QWORD *v4; // rcx
  __int64 v5; // rdx
  _QWORD *j; // rax

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
