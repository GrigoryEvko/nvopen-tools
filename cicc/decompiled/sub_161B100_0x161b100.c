// Function: sub_161B100
// Address: 0x161b100
//
char *__fastcall sub_161B100(_QWORD *src, _BYTE *a2, char *a3, char *a4, _QWORD *a5)
{
  __int64 v7; // rdx
  size_t v8; // r13
  char *v9; // r8

  if ( a2 != (_BYTE *)src )
  {
    while ( a4 != a3 )
    {
      v7 = *src;
      if ( *src > *(_QWORD *)a3 )
      {
        *a5 = *(_QWORD *)a3;
        a3 += 8;
        ++a5;
        if ( a2 == (_BYTE *)src )
          break;
      }
      else
      {
        ++src;
        *a5++ = v7;
        if ( a2 == (_BYTE *)src )
          break;
      }
    }
  }
  v8 = a2 - (_BYTE *)src;
  if ( a2 != (_BYTE *)src )
    a5 = memmove(a5, src, v8);
  v9 = (char *)a5 + v8;
  if ( a4 != a3 )
    v9 = (char *)memmove(v9, a3, a4 - a3);
  return &v9[a4 - a3];
}
