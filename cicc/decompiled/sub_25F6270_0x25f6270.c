// Function: sub_25F6270
// Address: 0x25f6270
//
char *__fastcall sub_25F6270(_DWORD *src, _BYTE *a2, char *a3, char *a4, _DWORD *a5)
{
  int v7; // edx
  size_t v8; // r13
  char *v9; // r8

  if ( a2 != (_BYTE *)src )
  {
    while ( a4 != a3 )
    {
      v7 = *src;
      if ( *(_DWORD *)a3 < *src )
      {
        *a5 = *(_DWORD *)a3;
        a3 += 4;
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
