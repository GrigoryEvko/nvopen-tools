// Function: sub_28E9150
// Address: 0x28e9150
//
char *__fastcall sub_28E9150(char *src, char *a2, char *a3, char *a4, _QWORD *a5)
{
  __int64 v7; // rdx
  size_t v8; // r13
  char *v9; // r8

  if ( a2 != src )
  {
    while ( a4 != a3 )
    {
      v7 = *(_QWORD *)src;
      if ( *(_DWORD *)(*(_QWORD *)a3 + 32LL) < *(_DWORD *)(*(_QWORD *)src + 32LL) )
      {
        *a5 = *(_QWORD *)a3;
        a3 += 8;
        ++a5;
        if ( a2 == src )
          break;
      }
      else
      {
        src += 8;
        *a5++ = v7;
        if ( a2 == src )
          break;
      }
    }
  }
  v8 = a2 - src;
  if ( a2 != src )
    a5 = memmove(a5, src, v8);
  v9 = (char *)a5 + v8;
  if ( a4 != a3 )
    v9 = (char *)memmove(v9, a3, a4 - a3);
  return &v9[a4 - a3];
}
