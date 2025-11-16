// Function: sub_25F5CE0
// Address: 0x25f5ce0
//
__int64 __fastcall sub_25F5CE0(int *src, int *a2, char *a3, char *a4, _DWORD *a5)
{
  int v7; // edx
  signed __int64 v8; // rbx

  if ( src == a2 )
  {
LABEL_7:
    v8 = a4 - a3;
    if ( a4 != a3 )
      return (__int64)memmove(a5, a3, a4 - a3) + v8;
  }
  else
  {
    while ( a4 != a3 )
    {
      v7 = *src;
      if ( *(_DWORD *)a3 < (unsigned int)*src )
      {
        *a5 = *(_DWORD *)a3;
        a3 += 4;
        ++a5;
        if ( src == a2 )
          goto LABEL_7;
      }
      else
      {
        ++src;
        *a5++ = v7;
        if ( src == a2 )
          goto LABEL_7;
      }
    }
    a5 = (char *)memmove(a5, src, (char *)a2 - (char *)src) + (char *)a2 - (char *)src;
    v8 = 0;
  }
  return (__int64)a5 + v8;
}
