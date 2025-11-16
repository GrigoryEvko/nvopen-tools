// Function: sub_2FCEAD0
// Address: 0x2fcead0
//
__int64 __fastcall sub_2FCEAD0(char *src, char *a2, char *a3, _BYTE *a4, _QWORD *a5)
{
  __int64 v7; // rdx
  __int64 v8; // rbx

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
      v7 = *(_QWORD *)src;
      if ( *(float *)(*(_QWORD *)a3 + 116LL) > *(float *)(*(_QWORD *)src + 116LL) )
      {
        *a5 = *(_QWORD *)a3;
        a3 += 8;
        ++a5;
        if ( src == a2 )
          goto LABEL_7;
      }
      else
      {
        src += 8;
        *a5++ = v7;
        if ( src == a2 )
          goto LABEL_7;
      }
    }
    a5 = (char *)memmove(a5, src, a2 - src) + a2 - src;
    v8 = 0;
  }
  return (__int64)a5 + v8;
}
