// Function: sub_3510F80
// Address: 0x3510f80
//
_DWORD *__fastcall sub_3510F80(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v3; // r8
  __int64 i; // rsi
  _DWORD *v5; // rax

  v3 = a1;
  for ( i = (a2 - (__int64)a1) >> 4; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v5 = &v3[4 * (i >> 1)];
      if ( *a3 >= *v5 )
        break;
      v3 = v5 + 4;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
