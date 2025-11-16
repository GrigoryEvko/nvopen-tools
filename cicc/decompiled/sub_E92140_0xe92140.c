// Function: sub_E92140
// Address: 0xe92140
//
_DWORD *__fastcall sub_E92140(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v3; // r8
  __int64 i; // rsi
  _DWORD *v6; // rdx

  v3 = a1;
  for ( i = (a2 - (__int64)a1) >> 3; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = &v3[2 * (i >> 1)];
      if ( *a3 <= *v6 )
        break;
      v3 = v6 + 2;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
