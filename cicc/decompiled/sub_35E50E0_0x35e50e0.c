// Function: sub_35E50E0
// Address: 0x35e50e0
//
_DWORD *__fastcall sub_35E50E0(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v3; // r8
  __int64 i; // rsi
  _DWORD *v5; // rax

  v3 = a1;
  for ( i = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v5 = &v3[2 * (i >> 1) + 2 * (i & 0xFFFFFFFFFFFFFFFELL)];
      if ( *a3 < *v5 )
        break;
      v3 = v5 + 6;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
