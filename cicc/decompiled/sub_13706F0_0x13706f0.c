// Function: sub_13706F0
// Address: 0x13706f0
//
_DWORD *__fastcall sub_13706F0(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v3; // r8
  __int64 i; // rsi
  _DWORD *v6; // rdx

  v3 = a1;
  for ( i = (a2 - (__int64)a1) >> 2; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = &v3[i >> 1];
      if ( *v6 >= *a3 )
        break;
      v3 = v6 + 1;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
