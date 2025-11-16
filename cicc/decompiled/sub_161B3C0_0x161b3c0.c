// Function: sub_161B3C0
// Address: 0x161b3c0
//
_QWORD *__fastcall sub_161B3C0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // r8
  __int64 i; // rsi
  _QWORD *v6; // rdx

  v3 = a1;
  for ( i = (a2 - (__int64)a1) >> 3; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = &v3[i >> 1];
      if ( *a3 <= *v6 )
        break;
      v3 = v6 + 1;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
