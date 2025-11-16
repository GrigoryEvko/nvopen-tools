// Function: sub_3510E70
// Address: 0x3510e70
//
_QWORD *__fastcall sub_3510E70(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // r8
  __int64 i; // rsi
  _QWORD *v5; // rax

  v3 = a1;
  for ( i = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v5 = &v3[(i >> 1) + (i & 0xFFFFFFFFFFFFFFFELL)];
      if ( *a3 > *v5 )
        break;
      v3 = v5 + 3;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
