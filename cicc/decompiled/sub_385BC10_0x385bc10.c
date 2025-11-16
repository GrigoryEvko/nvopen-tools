// Function: sub_385BC10
// Address: 0x385bc10
//
unsigned int *__fastcall sub_385BC10(unsigned int *a1, __int64 a2, unsigned int *a3, _QWORD *a4)
{
  unsigned int *v4; // r8
  __int64 i; // rsi
  unsigned int *v6; // r9

  v4 = a1;
  for ( i = (a2 - (__int64)a1) >> 2; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = &v4[i >> 1];
      if ( *(_QWORD *)(*a4 + 16LL * *v6) >= *(_QWORD *)(*a4 + 16LL * *a3) )
        break;
      v4 = v6 + 1;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v4;
    }
  }
  return v4;
}
