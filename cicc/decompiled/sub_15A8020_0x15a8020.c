// Function: sub_15A8020
// Address: 0x15a8020
//
__int64 __fastcall sub_15A8020(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v4; // rsi
  __int64 i; // rax
  unsigned __int64 *v6; // rcx

  v4 = (_QWORD *)(a1 + 16);
  for ( i = (*(_DWORD *)(a1 + 12) >> 1) & 0x7FFFFFFF; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = &v4[i >> 1];
      if ( *v6 > a2 )
        break;
      v4 = v6 + 1;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return ((__int64)v4 - a1 - 24) >> 3;
    }
  }
  return ((__int64)v4 - a1 - 24) >> 3;
}
