// Function: sub_1D0C480
// Address: 0x1d0c480
//
__int64 __fastcall sub_1D0C480(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v6; // rdx

  v3 = a1;
  for ( i = (a2 - a1) >> 3; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = v3 + 8 * (i >> 1);
      if ( *(_DWORD *)(*(_QWORD *)a3 + 40LL) < *(_DWORD *)(*(_QWORD *)v6 + 40LL) )
        break;
      v3 = v6 + 8;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
