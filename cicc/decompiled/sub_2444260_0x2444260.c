// Function: sub_2444260
// Address: 0x2444260
//
__int64 __fastcall sub_2444260(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v5; // rax

  v3 = a1;
  for ( i = (a2 - a1) >> 4; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v5 = v3 + 16 * (i >> 1);
      if ( *(_QWORD *)(v5 + 8) <= *(_QWORD *)(a3 + 8) )
        break;
      v3 = v5 + 16;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
