// Function: sub_1873890
// Address: 0x1873890
//
__int64 __fastcall sub_1873890(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // r8
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rax

  v3 = a2 - a1;
  v5 = a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * (v3 >> 4);
  if ( v3 > 0 )
  {
    v7 = *(_QWORD *)(a3 + 48);
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = v5 + 80 * (v6 >> 1);
        if ( *(_QWORD *)(v9 + 48) <= v7 )
          break;
        v5 = v9 + 80;
        v6 = v6 - v8 - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v6 >>= 1;
    }
    while ( v8 > 0 );
  }
  return v5;
}
