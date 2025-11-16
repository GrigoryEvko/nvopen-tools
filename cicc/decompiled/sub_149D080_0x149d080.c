// Function: sub_149D080
// Address: 0x149d080
//
__int64 __fastcall sub_149D080(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        unsigned __int8 (__fastcall *a4)(__int64, _QWORD, _QWORD))
{
  __int64 v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v8; // r12

  v4 = a2 - a1;
  v5 = a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * (v4 >> 3);
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        if ( !a4(v5 + 40 * (v6 >> 1), *a3, a3[1]) )
          break;
        v5 += 40 * (v6 >> 1) + 40;
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
