// Function: sub_299EBE0
// Address: 0x299ebe0
//
__int64 __fastcall sub_299EBE0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 (__fastcall *a4)(__int64, __int64))
{
  __int64 v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v8; // r15

  v4 = a2 - a1;
  v5 = a1;
  v6 = 0x6DB6DB6DB6DB6DB7LL * (v4 >> 3);
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        if ( a4(a3, v5 + 56 * (v6 >> 1)) )
          break;
        v5 += 56 * (v6 >> 1) + 56;
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
