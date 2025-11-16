// Function: sub_311E130
// Address: 0x311e130
//
__int64 __fastcall sub_311E130(__int64 a1, __int64 a2, unsigned __int64 **a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v8; // r15
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v5 = a1;
  v6 = v4 >> 3;
  v10[0] = a4;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        if ( (unsigned __int8)sub_311D9B0(v10, a3, (unsigned __int64 **)(v5 + 8 * (v6 >> 1))) )
          break;
        v5 += 8 * (v6 >> 1) + 8;
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
