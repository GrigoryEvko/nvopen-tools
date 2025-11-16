// Function: sub_153CEB0
// Address: 0x153ceb0
//
__int64 __fastcall sub_153CEB0(__int64 a1, __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v8; // r14
  _QWORD v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v5 = a1;
  v6 = v4 >> 4;
  v10[0] = a4;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        if ( sub_153CA80((__int64)v10, a3, (__int64 **)(v5 + 16 * (v6 >> 1))) )
          break;
        v5 += 16 * (v6 >> 1) + 16;
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
