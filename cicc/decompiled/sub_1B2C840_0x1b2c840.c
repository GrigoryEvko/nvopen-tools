// Function: sub_1B2C840
// Address: 0x1b2c840
//
__int64 __fastcall sub_1B2C840(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v8; // r14
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 4);
  v6 = a1;
  v10[0] = a4;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v5 >> 1;
        if ( !sub_1B2B020(v10, v6 + 16 * ((v5 >> 1) + (v5 & 0xFFFFFFFFFFFFFFFELL)), a3) )
          break;
        v6 += 16 * ((v5 >> 1) + (v5 & 0xFFFFFFFFFFFFFFFELL)) + 48;
        v5 = v5 - v8 - 1;
        if ( v5 <= 0 )
          return v6;
      }
      v5 >>= 1;
    }
    while ( v8 > 0 );
  }
  return v6;
}
