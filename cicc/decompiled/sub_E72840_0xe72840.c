// Function: sub_E72840
// Address: 0xe72840
//
__int64 __fastcall sub_E72840(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v7; // r15

  v3 = a2 - a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 5);
  v5 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v4 >> 1;
        if ( !(unsigned __int8)sub_E72550(v5 + 32 * ((v4 >> 1) + (v4 & 0xFFFFFFFFFFFFFFFELL)), a3) )
          break;
        v5 += 32 * ((v4 >> 1) + (v4 & 0xFFFFFFFFFFFFFFFELL)) + 96;
        v4 = v4 - v7 - 1;
        if ( v4 <= 0 )
          return v5;
      }
      v4 >>= 1;
    }
    while ( v7 > 0 );
  }
  return v5;
}
