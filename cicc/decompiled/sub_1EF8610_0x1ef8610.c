// Function: sub_1EF8610
// Address: 0x1ef8610
//
void __fastcall sub_1EF8610(__m128i *a1, const __m128i *a2, __m128i *a3)
{
  __int64 v4; // rsi
  const __m128i *v7; // r15
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = (char *)a2 - (char *)a1;
  v7 = (__m128i *)((char *)a3 + v4);
  if ( v4 <= 240 )
  {
    sub_1EF80B0((__int64)a1, (__int64)a2);
  }
  else
  {
    v8 = (__int64)a1;
    do
    {
      v9 = v8;
      v8 += 280;
      sub_1EF80B0(v9, v8);
    }
    while ( (__int64)a2->m128i_i64 - v8 > 240 );
    sub_1EF80B0(v8, (__int64)a2);
    if ( v4 > 280 )
    {
      v10 = 7;
      do
      {
        sub_1EF8540(a1, a2, a3, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_1EF8540(a3, v7, a1, v11);
      }
      while ( (__int64)(0xCCCCCCCCCCCCCCCDLL * (v4 >> 3)) > v10 );
    }
  }
}
