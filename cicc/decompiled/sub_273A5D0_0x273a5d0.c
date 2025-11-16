// Function: sub_273A5D0
// Address: 0x273a5d0
//
char *__fastcall sub_273A5D0(const __m128i *src, __int64 a2, __m128i *a3, __int64 a4)
{
  __int64 v5; // rax
  const __m128i *v6; // r15
  __int64 v7; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  const __m128i *v11; // rdi
  __int64 v12; // rsi

  v5 = (a2 - (__int64)src) >> 6;
  v6 = src;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v9 = a4 << 7;
    v10 = -4 * a4;
    do
    {
      v11 = v6;
      v6 = (const __m128i *)((char *)v6 + v9);
      a3 = (__m128i *)sub_273A410(v11, &v6[v10], (__int64)v6[v10].m128i_i64, (__int64)v6, a3);
      v5 = (a2 - (__int64)v6) >> 6;
    }
    while ( v5 >= v7 );
  }
  v12 = a4;
  if ( v5 <= a4 )
    v12 = v5;
  return sub_273A410(v6, &v6[4 * v12], (__int64)v6[4 * v12].m128i_i64, a2, a3);
}
