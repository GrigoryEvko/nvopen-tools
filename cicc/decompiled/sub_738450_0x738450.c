// Function: sub_738450
// Address: 0x738450
//
void __fastcall sub_738450(const __m128i **a1, const __m128i *a2)
{
  const __m128i *v2; // r13
  const __m128i *v3; // r15
  __m128i *v5; // rax
  __m128i *v6; // r14
  const __m128i *v7; // rdx
  __m128i *v8; // rsi
  __int64 v9; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  if ( (__int64)v2 < (__int64)a2 )
  {
    v3 = *a1;
    v9 = (__int64)a1[2];
    v5 = (__m128i *)sub_823970(24LL * (_QWORD)a2);
    v6 = v5;
    if ( v9 > 0 )
    {
      v7 = v3;
      v8 = (__m128i *)((char *)v5 + 24 * v9);
      do
      {
        if ( v5 )
        {
          *v5 = _mm_loadu_si128(v7);
          v5[1].m128i_i64[0] = v7[1].m128i_i64[0];
        }
        v5 = (__m128i *)((char *)v5 + 24);
        v7 = (const __m128i *)((char *)v7 + 24);
      }
      while ( v8 != v5 );
    }
    sub_823A00(v3, 24LL * (_QWORD)v2);
    *a1 = v6;
    a1[1] = a2;
  }
}
