// Function: sub_2912200
// Address: 0x2912200
//
void __fastcall sub_2912200(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // r14
  __m128i *v6; // rax
  __m128i *v7; // rdx
  __int8 *v8; // rsi
  __int8 *v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rcx
  const __m128i *v12; // rax
  __m128i v13; // xmm2
  __int64 v14; // rax

  v3 = 0x555555555555555LL;
  if ( a3 <= 0x555555555555555LL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v5 = 24 * v3;
      v6 = (__m128i *)sub_2207800(24 * v3);
      v7 = v6;
      if ( v6 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v8 = &v6->m128i_i8[v5];
    *v6 = _mm_loadu_si128(a2);
    v6[1].m128i_i64[0] = a2[1].m128i_i64[0];
    v9 = &v6[1].m128i_i8[8];
    if ( v8 == (__int8 *)&v7[1].m128i_u64[1] )
    {
      v12 = v7;
    }
    else
    {
      do
      {
        v10 = _mm_loadu_si128((const __m128i *)(v9 - 24));
        v11 = *((_QWORD *)v9 - 1);
        v9 += 24;
        *(__m128i *)(v9 - 24) = v10;
        *((_QWORD *)v9 - 1) = v11;
      }
      while ( v8 != v9 );
      v12 = (__m128i *)((char *)v7 + v5 - 24);
    }
    v13 = _mm_loadu_si128(v12);
    v14 = v12[1].m128i_i64[0];
    a1[2] = (__int64)v7;
    a1[1] = v3;
    a2[1].m128i_i64[0] = v14;
    *a2 = v13;
  }
}
