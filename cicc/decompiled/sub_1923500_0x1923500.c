// Function: sub_1923500
// Address: 0x1923500
//
void __fastcall sub_1923500(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // r14
  __m128i *v6; // rax
  __m128i *v7; // rdx
  __int8 *v8; // rsi
  __int8 *v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rax

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
      v6 = (__m128i *)sub_2207800(24 * v3, &unk_435FF63);
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
      v12 = (__int64 *)v7;
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
      v12 = (__int64 *)((char *)&v7[-1] + v5 - 8);
    }
    v13 = v12[1];
    v14 = v12[2];
    a1[2] = (__int64)v7;
    v15 = *v12;
    a1[1] = v3;
    a2->m128i_i64[1] = v13;
    a2->m128i_i64[0] = v15;
    a2[1].m128i_i64[0] = v14;
  }
}
