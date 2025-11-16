// Function: sub_C9F6B0
// Address: 0xc9f6b0
//
void __fastcall sub_C9F6B0(__m128i **a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __m128i *v5; // r12
  __m128i v6; // xmm0
  __m128i v7; // xmm1

  v5 = a1[1];
  if ( v5 == a1[2] )
  {
    sub_C9F3C0(a1, a1[1], a2, a3, a4);
  }
  else
  {
    if ( v5 )
    {
      v6 = _mm_loadu_si128(a2);
      v7 = _mm_loadu_si128(a2 + 1);
      v5[2].m128i_i64[0] = a2[2].m128i_i64[0];
      *v5 = v6;
      v5[1] = v7;
      v5[2].m128i_i64[1] = (__int64)&v5[3].m128i_i64[1];
      sub_C9CAB0(&v5[2].m128i_i64[1], *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
      v5[4].m128i_i64[1] = (__int64)&v5[5].m128i_i64[1];
      sub_C9CAB0(&v5[4].m128i_i64[1], *(_BYTE **)a4, *(_QWORD *)a4 + *(_QWORD *)(a4 + 8));
      v5 = a1[1];
    }
    a1[1] = (__m128i *)((char *)v5 + 104);
  }
}
