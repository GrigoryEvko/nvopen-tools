// Function: sub_16D91B0
// Address: 0x16d91b0
//
void __fastcall sub_16D91B0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __m128i *v5; // r12
  __m128i v6; // xmm0
  __m128i v7; // xmm1

  v5 = *(__m128i **)(a1 + 8);
  if ( v5 == *(__m128i **)(a1 + 16) )
  {
    sub_16D8ED0((__int64 *)a1, *(char **)(a1 + 8), a2, a3, a4);
  }
  else
  {
    if ( v5 )
    {
      v6 = _mm_loadu_si128(a2);
      v7 = _mm_loadu_si128(a2 + 1);
      v5[2].m128i_i64[0] = (__int64)v5[3].m128i_i64;
      *v5 = v6;
      v5[1] = v7;
      sub_16D5EB0(v5[2].m128i_i64, *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
      v5[4].m128i_i64[0] = (__int64)v5[5].m128i_i64;
      sub_16D5EB0(v5[4].m128i_i64, *(_BYTE **)a4, *(_QWORD *)a4 + *(_QWORD *)(a4 + 8));
      v5 = *(__m128i **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v5 + 6;
  }
}
