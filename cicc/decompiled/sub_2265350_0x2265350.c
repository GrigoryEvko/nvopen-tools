// Function: sub_2265350
// Address: 0x2265350
//
__int64 __fastcall sub_2265350(unsigned __int64 *a1, __m128i *a2)
{
  __m128i *v2; // rax
  __m128i v3; // xmm1
  __int64 v4; // rcx
  __int64 v5; // rdx
  __m128i v6; // xmm0
  __int64 v7; // rdx
  __int64 v8; // rdx
  char *v9; // rax

  v2 = (__m128i *)a1[6];
  if ( v2 == (__m128i *)(a1[8] - 40) )
  {
    sub_2265100(a1, a2);
    v9 = (char *)a1[6];
  }
  else
  {
    if ( v2 )
    {
      v3 = _mm_loadu_si128(v2);
      v4 = v2[1].m128i_i64[1];
      v2[1].m128i_i64[0] = 0;
      v5 = a2[1].m128i_i64[0];
      v6 = _mm_loadu_si128(a2);
      a2[1].m128i_i64[0] = 0;
      *a2 = v3;
      v2[1].m128i_i64[0] = v5;
      v7 = a2[1].m128i_i64[1];
      *v2 = v6;
      v2[1].m128i_i64[1] = v7;
      v8 = a2[2].m128i_i64[0];
      a2[1].m128i_i64[1] = v4;
      v2[2].m128i_i64[0] = v8;
      v2 = (__m128i *)a1[6];
    }
    v9 = &v2[2].m128i_i8[8];
    a1[6] = (unsigned __int64)v9;
  }
  if ( (char *)a1[7] == v9 )
    return *(_QWORD *)(a1[9] - 8) + 440LL;
  else
    return (__int64)(v9 - 40);
}
