// Function: sub_2BE3490
// Address: 0x2be3490
//
__int8 *__fastcall sub_2BE3490(unsigned __int64 *a1, const __m128i *a2)
{
  __m128i *v2; // rax
  __int8 *result; // rax
  unsigned __int64 v4; // r13
  __m128i *v5; // rax
  __int8 **v6; // rdx
  __int64 v7; // rdx

  v2 = (__m128i *)a1[6];
  if ( v2 == (__m128i *)(a1[8] - 24) )
  {
    v4 = a1[9];
    if ( 21 * (((__int64)(v4 - a1[5]) >> 3) - 1)
       - 0x5555555555555555LL * ((__int64)((__int64)v2->m128i_i64 - a1[7]) >> 3)
       - 0x5555555555555555LL * ((__int64)(a1[4] - a1[2]) >> 3) == 0x555555555555555LL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( a1[1] - ((__int64)(v4 - *a1) >> 3) <= 1 )
    {
      sub_2BE31D0(a1, 1u, 0);
      v4 = a1[9];
    }
    *(_QWORD *)(v4 + 8) = sub_22077B0(0x1F8u);
    v5 = (__m128i *)a1[6];
    if ( v5 )
    {
      *v5 = _mm_loadu_si128(a2);
      v5[1].m128i_i64[0] = a2[1].m128i_i64[0];
    }
    v6 = (__int8 **)(a1[9] + 8);
    a1[9] = (unsigned __int64)v6;
    result = *v6;
    v7 = (__int64)(*v6 + 504);
    a1[7] = (unsigned __int64)result;
    a1[8] = v7;
    a1[6] = (unsigned __int64)result;
  }
  else
  {
    if ( v2 )
    {
      *v2 = _mm_loadu_si128(a2);
      v2[1].m128i_i64[0] = a2[1].m128i_i64[0];
      v2 = (__m128i *)a1[6];
    }
    result = &v2[1].m128i_i8[8];
    a1[6] = (unsigned __int64)result;
  }
  return result;
}
