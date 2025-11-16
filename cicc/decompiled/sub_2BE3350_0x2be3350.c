// Function: sub_2BE3350
// Address: 0x2be3350
//
unsigned __int64 __fastcall sub_2BE3350(unsigned __int64 *a1, const __m128i *a2)
{
  unsigned __int64 v2; // r13
  __m128i *v3; // rax
  unsigned __int64 *v4; // rdx
  unsigned __int64 result; // rax
  __int64 v6; // rdx

  v2 = a1[9];
  if ( 21 * (((__int64)(v2 - a1[5]) >> 3) - 1)
     - 0x5555555555555555LL * ((__int64)(a1[6] - a1[7]) >> 3)
     - 0x5555555555555555LL * ((__int64)(a1[4] - a1[2]) >> 3) == 0x555555555555555LL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( a1[1] - ((__int64)(v2 - *a1) >> 3) <= 1 )
  {
    sub_2BE31D0(a1, 1u, 0);
    v2 = a1[9];
  }
  *(_QWORD *)(v2 + 8) = sub_22077B0(0x1F8u);
  v3 = (__m128i *)a1[6];
  if ( v3 )
  {
    *v3 = _mm_loadu_si128(a2);
    v3[1].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v4 = (unsigned __int64 *)(a1[9] + 8);
  a1[9] = (unsigned __int64)v4;
  result = *v4;
  v6 = *v4 + 504;
  a1[7] = result;
  a1[8] = v6;
  a1[6] = result;
  return result;
}
