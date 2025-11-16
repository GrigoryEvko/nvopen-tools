// Function: sub_2D23D00
// Address: 0x2d23d00
//
__int64 __fastcall sub_2D23D00(__int64 a1, _QWORD *a2, unsigned int a3, const __m128i *a4)
{
  __m128i *v6; // rdx
  __m128i v7; // xmm0
  __int64 result; // rax
  _QWORD *v9; // r11
  __int64 v10; // rdx
  unsigned int *v11; // r8
  unsigned int *v12; // rdx
  __int64 v13; // rcx

  *(_QWORD *)(*a2 + 8LL * (a3 >> 6)) |= 1LL << a3;
  v6 = (__m128i *)(a2[9] + 24LL * a3);
  *v6 = _mm_loadu_si128(a4);
  v6[1].m128i_i64[0] = a4[1].m128i_i64[0];
  v7 = _mm_loadu_si128(a4);
  result = sub_2D22AD0(a1, a3);
  v11 = (unsigned int *)(result + 4 * v10);
  if ( (unsigned int *)result != v11 )
  {
    v12 = (unsigned int *)result;
    do
    {
      v13 = *v12++;
      *(_QWORD *)(*v9 + 8LL * ((unsigned int)v13 >> 6)) |= 1LL << v13;
      result = v9[9] + 24 * v13;
      *(_QWORD *)(result + 16) = 0;
      *(__m128i *)result = v7;
    }
    while ( v11 != v12 );
  }
  return result;
}
