// Function: sub_1AA4200
// Address: 0x1aa4200
//
__int64 __fastcall sub_1AA4200(const __m128i *a1, unsigned __int8 (__fastcall *a2)(__m128i *, __int8 *))
{
  __int8 *v2; // rbx
  __int64 v3; // rax
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 v6; // rax
  __m128i v7; // xmm2
  __m128i v8; // xmm7
  __m128i v9; // xmm3
  __int64 result; // rax
  __m128i v11; // [rsp+0h] [rbp-60h] BYREF
  __m128i v12; // [rsp+10h] [rbp-50h] BYREF
  __m128i v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+30h] [rbp-30h]

  v2 = &a1[-4].m128i_i8[8];
  v3 = a1[3].m128i_i64[0];
  v11 = _mm_loadu_si128(a1);
  v14 = v3;
  v12 = _mm_loadu_si128(a1 + 1);
  v13 = _mm_loadu_si128(a1 + 2);
  while ( a2(&v11, v2) )
  {
    v4 = _mm_loadu_si128((const __m128i *)v2);
    v5 = _mm_loadu_si128((const __m128i *)v2 + 1);
    v2 -= 56;
    v6 = *((_QWORD *)v2 + 13);
    v7 = _mm_loadu_si128((const __m128i *)(v2 + 88));
    *((__m128i *)v2 + 7) = v4;
    *((_QWORD *)v2 + 20) = v6;
    *((__m128i *)v2 + 8) = v5;
    *((__m128i *)v2 + 9) = v7;
  }
  v8 = _mm_loadu_si128(&v12);
  v9 = _mm_loadu_si128(&v13);
  result = v14;
  *(__m128i *)(v2 + 56) = _mm_loadu_si128(&v11);
  *((_QWORD *)v2 + 13) = result;
  *(__m128i *)(v2 + 72) = v8;
  *(__m128i *)(v2 + 88) = v9;
  return result;
}
