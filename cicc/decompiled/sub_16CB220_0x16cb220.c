// Function: sub_16CB220
// Address: 0x16cb220
//
__int64 __fastcall sub_16CB220(__m128i *a1)
{
  __int32 v1; // eax
  __int64 result; // rax
  __m128i v3; // xmm5
  __m128i v4; // xmm6
  __m128i v5; // xmm7
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __int32 v8; // ecx
  __m128i v9; // [rsp+0h] [rbp-70h] BYREF
  __m128i v10; // [rsp+10h] [rbp-60h] BYREF
  __m128i v11; // [rsp+20h] [rbp-50h] BYREF
  __m128i v12; // [rsp+30h] [rbp-40h] BYREF
  __m128i v13; // [rsp+40h] [rbp-30h] BYREF
  __int64 v14; // [rsp+50h] [rbp-20h]
  __int32 v15; // [rsp+58h] [rbp-18h]

  v14 = a1[5].m128i_i64[0];
  v1 = a1[5].m128i_i32[2];
  v9 = _mm_loadu_si128(a1);
  v10 = _mm_loadu_si128(a1 + 1);
  v15 = v1;
  v11 = _mm_loadu_si128(a1 + 2);
  v12 = _mm_loadu_si128(a1 + 3);
  v13 = _mm_loadu_si128(a1 + 4);
  result = sub_16CB1E0((__int64)a1);
  v3 = _mm_loadu_si128(&v9);
  v4 = _mm_loadu_si128(&v10);
  v5 = _mm_loadu_si128(&v11);
  v6 = _mm_loadu_si128(&v12);
  v7 = _mm_loadu_si128(&v13);
  a1[5].m128i_i64[0] = v14;
  v8 = v15;
  *a1 = v3;
  a1[1] = v4;
  a1[5].m128i_i32[2] = v8;
  a1[2] = v5;
  a1[3] = v6;
  a1[4] = v7;
  return result;
}
