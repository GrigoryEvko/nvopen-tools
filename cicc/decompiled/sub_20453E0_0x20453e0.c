// Function: sub_20453E0
// Address: 0x20453e0
//
__int64 __fastcall sub_20453E0(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned __int32 v4; // r14d
  __int64 v5; // r12
  bool v6; // al
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __m128i *v9; // r13
  __int64 v10; // rax
  __m128i v11; // xmm5
  __int64 result; // rax
  __m128i v13; // [rsp+0h] [rbp-50h] BYREF
  __m128i v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+20h] [rbp-30h]

  v1 = a1;
  v2 = a1[2].m128i_i64[0];
  v3 = a1->m128i_i64[1];
  v4 = a1[2].m128i_u32[0];
  v13 = _mm_loadu_si128(a1);
  v15 = v2;
  v5 = v3 + 24;
  v14 = _mm_loadu_si128(a1 + 1);
  while ( 1 )
  {
    v9 = (__m128i *)v1;
    if ( v4 == v1[-1].m128i_i32[2] )
      break;
    v6 = v4 > v1[-1].m128i_i32[2];
    v1 = (const __m128i *)((char *)v1 - 40);
    if ( !v6 )
      goto LABEL_6;
LABEL_3:
    v7 = _mm_loadu_si128(v1);
    v8 = _mm_loadu_si128(v1 + 1);
    v1[4].m128i_i32[2] = v1[2].m128i_i32[0];
    *(__m128i *)((char *)v1 + 40) = v7;
    *(__m128i *)((char *)v1 + 56) = v8;
  }
  v10 = v1[-2].m128i_i64[0];
  v1 = (const __m128i *)((char *)v1 - 40);
  if ( (int)sub_16AEA10(v5, v10 + 24) < 0 )
    goto LABEL_3;
LABEL_6:
  v11 = _mm_loadu_si128(&v14);
  result = (unsigned int)v15;
  *v9 = _mm_loadu_si128(&v13);
  v9[2].m128i_i32[0] = result;
  v9[1] = v11;
  return result;
}
