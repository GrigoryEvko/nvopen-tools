// Function: sub_16D62A0
// Address: 0x16d62a0
//
__int64 __fastcall sub_16D62A0(const __m128i *a1)
{
  __m128i *v1; // rbx
  const __m128i *v2; // r12
  const __m128i *v3; // r13
  _BYTE *v4; // rsi
  __int64 v5; // rdx
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  const __m128i *v10; // rdi
  const __m128i *v11; // rdi
  __m128i v12; // xmm6
  __int64 result; // rax
  __m128i v14; // [rsp+10h] [rbp-90h] BYREF
  __m128i v15; // [rsp+20h] [rbp-80h] BYREF
  __int64 v16[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v17[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v18[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v19[8]; // [rsp+60h] [rbp-40h] BYREF

  v1 = (__m128i *)a1;
  v2 = a1 + 2;
  v3 = a1 + 4;
  v4 = (_BYTE *)a1[2].m128i_i64[0];
  v5 = a1[2].m128i_i64[1];
  v16[0] = (__int64)v17;
  v14 = _mm_loadu_si128(a1);
  v15 = _mm_loadu_si128(a1 + 1);
  sub_16D5EB0(v16, v4, (__int64)&v4[v5]);
  v6 = (_BYTE *)a1[4].m128i_i64[0];
  v7 = (__int64)&v6[a1[4].m128i_i64[1]];
  v18[0] = (__int64)v19;
  sub_16D5EB0(v18, v6, v7);
  if ( *(double *)a1[-6].m128i_i64 > *(double *)v14.m128i_i64 )
  {
    do
    {
      v8 = _mm_loadu_si128(v2 - 8);
      v9 = _mm_loadu_si128(v2 - 7);
      v10 = v2;
      v1 = (__m128i *)&v2[-8];
      v2 -= 6;
      v2[4] = v8;
      v2[5] = v9;
      sub_2240AE0(v10, v2);
      v11 = v3;
      v3 -= 6;
      sub_2240AE0(v11, v3);
    }
    while ( *(double *)v2[-8].m128i_i64 > *(double *)v14.m128i_i64 );
  }
  v12 = _mm_loadu_si128(&v15);
  *v1 = _mm_loadu_si128(&v14);
  v1[1] = v12;
  sub_2240AE0(v2, v16);
  result = sub_2240AE0(v3, v18);
  if ( (_QWORD *)v18[0] != v19 )
    result = j_j___libc_free_0(v18[0], v19[0] + 1LL);
  if ( (_QWORD *)v16[0] != v17 )
    return j_j___libc_free_0(v16[0], v17[0] + 1LL);
  return result;
}
