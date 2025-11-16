// Function: sub_C9CB60
// Address: 0xc9cb60
//
__int64 __fastcall sub_C9CB60(const __m128i *a1)
{
  __m128i *v1; // rbx
  __m128i *v2; // r12
  __int8 *v3; // r13
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __m128i v6; // xmm3
  __m128i v7; // xmm4
  __int64 v8; // rdx
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __m128i v12; // xmm2
  __m128i *v13; // rdi
  __m128i v14; // xmm1
  __int8 *v15; // rdi
  __m128i v16; // xmm5
  __m128i v17; // xmm6
  __int64 result; // rax
  __m128i v19; // [rsp+10h] [rbp-A0h] BYREF
  __m128i v20; // [rsp+20h] [rbp-90h] BYREF
  __int64 v21; // [rsp+30h] [rbp-80h]
  __int64 v22[2]; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v23[2]; // [rsp+48h] [rbp-68h] BYREF
  __int64 v24[2]; // [rsp+58h] [rbp-58h] BYREF
  _QWORD v25[9]; // [rsp+68h] [rbp-48h] BYREF

  v1 = (__m128i *)a1;
  v2 = (__m128i *)&a1[2].m128i_u64[1];
  v3 = &a1[4].m128i_i8[8];
  v4 = a1[2].m128i_i64[0];
  v5 = (_BYTE *)a1[2].m128i_i64[1];
  v6 = _mm_loadu_si128(a1);
  v7 = _mm_loadu_si128(a1 + 1);
  v22[0] = (__int64)v23;
  v21 = v4;
  v8 = a1[3].m128i_i64[0];
  v19 = v6;
  v20 = v7;
  sub_C9CAB0(v22, v5, (__int64)&v5[v8]);
  v9 = (_BYTE *)a1[4].m128i_i64[1];
  v10 = (__int64)&v9[a1[5].m128i_i64[0]];
  v24[0] = (__int64)v25;
  sub_C9CAB0(v24, v9, v10);
  if ( *(double *)&a1[-7].m128i_i64[1] > *(double *)v6.m128i_i64 )
  {
    do
    {
      v11 = v2[-7].m128i_i64[0];
      v12 = _mm_loadu_si128(v2 - 8);
      v13 = v2;
      v1 = v2 - 9;
      v14 = _mm_loadu_si128(v2 - 9);
      v2 = (__m128i *)((char *)v2 - 104);
      v2[6].m128i_i64[0] = v11;
      v2[4] = v14;
      v2[5] = v12;
      sub_2240AE0(v13, v2);
      v15 = v3;
      v3 -= 104;
      sub_2240AE0(v15, v3);
    }
    while ( *(double *)v2[-9].m128i_i64 > *(double *)v19.m128i_i64 );
  }
  v16 = _mm_loadu_si128(&v19);
  v17 = _mm_loadu_si128(&v20);
  v1[2].m128i_i64[0] = v21;
  *v1 = v16;
  v1[1] = v17;
  sub_2240AE0(v2, v22);
  result = sub_2240AE0(v3, v24);
  if ( (_QWORD *)v24[0] != v25 )
    result = j_j___libc_free_0(v24[0], v25[0] + 1LL);
  if ( (_QWORD *)v22[0] != v23 )
    return j_j___libc_free_0(v22[0], v23[0] + 1LL);
  return result;
}
