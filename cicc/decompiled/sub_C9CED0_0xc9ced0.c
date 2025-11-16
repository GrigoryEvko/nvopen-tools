// Function: sub_C9CED0
// Address: 0xc9ced0
//
__int64 __fastcall sub_C9CED0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v4; // r8
  __m128i *v6; // rbx
  __int8 *v7; // rdi
  __int8 *v8; // r15
  __int64 i; // rdx
  __int64 v10; // rcx
  __int8 *v11; // r13
  __int8 *v12; // r14
  __m128i *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __int64 v19; // rdx
  _BYTE *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r15
  __int8 *v25; // rdi
  const __m128i *v26; // r12
  __m128i v27; // xmm7
  __m128i v28; // xmm5
  __int64 result; // rax
  const __m128i *v30; // r15
  __int8 *v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-D8h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  __int64 v36; // [rsp+30h] [rbp-B0h]
  __int64 v37; // [rsp+30h] [rbp-B0h]
  __int64 v38; // [rsp+38h] [rbp-A8h]
  __int64 v39; // [rsp+38h] [rbp-A8h]
  __int64 v40; // [rsp+38h] [rbp-A8h]
  __int64 v41; // [rsp+38h] [rbp-A8h]
  __m128i v42; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v43; // [rsp+50h] [rbp-90h] BYREF
  __int64 v44; // [rsp+60h] [rbp-80h]
  __int64 v45[2]; // [rsp+68h] [rbp-78h] BYREF
  _QWORD v46[2]; // [rsp+78h] [rbp-68h] BYREF
  __int64 v47[2]; // [rsp+88h] [rbp-58h] BYREF
  _QWORD v48[9]; // [rsp+98h] [rbp-48h] BYREF

  v4 = a1;
  v6 = (__m128i *)(a1 + 104 * a2);
  v7 = &v6[2].m128i_i8[8];
  v8 = &v6[4].m128i_i8[8];
  v34 = (a3 - 1) / 2;
  if ( a2 >= v34 )
  {
    v14 = a2;
    v12 = &v6[4].m128i_i8[8];
    v11 = &v6[2].m128i_i8[8];
  }
  else
  {
    for ( i = a2; ; i = v36 )
    {
      v10 = 2 * (i + 1);
      v6 = (__m128i *)(v4 + 208 * (i + 1));
      if ( *(double *)(v4 + 104 * (v10 - 1)) > *(double *)v6->m128i_i64 )
        v6 = (__m128i *)(v4 + 104 * --v10);
      v11 = &v6[2].m128i_i8[8];
      v36 = v10;
      v12 = &v6[4].m128i_i8[8];
      v38 = v4;
      v13 = (__m128i *)(v4 + 104 * i);
      *v13 = _mm_loadu_si128(v6);
      v13[1] = _mm_loadu_si128(v6 + 1);
      v13[2].m128i_i64[0] = v6[2].m128i_i64[0];
      sub_2240AE0(v7, &v6[2].m128i_u64[1]);
      sub_2240AE0(v8, &v6[4].m128i_u64[1]);
      v14 = v36;
      v4 = v38;
      if ( v36 >= v34 )
        break;
      v8 = &v6[4].m128i_i8[8];
      v7 = &v6[2].m128i_i8[8];
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v14 )
  {
    v41 = v4;
    v37 = 2 * v14 + 1;
    v30 = (const __m128i *)(v4 + 104 * v37);
    *v6 = _mm_loadu_si128(v30);
    v6[1] = _mm_loadu_si128(v30 + 1);
    v6[2].m128i_i64[0] = v30[2].m128i_i64[0];
    sub_2240AE0(v11, &v30[2].m128i_u64[1]);
    v31 = v12;
    v12 = &v30[4].m128i_i8[8];
    sub_2240AE0(v31, &v30[4].m128i_u64[1]);
    v11 = &v30[2].m128i_i8[8];
    v6 = (__m128i *)v30;
    v14 = v37;
    v4 = v41;
  }
  v15 = a4[2].m128i_i64[0];
  v16 = (_BYTE *)a4[2].m128i_i64[1];
  v32 = v4;
  v17 = _mm_loadu_si128(a4);
  v18 = _mm_loadu_si128(a4 + 1);
  v45[0] = (__int64)v46;
  v44 = v15;
  v19 = (__int64)&v16[a4[3].m128i_i64[0]];
  v39 = v14;
  v42 = v17;
  v43 = v18;
  sub_C9CAB0(v45, v16, v19);
  v20 = (_BYTE *)a4[4].m128i_i64[1];
  v21 = a4[5].m128i_i64[0];
  v47[0] = (__int64)v48;
  sub_C9CAB0(v47, v20, (__int64)&v20[v21]);
  v22 = v39;
  v23 = v32;
  v24 = (v39 - 1) / 2;
  if ( v39 > a2 )
  {
    while ( 1 )
    {
      v40 = v23;
      v26 = (const __m128i *)(v23 + 104 * v24);
      v6 = (__m128i *)(v23 + 104 * v22);
      if ( *(double *)v42.m128i_i64 <= *(double *)v26->m128i_i64 )
        break;
      *v6 = _mm_loadu_si128(v26);
      v6[1] = _mm_loadu_si128(v26 + 1);
      v6[2].m128i_i64[0] = v26[2].m128i_i64[0];
      sub_2240AE0(v11, &v26[2].m128i_u64[1]);
      v25 = v12;
      v12 = &v26[4].m128i_i8[8];
      sub_2240AE0(v25, &v26[4].m128i_u64[1]);
      v11 = &v26[2].m128i_i8[8];
      v23 = v40;
      v22 = v24;
      if ( a2 >= v24 )
      {
        v6 = (__m128i *)v26;
        break;
      }
      v24 = (v24 - 1) / 2;
    }
  }
  v27 = _mm_loadu_si128(&v42);
  v28 = _mm_loadu_si128(&v43);
  v6[2].m128i_i64[0] = v44;
  *v6 = v27;
  v6[1] = v28;
  sub_2240AE0(v11, v45);
  result = sub_2240AE0(v12, v47);
  if ( (_QWORD *)v47[0] != v48 )
    result = j_j___libc_free_0(v47[0], v48[0] + 1LL);
  if ( (_QWORD *)v45[0] != v46 )
    return j_j___libc_free_0(v45[0], v46[0] + 1LL);
  return result;
}
