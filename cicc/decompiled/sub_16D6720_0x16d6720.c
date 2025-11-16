// Function: sub_16D6720
// Address: 0x16d6720
//
__int64 __fastcall sub_16D6720(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v4; // r8
  __m128i *v6; // rbx
  __m128i *v7; // rdi
  __m128i *v8; // r15
  __int64 v9; // rcx
  __m128i *v10; // rdx
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __m128i v13; // xmm5
  __m128i v14; // xmm6
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r15
  __m128i *v20; // rdi
  const __m128i *v21; // r12
  __m128i v22; // xmm5
  __int64 result; // rax
  const __m128i *v24; // r15
  __m128i *v25; // rdi
  __int64 v26; // rcx
  __m128i *v27; // r14
  __m128i *v28; // r13
  __int64 v29; // [rsp+8h] [rbp-C8h]
  __int64 v31; // [rsp+20h] [rbp-B0h]
  __int64 v32; // [rsp+28h] [rbp-A8h]
  __int64 v33; // [rsp+30h] [rbp-A0h]
  __int64 v34; // [rsp+30h] [rbp-A0h]
  __int64 v35; // [rsp+38h] [rbp-98h]
  __int64 v36; // [rsp+38h] [rbp-98h]
  __int64 v37; // [rsp+38h] [rbp-98h]
  __int64 v38; // [rsp+38h] [rbp-98h]
  __m128i v39; // [rsp+40h] [rbp-90h] BYREF
  __m128i v40; // [rsp+50h] [rbp-80h] BYREF
  __int64 v41[2]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v42[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v43[2]; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v44[8]; // [rsp+90h] [rbp-40h] BYREF

  v4 = a1;
  v6 = (__m128i *)(a1 + 96 * a2);
  v7 = v6 + 2;
  v32 = a2;
  v8 = v6 + 4;
  v31 = (a3 - 1) / 2;
  if ( a2 < v31 )
  {
    while ( 1 )
    {
      v9 = 2 * (a2 + 1);
      v6 = (__m128i *)(v4 + 192 * (a2 + 1));
      if ( *(double *)(v4 + 96 * (v9 - 1)) > *(double *)v6->m128i_i64 )
        v6 = (__m128i *)(v4 + 96 * --v9);
      v28 = v6 + 2;
      v33 = v9;
      v27 = v6 + 4;
      v35 = v4;
      v10 = (__m128i *)(v4 + 96 * a2);
      *v10 = _mm_loadu_si128(v6);
      v10[1] = _mm_loadu_si128(v6 + 1);
      sub_2240AE0(v7, &v6[2]);
      sub_2240AE0(v8, &v6[4]);
      v26 = v33;
      v4 = v35;
      if ( v33 >= v31 )
        break;
      v8 = v6 + 4;
      v7 = v6 + 2;
      a2 = v33;
    }
  }
  else
  {
    v26 = a2;
    v27 = v6 + 4;
    v28 = v6 + 2;
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v26 )
  {
    v38 = v4;
    v34 = 2 * v26 + 1;
    v24 = (const __m128i *)(v4 + 96 * v34);
    *v6 = _mm_loadu_si128(v24);
    v6[1] = _mm_loadu_si128(v24 + 1);
    sub_2240AE0(v28, &v24[2]);
    v25 = v27;
    v27 = (__m128i *)&v24[4];
    sub_2240AE0(v25, &v24[4]);
    v28 = (__m128i *)&v24[2];
    v6 = (__m128i *)v24;
    v26 = v34;
    v4 = v38;
  }
  v11 = (_BYTE *)a4[2].m128i_i64[0];
  v12 = a4[2].m128i_i64[1];
  v29 = v4;
  v13 = _mm_loadu_si128(a4);
  v14 = _mm_loadu_si128(a4 + 1);
  v41[0] = (__int64)v42;
  v36 = v26;
  v39 = v13;
  v40 = v14;
  sub_16D5EB0(v41, v11, (__int64)&v11[v12]);
  v15 = (_BYTE *)a4[4].m128i_i64[0];
  v16 = a4[4].m128i_i64[1];
  v43[0] = (__int64)v44;
  sub_16D5EB0(v43, v15, (__int64)&v15[v16]);
  v17 = v36;
  v18 = v29;
  v19 = (v36 - 1) / 2;
  if ( v36 > v32 )
  {
    while ( 1 )
    {
      v37 = v18;
      v21 = (const __m128i *)(v18 + 96 * v19);
      v6 = (__m128i *)(v18 + 96 * v17);
      if ( *(double *)v39.m128i_i64 <= *(double *)v21->m128i_i64 )
        break;
      *v6 = _mm_loadu_si128(v21);
      v6[1] = _mm_loadu_si128(v21 + 1);
      sub_2240AE0(v28, &v21[2]);
      v20 = v27;
      v27 = (__m128i *)&v21[4];
      sub_2240AE0(v20, &v21[4]);
      v28 = (__m128i *)&v21[2];
      v18 = v37;
      v17 = v19;
      if ( v32 >= v19 )
      {
        v6 = (__m128i *)v21;
        break;
      }
      v19 = (v19 - 1) / 2;
    }
  }
  v22 = _mm_loadu_si128(&v40);
  *v6 = _mm_loadu_si128(&v39);
  v6[1] = v22;
  sub_2240AE0(v28, v41);
  result = sub_2240AE0(v27, v43);
  if ( (_QWORD *)v43[0] != v44 )
    result = j_j___libc_free_0(v43[0], v44[0] + 1LL);
  if ( (_QWORD *)v41[0] != v42 )
    return j_j___libc_free_0(v41[0], v42[0] + 1LL);
  return result;
}
