// Function: sub_1DA0CF0
// Address: 0x1da0cf0
//
__m128i *__fastcall sub_1DA0CF0(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v6; // rax
  const __m128i *v7; // r15
  __m128i *v8; // r12
  __m128i v9; // xmm0
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  __int32 v13; // eax
  unsigned __int64 v14; // rdx
  _QWORD *v15; // r8
  _QWORD *v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  bool v19; // di
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  _QWORD *v22; // rdx
  bool v23; // al
  unsigned __int64 v24; // r10
  unsigned __int64 v25; // r9
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rsi
  bool v28; // al
  bool v29; // al
  __int64 v30; // rdi
  bool v32; // dl
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rdi
  __int64 v37; // rsi
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // r9
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rsi
  unsigned __int64 v42; // rcx
  unsigned __int64 v43; // rcx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // r9
  bool v47; // al
  unsigned __int64 v48; // [rsp+8h] [rbp-48h]
  unsigned __int64 v49; // [rsp+10h] [rbp-40h]
  _QWORD *v50; // [rsp+10h] [rbp-40h]
  unsigned __int64 v51; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v52; // [rsp+18h] [rbp-38h]
  _QWORD *v53; // [rsp+18h] [rbp-38h]
  _QWORD *v54; // [rsp+18h] [rbp-38h]
  _QWORD *v55; // [rsp+18h] [rbp-38h]

  v6 = sub_22077B0(168);
  v7 = *a3;
  v8 = (__m128i *)v6;
  v9 = _mm_loadu_si128(*a3);
  v52 = (unsigned __int64 *)(v6 + 32);
  v10 = v6 + 56;
  v11 = v7[1].m128i_i64[0];
  v8[2] = v9;
  v8[3].m128i_i64[0] = v11;
  v12 = v7[1].m128i_i64[1];
  v8[3].m128i_i64[1] = v12;
  if ( v12 )
    sub_1623A60(v10, v12, 2);
  v8[4].m128i_i64[0] = v7[2].m128i_i64[0];
  sub_16CCCB0(&v8[4].m128i_i64[1], (__int64)v8[7].m128i_i64, (__int64)&v7[2].m128i_i64[1]);
  v13 = v7[7].m128i_i32[0];
  v14 = v7[7].m128i_u64[1];
  v15 = a1 + 1;
  v8[10].m128i_i32[0] = 0;
  v8[9].m128i_i32[0] = v13;
  v16 = a2;
  v8[9].m128i_i64[1] = v14;
  if ( a1 + 1 == a2 )
  {
    if ( a1[5] )
    {
      v22 = (_QWORD *)a1[4];
      v26 = v8[2].m128i_u64[0];
      v27 = v22[4];
      v28 = v27 < v26;
      if ( v27 == v26 )
      {
        v41 = v22[5];
        v42 = v8[2].m128i_u64[1];
        v28 = v41 < v42;
        if ( v41 == v42 )
          v28 = v22[19] < v8[9].m128i_i64[1];
      }
      if ( v28 )
        goto LABEL_35;
    }
    goto LABEL_17;
  }
  v17 = v8[2].m128i_u64[0];
  v18 = a2[4];
  v19 = v17 < v18;
  if ( v17 == v18 )
  {
    v24 = v8[2].m128i_u64[1];
    v25 = a2[5];
    v19 = v24 < v25;
    if ( v24 == v25 )
      v19 = v14 < a2[19];
  }
  if ( !v19 )
  {
    v32 = v17 > v18;
    if ( v17 == v18 )
    {
      v39 = a2[5];
      v40 = v8[2].m128i_u64[1];
      v32 = v39 < v40;
      if ( v39 == v40 )
        v32 = a2[19] < v8[9].m128i_i64[1];
    }
    if ( !v32 )
      goto LABEL_30;
    v48 = a2[4];
    v51 = v8[2].m128i_u64[0];
    if ( (_QWORD *)a1[4] == a2 )
    {
      v16 = 0;
      goto LABEL_11;
    }
    v33 = sub_220EEE0(a2);
    v34 = v51;
    v15 = a1 + 1;
    v35 = v48;
    v22 = (_QWORD *)v33;
    if ( v51 == *(_QWORD *)(v33 + 32) )
    {
      v45 = *(_QWORD *)(v33 + 40);
      v46 = v8[2].m128i_u64[1];
      v47 = v46 < v45;
      if ( v46 == v45 )
        v47 = v8[9].m128i_i64[1] < v22[19];
      if ( !v47 )
        goto LABEL_17;
      if ( !a2[3] )
      {
LABEL_29:
        v22 = a2;
        goto LABEL_41;
      }
    }
    else
    {
      if ( v51 >= *(_QWORD *)(v33 + 32) )
        goto LABEL_17;
      if ( !a2[3] )
        goto LABEL_29;
    }
LABEL_21:
    v30 = 1;
LABEL_22:
    sub_220F040(v30, v8, v22, v15);
    ++a1[5];
    return v8;
  }
  v49 = v8[2].m128i_u64[0];
  if ( (_QWORD *)a1[3] == a2 )
  {
LABEL_10:
    v16 = a2;
LABEL_11:
    v22 = a2;
LABEL_18:
    v29 = v16 != 0;
    goto LABEL_19;
  }
  v20 = sub_220EF80(a2);
  v15 = a1 + 1;
  v21 = *(_QWORD *)(v20 + 32);
  v22 = (_QWORD *)v20;
  v23 = v49 > v21;
  if ( v49 == v21 )
  {
    v38 = v8[2].m128i_u64[1];
    if ( v22[5] != v38 )
    {
      if ( v22[5] < v38 )
      {
LABEL_9:
        if ( v22[3] )
          goto LABEL_10;
LABEL_35:
        v29 = 0;
LABEL_19:
        if ( v15 == v22 || v29 )
          goto LABEL_21;
        v34 = v8[2].m128i_u64[0];
        v35 = v22[4];
LABEL_41:
        LOBYTE(v30) = v34 < v35;
        if ( v34 == v35 )
        {
          v43 = v8[2].m128i_u64[1];
          v44 = v22[5];
          LOBYTE(v30) = v43 < v44;
          if ( v43 == v44 )
            LOBYTE(v30) = v8[9].m128i_i64[1] < v22[19];
        }
        v30 = (unsigned __int8)v30;
        goto LABEL_22;
      }
      goto LABEL_17;
    }
    v23 = v22[19] < v8[9].m128i_i64[1];
  }
  if ( v23 )
    goto LABEL_9;
LABEL_17:
  v50 = v15;
  v16 = sub_1DA0B60((__int64)a1, v52);
  v15 = v50;
  if ( v22 )
    goto LABEL_18;
LABEL_30:
  v36 = v8[5].m128i_u64[1];
  if ( v36 != v8[5].m128i_i64[0] )
  {
    v53 = v16;
    _libc_free(v36);
    v16 = v53;
  }
  v37 = v8[3].m128i_i64[1];
  if ( v37 )
  {
    v54 = v16;
    sub_161E7C0(v10, v37);
    v16 = v54;
  }
  v55 = v16;
  j_j___libc_free_0(v8, 168);
  return (__m128i *)v55;
}
