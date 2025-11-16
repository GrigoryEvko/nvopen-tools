// Function: sub_C286D0
// Address: 0xc286d0
//
__m128i *__fastcall sub_C286D0(_QWORD *a1, __int64 **a2, const __m128i **a3)
{
  __m128i *v5; // rax
  __m128i *v6; // r12
  const __m128i *v7; // rax
  __m128i *v8; // rcx
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int32 v14; // edi
  __int64 v15; // rdx
  __int64 v16; // rdx
  __m128i *v17; // rcx
  __int32 v18; // edi
  __int64 v19; // rdx
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // r13
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // r14
  __int64 v27; // r13
  _QWORD *v28; // rdi
  char v30; // al
  unsigned __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // r8
  char *v34; // r14
  __int64 v35; // r13
  __m128i ***v36; // rax
  __int64 *v37; // rdx
  size_t v38; // r13
  _QWORD *v39; // r10
  _QWORD *v40; // rsi
  unsigned __int64 v41; // rdi
  _QWORD *v42; // rcx
  unsigned __int64 v43; // rdx
  char *v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // [rsp+8h] [rbp-38h]
  unsigned __int64 v47; // [rsp+8h] [rbp-38h]

  v5 = (__m128i *)sub_22077B0(200);
  v6 = v5;
  if ( v5 )
    v5->m128i_i64[0] = 0;
  v7 = *a3;
  v8 = v6 + 6;
  v6->m128i_i64[1] = **a2;
  v9 = _mm_loadu_si128(v7 + 1);
  v10 = _mm_loadu_si128(v7 + 2);
  v6[1].m128i_i64[0] = v7->m128i_i64[0];
  v11 = v7->m128i_i64[1];
  v6[2] = v9;
  v6[1].m128i_i64[1] = v11;
  v12 = v7[3].m128i_i64[0];
  v6[3] = v10;
  v6[4].m128i_i64[0] = v12;
  v6[4].m128i_i64[1] = v7[3].m128i_i64[1];
  v6[5].m128i_i64[0] = v7[4].m128i_i64[0];
  v13 = v7[5].m128i_i64[1];
  if ( v13 )
  {
    v14 = v7[5].m128i_i32[0];
    v6[6].m128i_i64[1] = v13;
    v6[6].m128i_i32[0] = v14;
    v6[7].m128i_i64[0] = v7[6].m128i_i64[0];
    v6[7].m128i_i64[1] = v7[6].m128i_i64[1];
    *(_QWORD *)(v13 + 8) = v8;
    v15 = v7[7].m128i_i64[0];
    v7[5].m128i_i64[1] = 0;
    v6[8].m128i_i64[0] = v15;
    v7[6].m128i_i64[0] = (__int64)v7[5].m128i_i64;
    v7[6].m128i_i64[1] = (__int64)v7[5].m128i_i64;
    v7[7].m128i_i64[0] = 0;
  }
  else
  {
    v6[6].m128i_i32[0] = 0;
    v6[6].m128i_i64[1] = 0;
    v6[7].m128i_i64[0] = (__int64)v8;
    v6[7].m128i_i64[1] = (__int64)v8;
    v6[8].m128i_i64[0] = 0;
  }
  v16 = v7[8].m128i_i64[1];
  v17 = v6 + 9;
  if ( v16 )
  {
    v18 = v7[8].m128i_i32[0];
    v6[9].m128i_i64[1] = v16;
    v6[9].m128i_i32[0] = v18;
    v6[10].m128i_i64[0] = v7[9].m128i_i64[0];
    v6[10].m128i_i64[1] = v7[9].m128i_i64[1];
    *(_QWORD *)(v16 + 8) = v17;
    v19 = v7[10].m128i_i64[0];
    v7[8].m128i_i64[1] = 0;
    v6[11].m128i_i64[0] = v19;
    v7[9].m128i_i64[0] = (__int64)v7[8].m128i_i64;
    v7[9].m128i_i64[1] = (__int64)v7[8].m128i_i64;
    v7[10].m128i_i64[0] = 0;
  }
  else
  {
    v6[9].m128i_i32[0] = 0;
    v6[9].m128i_i64[1] = 0;
    v6[10].m128i_i64[0] = (__int64)v17;
    v6[10].m128i_i64[1] = (__int64)v17;
    v6[11].m128i_i64[0] = 0;
  }
  v20 = a1[1];
  v21 = v6->m128i_u64[1];
  v6[11].m128i_i64[1] = v7[10].m128i_i64[1];
  v22 = v21 % v20;
  v23 = sub_C1DD00(a1, v21 % v20, &v6->m128i_i64[1], v21);
  if ( v23 )
  {
    v24 = *v23;
    if ( v24 )
    {
      v25 = v6[9].m128i_i64[1];
      v26 = v24;
      while ( v25 )
      {
        v27 = v25;
        sub_C1F230(*(_QWORD **)(v25 + 24));
        v28 = *(_QWORD **)(v25 + 56);
        v25 = *(_QWORD *)(v25 + 16);
        sub_C1F480(v28);
        j_j___libc_free_0(v27, 88);
      }
      sub_C1EF60((_QWORD *)v6[6].m128i_i64[1]);
      j_j___libc_free_0(v6, 200);
      return (__m128i *)v26;
    }
  }
  v30 = sub_222DA10(a1 + 4, v20, a1[3], 1);
  v33 = v31;
  if ( v30 )
  {
    if ( v31 == 1 )
    {
      v34 = (char *)(a1 + 6);
      a1[6] = 0;
      v39 = a1 + 6;
    }
    else
    {
      if ( v31 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v20, v31, v32);
      v38 = 8 * v31;
      v46 = v31;
      v34 = (char *)sub_22077B0(8 * v31);
      memset(v34, 0, v38);
      v33 = v46;
      v39 = a1 + 6;
    }
    v40 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v40 )
    {
LABEL_30:
      if ( (_QWORD *)*a1 != v39 )
      {
        v47 = v33;
        j_j___libc_free_0(*a1, 8LL * a1[1]);
        v33 = v47;
      }
      a1[1] = v33;
      *a1 = v34;
      v22 = v21 % v33;
      goto LABEL_15;
    }
    v41 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v42 = v40;
        v40 = (_QWORD *)*v40;
        v43 = v42[24] % v33;
        v44 = &v34[8 * v43];
        if ( !*(_QWORD *)v44 )
          break;
        *v42 = **(_QWORD **)v44;
        **(_QWORD **)v44 = v42;
LABEL_26:
        if ( !v40 )
          goto LABEL_30;
      }
      *v42 = a1[2];
      a1[2] = v42;
      *(_QWORD *)v44 = a1 + 2;
      if ( !*v42 )
      {
        v41 = v43;
        goto LABEL_26;
      }
      *(_QWORD *)&v34[8 * v41] = v42;
      v41 = v43;
      if ( !v40 )
        goto LABEL_30;
    }
  }
  v34 = (char *)*a1;
LABEL_15:
  v35 = 8 * v22;
  v6[12].m128i_i64[0] = v21;
  v36 = (__m128i ***)&v34[v35];
  v37 = *(__int64 **)&v34[v35];
  if ( v37 )
  {
    v6->m128i_i64[0] = *v37;
    **v36 = v6;
  }
  else
  {
    v45 = a1[2];
    a1[2] = v6;
    v6->m128i_i64[0] = v45;
    if ( v45 )
    {
      *(_QWORD *)&v34[8 * (*(_QWORD *)(v45 + 192) % a1[1])] = v6;
      v36 = (__m128i ***)(v35 + *a1);
    }
    *v36 = (__m128i **)(a1 + 2);
  }
  ++a1[3];
  return v6;
}
