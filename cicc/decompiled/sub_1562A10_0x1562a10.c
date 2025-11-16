// Function: sub_1562A10
// Address: 0x1562a10
//
__m128i *__fastcall sub_1562A10(__m128i *a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v6; // rbx
  __m128i *v7; // r15
  void *v8; // r13
  size_t v9; // r14
  size_t v10; // r12
  size_t v11; // rdx
  int v12; // eax
  size_t v13; // rbx
  size_t v14; // rdx
  int v15; // eax
  __m128i *v16; // r13
  __m128i *v17; // rdi
  size_t v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rsi
  __m128i *v22; // rax
  __int64 *v23; // rsi
  __m128i *v24; // r12
  size_t v25; // rbx
  __int64 v26; // rax
  __m128i *v27; // rdx
  __m128i *v28; // r8
  __int64 v29; // rdi
  __m128i *v30; // rdi
  size_t v31; // r12
  size_t v32; // rdx
  int v33; // eax
  unsigned int v34; // edi
  __m128i *v35; // [rsp+8h] [rbp-98h]
  __m128i *v36; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  __int64 v38; // [rsp+10h] [rbp-90h]
  void *s2; // [rsp+30h] [rbp-70h] BYREF
  size_t v41; // [rsp+38h] [rbp-68h]
  __m128i v42; // [rsp+40h] [rbp-60h] BYREF
  __m128i *v43; // [rsp+50h] [rbp-50h] BYREF
  size_t n; // [rsp+58h] [rbp-48h]
  _QWORD v45[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( a4 )
  {
    v43 = (__m128i *)v45;
    sub_155C9E0((__int64 *)&v43, a4, (__int64)&a4[a5]);
  }
  else
  {
    n = 0;
    v43 = (__m128i *)v45;
    LOBYTE(v45[0]) = 0;
  }
  if ( a2 )
  {
    s2 = &v42;
    sub_155C9E0((__int64 *)&s2, a2, (__int64)&a2[a3]);
  }
  else
  {
    v41 = 0;
    s2 = &v42;
    v42.m128i_i8[0] = 0;
  }
  v6 = a1[1].m128i_i64[1];
  v7 = a1 + 1;
  v36 = a1 + 1;
  if ( !v6 )
  {
    v7 = a1 + 1;
    goto LABEL_37;
  }
  v8 = s2;
  v9 = v41;
  do
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v6 + 40);
      v11 = v9;
      if ( v10 <= v9 )
        v11 = *(_QWORD *)(v6 + 40);
      if ( v11 )
      {
        v12 = memcmp(*(const void **)(v6 + 32), v8, v11);
        if ( v12 )
          break;
      }
      if ( (__int64)(v10 - v9) >= 0x80000000LL )
        goto LABEL_16;
      if ( (__int64)(v10 - v9) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v12 = v10 - v9;
        break;
      }
LABEL_7:
      v6 = *(_QWORD *)(v6 + 24);
      if ( !v6 )
        goto LABEL_17;
    }
    if ( v12 < 0 )
      goto LABEL_7;
LABEL_16:
    v7 = (__m128i *)v6;
    v6 = *(_QWORD *)(v6 + 16);
  }
  while ( v6 );
LABEL_17:
  if ( v36 == v7 )
    goto LABEL_37;
  v13 = v7[2].m128i_u64[1];
  v14 = v9;
  if ( v13 <= v9 )
    v14 = v7[2].m128i_u64[1];
  if ( v14 )
  {
    v15 = memcmp(v8, (const void *)v7[2].m128i_i64[0], v14);
    if ( v15 )
    {
LABEL_25:
      if ( v15 < 0 )
        goto LABEL_37;
      goto LABEL_26;
    }
  }
  if ( (__int64)(v9 - v13) > 0x7FFFFFFF )
  {
LABEL_26:
    v16 = v7 + 5;
    goto LABEL_27;
  }
  if ( (__int64)(v9 - v13) >= (__int64)0xFFFFFFFF80000000LL )
  {
    v15 = v9 - v13;
    goto LABEL_25;
  }
LABEL_37:
  v22 = (__m128i *)sub_22077B0(96);
  v23 = (__int64 *)v7;
  v24 = v22 + 3;
  v7 = v22;
  v22[2].m128i_i64[0] = (__int64)v22[3].m128i_i64;
  if ( s2 == &v42 )
  {
    v22[3] = _mm_load_si128(&v42);
  }
  else
  {
    v22[2].m128i_i64[0] = (__int64)s2;
    v22[3].m128i_i64[0] = v42.m128i_i64[0];
  }
  v25 = v41;
  v16 = v22 + 5;
  v22[4].m128i_i64[1] = 0;
  v22[4].m128i_i64[0] = (__int64)v22[5].m128i_i64;
  s2 = &v42;
  v22[2].m128i_i64[1] = v25;
  v22[5].m128i_i8[0] = 0;
  v41 = 0;
  v42.m128i_i8[0] = 0;
  v26 = sub_1562360(&a1->m128i_i64[1], v23, (__int64)v22[2].m128i_i64);
  v28 = v27;
  if ( v27 )
  {
    if ( v36 == v27 || v26 )
    {
LABEL_42:
      v29 = 1;
      goto LABEL_43;
    }
    v32 = v27[2].m128i_u64[1];
    v31 = v32;
    if ( v25 <= v32 )
      v32 = v25;
    if ( v32
      && (v35 = v28,
          v33 = memcmp((const void *)v7[2].m128i_i64[0], (const void *)v28[2].m128i_i64[0], v32),
          v28 = v35,
          (v34 = v33) != 0) )
    {
LABEL_61:
      v29 = v34 >> 31;
    }
    else
    {
      v29 = 0;
      if ( (__int64)(v25 - v31) <= 0x7FFFFFFF )
      {
        if ( (__int64)(v25 - v31) < (__int64)0xFFFFFFFF80000000LL )
          goto LABEL_42;
        v34 = v25 - v31;
        goto LABEL_61;
      }
    }
LABEL_43:
    sub_220F040(v29, v7, v28, v36);
    ++a1[3].m128i_i64[0];
  }
  else
  {
    v30 = (__m128i *)v7[2].m128i_i64[0];
    if ( v24 != v30 )
    {
      v37 = v26;
      j_j___libc_free_0(v30, v7[3].m128i_i64[0] + 1);
      v26 = v37;
    }
    v38 = v26;
    j_j___libc_free_0(v7, 96);
    v16 = (__m128i *)(v38 + 80);
    v7 = (__m128i *)v38;
  }
LABEL_27:
  v17 = (__m128i *)v7[4].m128i_i64[0];
  v18 = n;
  if ( v43 == (__m128i *)v45 )
  {
    if ( n )
    {
      if ( n == 1 )
        v17->m128i_i8[0] = v45[0];
      else
        memcpy(v17, v45, n);
      v18 = n;
      v17 = (__m128i *)v7[4].m128i_i64[0];
    }
    v7[4].m128i_i64[1] = v18;
    v17->m128i_i8[v18] = 0;
    v17 = v43;
  }
  else
  {
    v19 = v45[0];
    if ( v17 == v16 )
    {
      v7[4].m128i_i64[0] = (__int64)v43;
      v7[4].m128i_i64[1] = v18;
      v7[5].m128i_i64[0] = v19;
    }
    else
    {
      v20 = v7[5].m128i_i64[0];
      v7[4].m128i_i64[0] = (__int64)v43;
      v7[4].m128i_i64[1] = v18;
      v7[5].m128i_i64[0] = v19;
      if ( v17 )
      {
        v43 = v17;
        v45[0] = v20;
        goto LABEL_31;
      }
    }
    v43 = (__m128i *)v45;
    v17 = (__m128i *)v45;
  }
LABEL_31:
  n = 0;
  v17->m128i_i8[0] = 0;
  if ( s2 != &v42 )
    j_j___libc_free_0(s2, v42.m128i_i64[0] + 1);
  if ( v43 != (__m128i *)v45 )
    j_j___libc_free_0(v43, v45[0] + 1LL);
  return a1;
}
