// Function: sub_38C2480
// Address: 0x38c2480
//
unsigned __int64 __fastcall sub_38C2480(_QWORD *a1, __m128i *a2)
{
  __m128i *v2; // rax
  __m128i *v3; // rdx
  unsigned __int64 v4; // r13
  size_t v5; // r15
  __m128i v6; // xmm0
  __int32 v7; // eax
  __int64 v8; // rbx
  size_t v9; // rdx
  signed __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rsi
  size_t v14; // r14
  const void *v15; // r12
  size_t v16; // r8
  size_t v17; // r9
  const void *v18; // rdi
  const void *v19; // rsi
  int v20; // eax
  const void *v21; // rcx
  _QWORD *v22; // r8
  size_t v23; // r12
  const void *v24; // r14
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // r12
  char v28; // di
  __int64 v30; // rax
  size_t v31; // r12
  size_t v32; // rcx
  const void *v33; // r15
  const void *v34; // rsi
  char v35; // al
  int v36; // eax
  __m128i *v37; // [rsp+8h] [rbp-78h]
  size_t v38; // [rsp+10h] [rbp-70h]
  unsigned __int32 v39; // [rsp+24h] [rbp-5Ch]
  _QWORD *v40; // [rsp+28h] [rbp-58h]
  size_t v41; // [rsp+30h] [rbp-50h]
  size_t v42; // [rsp+30h] [rbp-50h]
  _QWORD *v43; // [rsp+30h] [rbp-50h]
  _QWORD *v44; // [rsp+30h] [rbp-50h]
  size_t v45; // [rsp+30h] [rbp-50h]
  size_t v46; // [rsp+38h] [rbp-48h]
  _QWORD *v47; // [rsp+38h] [rbp-48h]
  size_t v48; // [rsp+38h] [rbp-48h]
  size_t v49; // [rsp+38h] [rbp-48h]
  _QWORD *v50; // [rsp+38h] [rbp-48h]
  size_t v51; // [rsp+38h] [rbp-48h]
  size_t v52; // [rsp+38h] [rbp-48h]
  _QWORD *v53; // [rsp+38h] [rbp-48h]
  __m128i *s1; // [rsp+48h] [rbp-38h]
  _QWORD *s1a; // [rsp+48h] [rbp-38h]

  v2 = (__m128i *)sub_22077B0(0x60u);
  v3 = (__m128i *)a2->m128i_i64[0];
  v4 = (unsigned __int64)v2;
  v37 = v2 + 3;
  v2[2].m128i_i64[0] = (__int64)v2[3].m128i_i64;
  if ( v3 == &a2[1] )
  {
    v2[3] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v2[2].m128i_i64[0] = (__int64)v3;
    v2[3].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v5 = a2->m128i_u64[1];
  v6 = _mm_loadu_si128(a2 + 2);
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v7 = a2[3].m128i_i32[0];
  a2->m128i_i64[1] = 0;
  a2[1].m128i_i8[0] = 0;
  v39 = v7;
  *(_DWORD *)(v4 + 80) = v7;
  *(_QWORD *)(v4 + 40) = v5;
  v8 = a1[2];
  *(__m128i *)(v4 + 64) = v6;
  *(_QWORD *)(v4 + 88) = 0;
  v40 = a1 + 1;
  if ( !v8 )
  {
    v8 = (__int64)(a1 + 1);
    if ( v40 != (_QWORD *)a1[3] )
    {
      s1 = *(__m128i **)(v4 + 32);
      goto LABEL_54;
    }
    v22 = a1 + 1;
    v28 = 1;
LABEL_38:
    sub_220F040(v28, v4, v22, v40);
    ++a1[5];
    return v4;
  }
  s1 = *(__m128i **)(v4 + 32);
  while ( 1 )
  {
    v14 = *(_QWORD *)(v8 + 40);
    v15 = *(const void **)(v8 + 32);
    if ( v5 != v14 )
    {
      v9 = *(_QWORD *)(v8 + 40);
      if ( v5 <= v14 )
        v9 = v5;
      if ( v9 )
      {
        LODWORD(v10) = memcmp(s1, *(const void **)(v8 + 32), v9);
        if ( (_DWORD)v10 )
          goto LABEL_11;
      }
      v10 = v5 - v14;
      if ( (__int64)(v5 - v14) < 0x80000000LL )
      {
        if ( v10 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_25;
LABEL_11:
        v11 = (unsigned int)v10 >> 31;
LABEL_12:
        v12 = *(_QWORD *)(v8 + 16);
        v13 = *(_QWORD *)(v8 + 24);
        if ( (_BYTE)v11 )
          goto LABEL_14;
        goto LABEL_13;
      }
      goto LABEL_42;
    }
    if ( v5 )
    {
      LODWORD(v10) = memcmp(s1, *(const void **)(v8 + 32), v5);
      if ( (_DWORD)v10 )
        goto LABEL_11;
    }
    v16 = *(_QWORD *)(v4 + 72);
    v17 = *(_QWORD *)(v8 + 72);
    v18 = *(const void **)(v4 + 64);
    v19 = *(const void **)(v8 + 64);
    if ( v16 == v17 )
    {
      v49 = *(_QWORD *)(v8 + 72);
      if ( !v16 || (v38 = *(_QWORD *)(v4 + 72), !memcmp(v18, v19, v16)) )
      {
        LOBYTE(v11) = v39 < *(_DWORD *)(v8 + 80);
        goto LABEL_12;
      }
      v16 = v38;
      v17 = v49;
      goto LABEL_21;
    }
    if ( v16 > v17 )
      break;
LABEL_21:
    if ( v16 )
    {
      v41 = v17;
      v46 = v16;
      v20 = memcmp(v18, v19, v16);
      v16 = v46;
      v17 = v41;
      if ( v20 )
        goto LABEL_41;
    }
    if ( v16 != v17 )
      goto LABEL_24;
LABEL_42:
    v13 = *(_QWORD *)(v8 + 24);
LABEL_13:
    v12 = v13;
    LOBYTE(v11) = 0;
LABEL_14:
    if ( !v12 )
      goto LABEL_26;
LABEL_15:
    v8 = v12;
  }
  if ( !v17 )
    goto LABEL_42;
  v42 = *(_QWORD *)(v4 + 72);
  v48 = *(_QWORD *)(v8 + 72);
  v20 = memcmp(v18, v19, v48);
  v17 = v48;
  v16 = v42;
  if ( !v20 )
  {
LABEL_24:
    if ( v16 < v17 )
      goto LABEL_25;
    goto LABEL_42;
  }
LABEL_41:
  if ( v20 >= 0 )
    goto LABEL_42;
LABEL_25:
  v12 = *(_QWORD *)(v8 + 16);
  LOBYTE(v11) = 1;
  if ( v12 )
    goto LABEL_15;
LABEL_26:
  v21 = v15;
  v22 = (_QWORD *)v8;
  v23 = v14;
  v24 = v21;
  if ( !(_BYTE)v11 )
    goto LABEL_27;
  if ( a1[3] == v8 )
  {
    v22 = (_QWORD *)v8;
    goto LABEL_36;
  }
LABEL_54:
  v30 = sub_220EF80(v8);
  v22 = (_QWORD *)v8;
  v24 = *(const void **)(v30 + 32);
  v23 = *(_QWORD *)(v30 + 40);
  v8 = v30;
LABEL_27:
  if ( v23 == v5 )
  {
    if ( v23 )
    {
      v50 = v22;
      v26 = memcmp(v24, s1, v23);
      v22 = v50;
      if ( v26 )
        goto LABEL_49;
    }
    v31 = *(_QWORD *)(v8 + 72);
    v32 = *(_QWORD *)(v4 + 72);
    v33 = *(const void **)(v8 + 64);
    v34 = *(const void **)(v4 + 64);
    if ( v31 == v32 )
    {
      v45 = *(_QWORD *)(v4 + 72);
      if ( !v31 || (v53 = v22, v36 = memcmp(v33, v34, v31), v22 = v53, !v36) )
      {
        if ( v39 > *(_DWORD *)(v8 + 80) )
          goto LABEL_35;
        goto LABEL_50;
      }
      v32 = v45;
    }
    else if ( v31 > v32 )
    {
      if ( !v32 )
        goto LABEL_50;
      v44 = v22;
      v52 = *(_QWORD *)(v4 + 72);
      v26 = memcmp(*(const void **)(v8 + 64), v34, v52);
      v32 = v52;
      v22 = v44;
      if ( v26 )
        goto LABEL_49;
      goto LABEL_61;
    }
    if ( v31 )
    {
      v43 = v22;
      v51 = v32;
      v26 = memcmp(v33, v34, v31);
      v32 = v51;
      v22 = v43;
      if ( v26 )
        goto LABEL_49;
    }
    if ( v31 == v32 )
      goto LABEL_50;
LABEL_61:
    if ( v31 < v32 )
      goto LABEL_35;
    goto LABEL_50;
  }
  v25 = v5;
  if ( v23 <= v5 )
    v25 = v23;
  if ( v25 )
  {
    v47 = v22;
    v26 = memcmp(v24, s1, v25);
    v22 = v47;
    if ( v26 )
    {
LABEL_49:
      if ( v26 < 0 )
        goto LABEL_35;
      goto LABEL_50;
    }
  }
  v27 = v23 - v5;
  if ( v27 <= 0x7FFFFFFF && (v27 < (__int64)0xFFFFFFFF80000000LL || (int)v27 < 0) )
  {
LABEL_35:
    if ( v22 )
    {
LABEL_36:
      v28 = 1;
      if ( v40 != v22 )
      {
        s1a = v22;
        v35 = sub_38BCBF0(v4 + 32, (__int64)(v22 + 4));
        v22 = s1a;
        v28 = v35;
      }
      goto LABEL_38;
    }
    v8 = 0;
  }
LABEL_50:
  if ( v37 != s1 )
    j_j___libc_free_0((unsigned __int64)s1);
  j_j___libc_free_0(v4);
  return v8;
}
