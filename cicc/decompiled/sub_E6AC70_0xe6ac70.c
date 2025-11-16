// Function: sub_E6AC70
// Address: 0xe6ac70
//
__int64 __fastcall sub_E6AC70(_QWORD *a1, __m128i *a2)
{
  __m128i *v2; // rax
  __m128i *v3; // rdx
  __m128i *v4; // r12
  __int32 v5; // eax
  size_t v6; // r14
  __int8 v7; // bl
  __int64 v8; // r15
  char v9; // al
  size_t v10; // r13
  const void *v11; // r8
  const void *v12; // r9
  size_t v13; // rdx
  int v14; // eax
  signed __int64 v15; // rax
  int v16; // eax
  __int64 v17; // r13
  __int64 v18; // rax
  char v19; // dl
  size_t v20; // r13
  const void *v21; // r8
  const void *v22; // r9
  size_t v23; // rdx
  int v24; // eax
  signed __int64 v25; // rax
  signed __int64 v26; // rax
  _QWORD *v27; // r13
  __m128i *v28; // r8
  size_t v29; // rbx
  const void *v30; // r10
  size_t v31; // rdx
  int v32; // eax
  signed __int64 v33; // rax
  signed __int64 v34; // rax
  __int64 v35; // rdi
  size_t v37; // rbx
  const void *v38; // r10
  size_t v39; // rdx
  int v40; // eax
  signed __int64 v41; // rax
  int v42; // eax
  __m128i *v43; // [rsp+0h] [rbp-60h]
  _QWORD *v44; // [rsp+8h] [rbp-58h]
  size_t n; // [rsp+18h] [rbp-48h]
  size_t na; // [rsp+18h] [rbp-48h]
  size_t nb; // [rsp+18h] [rbp-48h]
  size_t nc; // [rsp+18h] [rbp-48h]
  void *s1; // [rsp+20h] [rbp-40h]
  void *s1a; // [rsp+20h] [rbp-40h]
  __m128i *s1b; // [rsp+20h] [rbp-40h]
  __m128i *s1c; // [rsp+20h] [rbp-40h]
  void *s2; // [rsp+28h] [rbp-38h]
  void *s2a; // [rsp+28h] [rbp-38h]
  void *s2b; // [rsp+28h] [rbp-38h]
  __m128i *s2c; // [rsp+28h] [rbp-38h]
  void *s2d; // [rsp+28h] [rbp-38h]
  __m128i *s2e; // [rsp+28h] [rbp-38h]

  v2 = (__m128i *)sub_22077B0(80);
  v3 = (__m128i *)a2->m128i_i64[0];
  v4 = v2;
  v43 = v2 + 3;
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
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v5 = a2[2].m128i_i32[0];
  v6 = a2->m128i_u64[1];
  a2[1].m128i_i8[0] = 0;
  v4[4].m128i_i32[0] = v5;
  a2->m128i_i64[1] = 0;
  v7 = a2[2].m128i_i8[4];
  v8 = a1[2];
  v4[2].m128i_i64[1] = v6;
  v4[4].m128i_i8[4] = v7;
  v4[4].m128i_i64[1] = 0;
  v44 = a1 + 1;
  if ( !v8 )
  {
    if ( v44 == (_QWORD *)a1[3] )
    {
      v27 = a1 + 1;
      v35 = 1;
LABEL_58:
      sub_220F040(v35, v4, v27, v44);
      ++a1[5];
      return (__int64)v4;
    }
    v8 = (__int64)(a1 + 1);
LABEL_39:
    v27 = (_QWORD *)v8;
    v8 = sub_220EF80(v8);
    goto LABEL_40;
  }
  v9 = *(_BYTE *)(v8 + 68);
  if ( !v7 )
    goto LABEL_22;
LABEL_5:
  if ( !v9 )
    goto LABEL_20;
  v10 = *(_QWORD *)(v8 + 40);
  v11 = (const void *)v4[2].m128i_i64[0];
  v12 = *(const void **)(v8 + 32);
  v13 = v10;
  if ( v6 <= v10 )
    v13 = v6;
  if ( v13 )
  {
    n = v13;
    s1 = *(void **)(v8 + 32);
    s2 = (void *)v4[2].m128i_i64[0];
    v14 = memcmp(s2, s1, v13);
    v11 = s2;
    v12 = s1;
    v13 = n;
    if ( v14 )
    {
      if ( v14 >= 0 )
      {
        v16 = memcmp(s1, s2, n);
        if ( v16 )
          goto LABEL_18;
        goto LABEL_15;
      }
      goto LABEL_20;
    }
    v15 = v6 - v10;
    if ( (__int64)(v6 - v10) >= 0x80000000LL )
      goto LABEL_14;
  }
  else
  {
    v15 = v6 - v10;
    if ( (__int64)(v6 - v10) >= 0x80000000LL )
      goto LABEL_15;
  }
  if ( v15 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v15 >= 0 )
  {
    if ( v13 )
    {
LABEL_14:
      v16 = memcmp(v12, v11, v13);
      if ( v16 )
        goto LABEL_18;
    }
LABEL_15:
    v17 = v10 - v6;
    if ( v17 >= 0x80000000LL )
      goto LABEL_19;
    if ( v17 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      goto LABEL_36;
    v16 = v17;
LABEL_18:
    if ( v16 < 0 )
      goto LABEL_36;
LABEL_19:
    if ( v4[4].m128i_i8[0] < *(_BYTE *)(v8 + 64) )
      goto LABEL_20;
    while ( 1 )
    {
      do
      {
LABEL_36:
        v18 = *(_QWORD *)(v8 + 24);
        v19 = 0;
        if ( !v18 )
          goto LABEL_37;
LABEL_21:
        v8 = v18;
        v9 = *(_BYTE *)(v18 + 68);
        if ( v7 )
          goto LABEL_5;
LABEL_22:
        ;
      }
      while ( v9 );
      v20 = *(_QWORD *)(v8 + 40);
      v21 = (const void *)v4[2].m128i_i64[0];
      v22 = *(const void **)(v8 + 32);
      v23 = v20;
      if ( v6 <= v20 )
        v23 = v6;
      if ( !v23 )
        break;
      na = v23;
      s1a = *(void **)(v8 + 32);
      s2a = (void *)v4[2].m128i_i64[0];
      v24 = memcmp(s2a, s1a, v23);
      v21 = s2a;
      v22 = s1a;
      v23 = na;
      if ( v24 )
      {
        if ( v24 < 0 )
          goto LABEL_20;
      }
      else
      {
        v25 = v6 - v20;
        if ( (__int64)(v6 - v20) < 0x80000000LL )
          goto LABEL_28;
      }
LABEL_31:
      LODWORD(v26) = memcmp(v22, v21, v23);
      if ( (_DWORD)v26 )
      {
LABEL_34:
        if ( (int)v26 >= 0 )
          goto LABEL_35;
      }
      else
      {
LABEL_32:
        v26 = v20 - v6;
        if ( (__int64)(v20 - v6) >= 0x80000000LL )
        {
LABEL_35:
          if ( v4[4].m128i_i32[0] < *(_DWORD *)(v8 + 64) )
            goto LABEL_20;
        }
        else if ( v26 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          goto LABEL_34;
        }
      }
    }
    v25 = v6 - v20;
    if ( (__int64)(v6 - v20) >= 0x80000000LL )
      goto LABEL_32;
LABEL_28:
    if ( v25 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v25 < 0 )
      goto LABEL_20;
    if ( !v23 )
      goto LABEL_32;
    goto LABEL_31;
  }
LABEL_20:
  v18 = *(_QWORD *)(v8 + 16);
  v19 = 1;
  if ( v18 )
    goto LABEL_21;
LABEL_37:
  v27 = (_QWORD *)v8;
  if ( v19 )
  {
    if ( a1[3] == v8 )
    {
      v27 = (_QWORD *)v8;
LABEL_56:
      v35 = 1;
      if ( v44 != v27 )
        v35 = (unsigned __int8)sub_E64230((__int64)v4[2].m128i_i64, (__int64)(v27 + 4));
      goto LABEL_58;
    }
    goto LABEL_39;
  }
LABEL_40:
  if ( *(_BYTE *)(v8 + 68) )
  {
    if ( !v7 )
      goto LABEL_55;
    v37 = *(_QWORD *)(v8 + 40);
    v38 = *(const void **)(v8 + 32);
    v28 = (__m128i *)v4[2].m128i_i64[0];
    v39 = v37;
    if ( v6 <= v37 )
      v39 = v6;
    if ( v39 )
    {
      nc = v39;
      s1c = (__m128i *)v4[2].m128i_i64[0];
      s2d = *(void **)(v8 + 32);
      v40 = memcmp(s2d, s1c, v39);
      v38 = s2d;
      v28 = s1c;
      v39 = nc;
      if ( v40 )
      {
        if ( v40 < 0 )
          goto LABEL_55;
LABEL_77:
        s2e = v28;
        v42 = memcmp(v28, v38, v39);
        v28 = s2e;
        if ( v42 )
          goto LABEL_81;
        goto LABEL_78;
      }
      v41 = v37 - v6;
      if ( (__int64)(v37 - v6) > 0x7FFFFFFF )
        goto LABEL_77;
    }
    else
    {
      v41 = v37 - v6;
      if ( (__int64)(v37 - v6) > 0x7FFFFFFF )
        goto LABEL_78;
    }
    if ( v41 < (__int64)0xFFFFFFFF80000000LL || (int)v41 < 0 )
      goto LABEL_55;
    if ( v39 )
      goto LABEL_77;
LABEL_78:
    if ( (__int64)(v6 - v37) > 0x7FFFFFFF )
    {
LABEL_82:
      if ( *(_BYTE *)(v8 + 64) >= v4[4].m128i_i8[0] )
        goto LABEL_83;
      goto LABEL_55;
    }
    if ( (__int64)(v6 - v37) < (__int64)0xFFFFFFFF80000000LL )
      goto LABEL_83;
    v42 = v6 - v37;
LABEL_81:
    if ( v42 < 0 )
      goto LABEL_83;
    goto LABEL_82;
  }
  v28 = (__m128i *)v4[2].m128i_i64[0];
  if ( v7 )
    goto LABEL_83;
  v29 = *(_QWORD *)(v8 + 40);
  v30 = *(const void **)(v8 + 32);
  v31 = v29;
  if ( v6 <= v29 )
    v31 = v6;
  if ( v31 )
  {
    nb = v31;
    s1b = (__m128i *)v4[2].m128i_i64[0];
    s2b = *(void **)(v8 + 32);
    v32 = memcmp(s2b, s1b, v31);
    v30 = s2b;
    v28 = s1b;
    v31 = nb;
    if ( v32 )
    {
      if ( v32 >= 0 )
      {
LABEL_50:
        s2c = v28;
        LODWORD(v34) = memcmp(v28, v30, v31);
        v28 = s2c;
        if ( (_DWORD)v34 )
          goto LABEL_53;
        goto LABEL_51;
      }
      goto LABEL_55;
    }
    v33 = v29 - v6;
    if ( (__int64)(v29 - v6) > 0x7FFFFFFF )
      goto LABEL_50;
  }
  else
  {
    v33 = v29 - v6;
    if ( (__int64)(v29 - v6) > 0x7FFFFFFF )
      goto LABEL_51;
  }
  if ( v33 >= (__int64)0xFFFFFFFF80000000LL && (int)v33 >= 0 )
  {
    if ( v31 )
      goto LABEL_50;
LABEL_51:
    v34 = v6 - v29;
    if ( (__int64)(v6 - v29) > 0x7FFFFFFF )
    {
LABEL_54:
      if ( *(_DWORD *)(v8 + 64) >= v4[4].m128i_i32[0] )
        goto LABEL_83;
      goto LABEL_55;
    }
    if ( v34 < (__int64)0xFFFFFFFF80000000LL )
      goto LABEL_83;
LABEL_53:
    if ( (int)v34 < 0 )
      goto LABEL_83;
    goto LABEL_54;
  }
LABEL_55:
  if ( v27 )
    goto LABEL_56;
  v28 = (__m128i *)v4[2].m128i_i64[0];
  v8 = 0;
LABEL_83:
  if ( v43 != v28 )
    j_j___libc_free_0(v28, v4[3].m128i_i64[0] + 1);
  j_j___libc_free_0(v4, 80);
  return v8;
}
