// Function: sub_E6A4F0
// Address: 0xe6a4f0
//
__m128i *__fastcall sub_E6A4F0(__int64 a1, __m128i *a2)
{
  __int64 v2; // rbx
  __m128i *v3; // rax
  __m128i *v4; // rdx
  __m128i *v5; // r14
  size_t v6; // r13
  __int32 v7; // eax
  __m128i v8; // xmm0
  __int64 v9; // r15
  __int32 v10; // eax
  const void *v11; // r12
  size_t v12; // rdx
  signed __int64 v13; // rax
  __int64 v14; // rax
  char v15; // dl
  size_t v16; // rbx
  const void *v17; // rsi
  size_t v18; // rbx
  void *v19; // r9
  const void *v20; // rdi
  const void *v21; // rsi
  bool v22; // al
  __int64 v23; // rdi
  size_t v25; // rdx
  __int64 v26; // r12
  __m128i *v27; // rdi
  __m128i *v28; // [rsp+8h] [rbp-68h]
  __int64 m128i_i64; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  void *s2; // [rsp+28h] [rbp-48h]
  void *s1; // [rsp+30h] [rbp-40h]
  unsigned __int32 v34; // [rsp+38h] [rbp-38h]
  __int32 v35; // [rsp+3Ch] [rbp-34h]

  v2 = a1;
  v3 = (__m128i *)sub_22077B0(96);
  v4 = (__m128i *)a2->m128i_i64[0];
  v5 = v3;
  m128i_i64 = (__int64)v3[2].m128i_i64;
  v28 = v3 + 3;
  v3[2].m128i_i64[0] = (__int64)v3[3].m128i_i64;
  if ( v4 == &a2[1] )
  {
    v3[3] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v3[2].m128i_i64[0] = (__int64)v4;
    v3[3].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v6 = a2->m128i_u64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v7 = a2[3].m128i_i32[0];
  a2[1].m128i_i8[0] = 0;
  a2->m128i_i64[1] = 0;
  v8 = _mm_loadu_si128(a2 + 2);
  v35 = v7;
  v9 = *(_QWORD *)(a1 + 16);
  v5[5].m128i_i32[0] = v7;
  v10 = a2[3].m128i_i32[1];
  v5[2].m128i_i64[1] = v6;
  v34 = v10;
  v5[5].m128i_i32[1] = v10;
  v5[5].m128i_i64[1] = 0;
  v31 = a1 + 8;
  v5[4] = v8;
  if ( !v9 )
  {
    v9 = a1 + 8;
    if ( v31 == *(_QWORD *)(a1 + 24) )
    {
      v9 = a1 + 8;
      v23 = 1;
      goto LABEL_28;
    }
    goto LABEL_37;
  }
  v11 = (const void *)v5[2].m128i_i64[0];
  while ( 1 )
  {
    v16 = *(_QWORD *)(v9 + 40);
    v17 = *(const void **)(v9 + 32);
    if ( v6 != v16 )
    {
      v12 = *(_QWORD *)(v9 + 40);
      if ( v6 <= v16 )
        v12 = v6;
      if ( !v12 || (LODWORD(v13) = memcmp(v11, v17, v12), !(_DWORD)v13) )
      {
        v13 = v6 - v16;
        if ( (__int64)(v6 - v16) >= 0x80000000LL )
          goto LABEL_23;
        if ( v13 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_12;
      }
      goto LABEL_11;
    }
    if ( v6 )
    {
      LODWORD(v13) = memcmp(v11, v17, v6);
      if ( (_DWORD)v13 )
        goto LABEL_11;
    }
    v18 = v5[4].m128i_u64[1];
    v19 = *(void **)(v9 + 72);
    v20 = (const void *)v5[4].m128i_i64[0];
    v21 = *(const void **)(v9 + 64);
    if ( (void *)v18 == v19 )
      break;
    v25 = *(_QWORD *)(v9 + 72);
    if ( v18 <= (unsigned __int64)v19 )
      v25 = v5[4].m128i_u64[1];
    if ( v25 )
    {
      s1 = *(void **)(v9 + 72);
      LODWORD(v13) = memcmp(v20, v21, v25);
      v19 = s1;
      if ( (_DWORD)v13 )
        goto LABEL_11;
    }
    if ( v18 >= (unsigned __int64)v19 )
      goto LABEL_23;
LABEL_12:
    v14 = *(_QWORD *)(v9 + 16);
    v15 = 1;
    if ( !v14 )
      goto LABEL_24;
LABEL_13:
    v9 = v14;
  }
  if ( v18 )
  {
    s2 = *(void **)(v9 + 64);
    if ( memcmp(v20, v21, v5[4].m128i_u64[1]) )
    {
      LODWORD(v13) = memcmp(v20, s2, v18);
      if ( !(_DWORD)v13 )
        goto LABEL_23;
LABEL_11:
      if ( (int)v13 >= 0 )
        goto LABEL_23;
      goto LABEL_12;
    }
  }
  if ( v35 == *(_DWORD *)(v9 + 80) )
    v22 = v34 < *(_DWORD *)(v9 + 84);
  else
    v22 = v35 < *(_DWORD *)(v9 + 80);
  if ( v22 )
    goto LABEL_12;
LABEL_23:
  v14 = *(_QWORD *)(v9 + 24);
  v15 = 0;
  if ( v14 )
    goto LABEL_13;
LABEL_24:
  v2 = a1;
  if ( v15 )
  {
    if ( *(_QWORD *)(a1 + 24) == v9 )
      goto LABEL_26;
LABEL_37:
    v26 = sub_220EF80(v9);
    if ( sub_E63F80(v26 + 32, m128i_i64) )
      goto LABEL_26;
    v9 = v26;
    goto LABEL_39;
  }
  if ( !sub_E63F80(v9 + 32, m128i_i64) )
  {
LABEL_39:
    v27 = (__m128i *)v5[2].m128i_i64[0];
    if ( v28 != v27 )
      j_j___libc_free_0(v27, v5[3].m128i_i64[0] + 1);
    j_j___libc_free_0(v5, 96);
    return (__m128i *)v9;
  }
LABEL_26:
  v23 = 1;
  if ( v31 != v9 )
    v23 = (unsigned __int8)sub_E63F80(m128i_i64, v9 + 32);
LABEL_28:
  sub_220F040(v23, v5, v9, v31);
  ++*(_QWORD *)(v2 + 40);
  return v5;
}
