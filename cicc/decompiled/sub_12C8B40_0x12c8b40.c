// Function: sub_12C8B40
// Address: 0x12c8b40
//
__m128i *__fastcall sub_12C8B40(_QWORD *a1, __int64 a2)
{
  __m128i *v2; // r15
  __int64 v3; // r12
  const void *v4; // r13
  size_t v5; // r14
  size_t v6; // rbx
  size_t v7; // rdx
  int v8; // eax
  size_t v9; // r12
  size_t v10; // rdx
  int v11; // eax
  __int64 *v13; // r14
  __m128i *v14; // rax
  __m128i *v15; // r13
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  __m128i *v18; // rdx
  __int64 v19; // rbx
  __m128i *v20; // r14
  __int64 v21; // rdi
  __m128i *v22; // rdi
  __m128i *v23; // rdi
  size_t v24; // rbx
  size_t v25; // rdx
  unsigned int v26; // edi
  __m128i *v28; // [rsp+28h] [rbp-38h]

  v2 = (__m128i *)(a1 + 1);
  v3 = a1[2];
  v28 = (__m128i *)(a1 + 1);
  if ( !v3 )
  {
    v2 = (__m128i *)(a1 + 1);
    goto LABEL_24;
  }
  v4 = *(const void **)a2;
  v5 = *(_QWORD *)(a2 + 8);
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + 40);
      v7 = v5;
      if ( v6 <= v5 )
        v7 = *(_QWORD *)(v3 + 40);
      if ( v7 )
      {
        v8 = memcmp(*(const void **)(v3 + 32), v4, v7);
        if ( v8 )
          break;
      }
      if ( (__int64)(v6 - v5) >= 0x80000000LL )
        goto LABEL_12;
      if ( (__int64)(v6 - v5) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v8 = v6 - v5;
        break;
      }
LABEL_3:
      v3 = *(_QWORD *)(v3 + 24);
      if ( !v3 )
        goto LABEL_13;
    }
    if ( v8 < 0 )
      goto LABEL_3;
LABEL_12:
    v2 = (__m128i *)v3;
    v3 = *(_QWORD *)(v3 + 16);
  }
  while ( v3 );
LABEL_13:
  if ( v28 == v2 )
    goto LABEL_24;
  v9 = v2[2].m128i_u64[1];
  v10 = v5;
  if ( v9 <= v5 )
    v10 = v2[2].m128i_u64[1];
  if ( v10 && (v11 = memcmp(v4, (const void *)v2[2].m128i_i64[0], v10)) != 0 )
  {
LABEL_21:
    if ( v11 < 0 )
      goto LABEL_24;
  }
  else if ( (__int64)(v5 - v9) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v5 - v9) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v11 = v5 - v9;
      goto LABEL_21;
    }
LABEL_24:
    v13 = (__int64 *)v2;
    v14 = (__m128i *)sub_22077B0(168);
    v15 = v14 + 3;
    v2 = v14;
    v14[2].m128i_i64[0] = (__int64)v14[3].m128i_i64;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v14[3] = _mm_loadu_si128((const __m128i *)(a2 + 16));
    }
    else
    {
      v14[2].m128i_i64[0] = *(_QWORD *)a2;
      v14[3].m128i_i64[0] = *(_QWORD *)(a2 + 16);
    }
    v14[4].m128i_i64[1] = 0;
    v14[5].m128i_i8[0] = 0;
    *(_QWORD *)a2 = a2 + 16;
    v16 = *(_QWORD *)(a2 + 8);
    v14[4].m128i_i64[0] = (__int64)v14[5].m128i_i64;
    v14[6].m128i_i64[0] = (__int64)v14[7].m128i_i64;
    *(_QWORD *)(a2 + 8) = 0;
    *(_BYTE *)(a2 + 16) = 0;
    v14[2].m128i_i64[1] = v16;
    v14[6].m128i_i64[1] = 0;
    v14[7].m128i_i8[0] = 0;
    v14[8].m128i_i64[0] = (__int64)v14[9].m128i_i64;
    v14[8].m128i_i64[1] = 0;
    v14[9].m128i_i8[0] = 0;
    v14[10].m128i_i64[0] = 0;
    v17 = sub_12C88B0(a1, v13, (__int64)v14[2].m128i_i64);
    v19 = v17;
    v20 = v18;
    if ( v18 )
    {
      if ( v28 == v18 || v17 )
      {
LABEL_29:
        v21 = 1;
        goto LABEL_30;
      }
      v25 = v18[2].m128i_u64[1];
      v24 = v25;
      if ( v16 <= v25 )
        v25 = v16;
      if ( v25 && (v26 = memcmp((const void *)v2[2].m128i_i64[0], (const void *)v20[2].m128i_i64[0], v25)) != 0 )
      {
LABEL_39:
        v21 = v26 >> 31;
      }
      else
      {
        v21 = 0;
        if ( (__int64)(v16 - v24) <= 0x7FFFFFFF )
        {
          if ( (__int64)(v16 - v24) < (__int64)0xFFFFFFFF80000000LL )
            goto LABEL_29;
          v26 = v16 - v24;
          goto LABEL_39;
        }
      }
LABEL_30:
      sub_220F040(v21, v2, v20, v28);
      ++a1[5];
    }
    else
    {
      v22 = (__m128i *)v2[2].m128i_i64[0];
      if ( v15 != v22 )
        j_j___libc_free_0(v22, v2[3].m128i_i64[0] + 1);
      v23 = v2;
      v2 = (__m128i *)v19;
      j_j___libc_free_0(v23, 168);
    }
  }
  return v2 + 4;
}
