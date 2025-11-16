// Function: sub_13FF050
// Address: 0x13ff050
//
__int64 *__fastcall sub_13FF050(__int64 *a1, const __m128i *a2)
{
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // rbx
  int v6; // eax
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  int v9; // esi
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // r8
  _QWORD *v13; // rax
  unsigned int v14; // esi
  __int64 v15; // rdi
  int v16; // r10d
  __int64 *v17; // rcx
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r9
  __int8 *v21; // r13
  unsigned __int64 v22; // r14
  __m128i *v23; // rbx
  __int64 v24; // rax
  const __m128i *v26; // rdi
  int v27; // eax
  int v28; // eax
  int v29; // edi
  __int64 v30; // rdx
  __int64 v31; // rax
  __m128i *v32; // rsi
  __int64 v33; // rdx
  __m128i *v34; // rsi
  unsigned __int64 v35; // r13
  __int64 v36; // rax
  __m128i *v37; // rdx
  const __m128i *v38; // rax
  int v39; // r9d
  __int64 *v40; // [rsp+8h] [rbp-68h] BYREF
  __m128i v41; // [rsp+10h] [rbp-60h] BYREF
  __int64 v42; // [rsp+20h] [rbp-50h]
  const __m128i *v43; // [rsp+30h] [rbp-40h] BYREF
  const __m128i *v44; // [rsp+38h] [rbp-38h] BYREF
  __m128i *v45; // [rsp+40h] [rbp-30h]
  const __m128i *v46; // [rsp+48h] [rbp-28h]

  v3 = a2->m128i_i64[0];
  v4 = a2->m128i_i64[1];
  v5 = **(_QWORD **)(*(_QWORD *)a2->m128i_i64[0] + 32LL);
  v46 = 0;
  v6 = *(_DWORD *)(v4 + 24);
  v43 = a2;
  v44 = 0;
  v45 = 0;
  v7 = *(_QWORD **)v3;
  if ( v6 )
  {
    v8 = *(_QWORD *)(v4 + 8);
    v9 = v6 - 1;
    v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v5 == *v11 )
    {
LABEL_3:
      v13 = (_QWORD *)v11[1];
      if ( v7 != v13 )
      {
        while ( v13 )
        {
          v13 = (_QWORD *)*v13;
          if ( v7 == v13 )
            goto LABEL_6;
        }
LABEL_17:
        v41.m128i_i64[0] = (__int64)v43;
        goto LABEL_12;
      }
    }
    else
    {
      v27 = 1;
      while ( v12 != -8 )
      {
        v39 = v27 + 1;
        v10 = v9 & (v27 + v10);
        v11 = (__int64 *)(v8 + 16LL * v10);
        v12 = *v11;
        if ( v5 == *v11 )
          goto LABEL_3;
        v27 = v39;
      }
      if ( v7 )
        goto LABEL_17;
    }
  }
  else if ( v7 )
  {
    v41.m128i_i64[0] = (__int64)a2;
LABEL_12:
    v26 = 0;
    v21 = 0;
    v23 = 0;
    goto LABEL_13;
  }
LABEL_6:
  v41.m128i_i64[0] = v5;
  v41.m128i_i32[2] = 0;
  v14 = *(_DWORD *)(v3 + 32);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v3 + 16);
    v16 = 1;
    v17 = 0;
    v18 = (v14 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v19 = (__int64 *)(v15 + 16LL * v18);
    v20 = *v19;
    if ( v5 == *v19 )
    {
LABEL_8:
      v21 = 0;
      v22 = 0;
      v23 = 0;
      v41.m128i_i64[0] = (__int64)v43;
      goto LABEL_9;
    }
    while ( v20 != -8 )
    {
      if ( v17 || v20 != -16 )
        v19 = v17;
      v18 = (v14 - 1) & (v16 + v18);
      v20 = *(_QWORD *)(v15 + 16LL * v18);
      if ( v5 == v20 )
        goto LABEL_8;
      ++v16;
      v17 = v19;
      v19 = (__int64 *)(v15 + 16LL * v18);
    }
    if ( !v17 )
      v17 = v19;
    v28 = *(_DWORD *)(v3 + 24);
    ++*(_QWORD *)(v3 + 8);
    v29 = v28 + 1;
    if ( 4 * (v28 + 1) < 3 * v14 )
    {
      v30 = v5;
      if ( v14 - *(_DWORD *)(v3 + 28) - v29 > v14 >> 3 )
        goto LABEL_27;
      goto LABEL_47;
    }
  }
  else
  {
    ++*(_QWORD *)(v3 + 8);
  }
  v14 *= 2;
LABEL_47:
  sub_13FEAC0(v3 + 8, v14);
  sub_13FDDE0(v3 + 8, v41.m128i_i64, &v40);
  v17 = v40;
  v30 = v41.m128i_i64[0];
  v29 = *(_DWORD *)(v3 + 24) + 1;
LABEL_27:
  *(_DWORD *)(v3 + 24) = v29;
  if ( *v17 != -8 )
    --*(_DWORD *)(v3 + 28);
  *v17 = v30;
  *((_DWORD *)v17 + 2) = v41.m128i_i32[2];
  v31 = sub_157EBA0(v5);
  v41.m128i_i64[0] = v5;
  v32 = v45;
  v41.m128i_i64[1] = v31;
  LODWORD(v42) = 0;
  if ( v45 == v46 )
  {
    sub_13FDF40(&v44, v45, &v41);
  }
  else
  {
    if ( v45 )
    {
      *v45 = _mm_loadu_si128(&v41);
      v32[1].m128i_i64[0] = v42;
      v32 = v45;
    }
    v45 = (__m128i *)((char *)v32 + 24);
  }
  sub_13FEC80(&v43);
  v34 = v45;
  v26 = v44;
  v41.m128i_i64[0] = (__int64)v43;
  v35 = (char *)v45 - (char *)v44;
  if ( v45 == v44 )
  {
    v23 = 0;
  }
  else
  {
    if ( v35 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v44, v45, v33);
    v36 = sub_22077B0((char *)v45 - (char *)v44);
    v34 = v45;
    v26 = v44;
    v23 = (__m128i *)v36;
  }
  v21 = &v23->m128i_i8[v35];
  if ( v26 != v34 )
  {
    v37 = v23;
    v38 = v26;
    do
    {
      if ( v37 )
      {
        *v37 = _mm_loadu_si128(v38);
        v37[1].m128i_i64[0] = v38[1].m128i_i64[0];
      }
      v38 = (const __m128i *)((char *)v38 + 24);
      v37 = (__m128i *)((char *)v37 + 24);
    }
    while ( v38 != v34 );
    v22 = (unsigned __int64)&v23[1].m128i_u64[((unsigned __int64)((char *)&v38[-2].m128i_u64[1] - (char *)v26) >> 3) + 1];
    goto LABEL_42;
  }
LABEL_13:
  v22 = (unsigned __int64)v23;
LABEL_42:
  if ( v26 )
    j_j___libc_free_0(v26, (char *)v46 - (char *)v26);
LABEL_9:
  v24 = v41.m128i_i64[0];
  a1[1] = (__int64)v23;
  a1[2] = v22;
  *a1 = v24;
  a1[3] = (__int64)v21;
  return a1;
}
