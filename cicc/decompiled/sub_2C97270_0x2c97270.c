// Function: sub_2C97270
// Address: 0x2c97270
//
__int64 __fastcall sub_2C97270(const __m128i *a1, __int64 a2)
{
  const __m128i *v3; // rbx
  __int64 *v4; // r14
  unsigned int v5; // edi
  __int64 v6; // r8
  __int64 *v7; // r9
  int v8; // r11d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r10
  unsigned int v12; // r9d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  __m128i v16; // xmm1
  unsigned int v17; // esi
  __int64 *v18; // r13
  __int64 *v19; // r15
  int v20; // ecx
  int v21; // ecx
  __int64 v22; // r8
  unsigned int v23; // edx
  int v24; // eax
  __int64 v25; // rdi
  int v26; // r11d
  __int64 *v27; // r10
  int v28; // eax
  __int64 v29; // rax
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // r9
  unsigned int v33; // edx
  int v34; // eax
  __int64 *v35; // rdi
  __int64 v36; // r8
  int v37; // ebx
  __int64 *v38; // r11
  __int64 *v39; // r11
  int v40; // eax
  __int64 v41; // rax
  int v43; // ecx
  int v44; // ecx
  __int64 v45; // r8
  int v46; // r11d
  unsigned int v47; // edx
  __int64 v48; // rdi
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // r9
  int v52; // ebx
  unsigned int v53; // edx
  __int64 v54; // r8
  int v55; // [rsp+14h] [rbp-4Ch]
  __int64 v56; // [rsp+18h] [rbp-48h]
  __int64 v57; // [rsp+20h] [rbp-40h]
  __int64 v58; // [rsp+28h] [rbp-38h]

  v3 = a1;
  v4 = (__int64 *)a1[1].m128i_i64[0];
  v56 = a1->m128i_i64[0];
  v57 = a1->m128i_i64[1];
  v58 = a1[1].m128i_i64[1];
  while ( 1 )
  {
    v17 = *(_DWORD *)(a2 + 24);
    v18 = (__int64 *)v3[-1].m128i_i64[0];
    v19 = (__int64 *)v3;
    if ( v17 )
    {
      v5 = v17 - 1;
      v6 = *(_QWORD *)(a2 + 8);
      v7 = 0;
      v8 = 1;
      v9 = (v17 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( *v4 == *v10 )
      {
LABEL_3:
        v12 = *((_DWORD *)v10 + 2);
        goto LABEL_4;
      }
      while ( v11 != -4096 )
      {
        if ( !v7 && v11 == -8192 )
          v7 = v10;
        v9 = v5 & (v8 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( *v4 == *v10 )
          goto LABEL_3;
        ++v8;
      }
      if ( !v7 )
        v7 = v10;
      v28 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v24 = v28 + 1;
      if ( 4 * v24 < 3 * v17 )
      {
        if ( v17 - *(_DWORD *)(a2 + 20) - v24 > v17 >> 3 )
          goto LABEL_26;
        sub_2C96F50(a2, v17);
        v43 = *(_DWORD *)(a2 + 24);
        if ( !v43 )
        {
LABEL_80:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a2 + 8);
        v27 = 0;
        v46 = 1;
        v47 = v44 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v24 = *(_DWORD *)(a2 + 16) + 1;
        v7 = (__int64 *)(v45 + 16LL * v47);
        v48 = *v7;
        if ( *v4 == *v7 )
          goto LABEL_26;
        while ( v48 != -4096 )
        {
          if ( !v27 && v48 == -8192 )
            v27 = v7;
          v47 = v44 & (v46 + v47);
          v7 = (__int64 *)(v45 + 16LL * v47);
          v48 = *v7;
          if ( *v4 == *v7 )
            goto LABEL_26;
          ++v46;
        }
        goto LABEL_13;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_2C96F50(a2, 2 * v17);
    v20 = *(_DWORD *)(a2 + 24);
    if ( !v20 )
      goto LABEL_80;
    v21 = v20 - 1;
    v22 = *(_QWORD *)(a2 + 8);
    v23 = v21 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
    v24 = *(_DWORD *)(a2 + 16) + 1;
    v7 = (__int64 *)(v22 + 16LL * v23);
    v25 = *v7;
    if ( *v4 == *v7 )
      goto LABEL_26;
    v26 = 1;
    v27 = 0;
    while ( v25 != -4096 )
    {
      if ( !v27 && v25 == -8192 )
        v27 = v7;
      v23 = v21 & (v26 + v23);
      v7 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v7;
      if ( *v4 == *v7 )
        goto LABEL_26;
      ++v26;
    }
LABEL_13:
    if ( v27 )
      v7 = v27;
LABEL_26:
    *(_DWORD *)(a2 + 16) = v24;
    if ( *v7 != -4096 )
      --*(_DWORD *)(a2 + 20);
    v29 = *v4;
    *((_DWORD *)v7 + 2) = 0;
    *v7 = v29;
    v17 = *(_DWORD *)(a2 + 24);
    if ( !v17 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_30;
    }
    v6 = *(_QWORD *)(a2 + 8);
    v5 = v17 - 1;
    v12 = 0;
LABEL_4:
    v13 = v5 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
    v14 = (__int64 *)(v6 + 16LL * v13);
    v15 = *v14;
    if ( *v18 != *v14 )
      break;
LABEL_5:
    v3 -= 2;
    if ( v12 >= *((_DWORD *)v14 + 2) )
      goto LABEL_46;
    v16 = _mm_loadu_si128(v3 + 1);
    v3[2] = _mm_loadu_si128(v3);
    v3[3] = v16;
  }
  v55 = 1;
  v39 = 0;
  while ( v15 != -4096 )
  {
    if ( !v39 && v15 == -8192 )
      v39 = v14;
    v13 = v5 & (v55 + v13);
    v14 = (__int64 *)(v6 + 16LL * v13);
    v15 = *v14;
    if ( *v18 == *v14 )
      goto LABEL_5;
    ++v55;
  }
  v35 = v39;
  if ( !v39 )
    v35 = v14;
  v40 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v34 = v40 + 1;
  if ( 4 * v34 < 3 * v17 )
  {
    if ( v17 - (v34 + *(_DWORD *)(a2 + 20)) > v17 >> 3 )
      goto LABEL_43;
    sub_2C96F50(a2, v17);
    v49 = *(_DWORD *)(a2 + 24);
    if ( v49 )
    {
      v50 = v49 - 1;
      v51 = *(_QWORD *)(a2 + 8);
      v38 = 0;
      v52 = 1;
      v53 = v50 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
      v34 = *(_DWORD *)(a2 + 16) + 1;
      v35 = (__int64 *)(v51 + 16LL * v53);
      v54 = *v35;
      if ( *v35 == *v18 )
        goto LABEL_43;
      while ( v54 != -4096 )
      {
        if ( !v38 && v54 == -8192 )
          v38 = v35;
        v53 = v50 & (v52 + v53);
        v35 = (__int64 *)(v51 + 16LL * v53);
        v54 = *v35;
        if ( *v18 == *v35 )
          goto LABEL_43;
        ++v52;
      }
LABEL_34:
      if ( v38 )
        v35 = v38;
      goto LABEL_43;
    }
LABEL_79:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
LABEL_30:
  sub_2C96F50(a2, 2 * v17);
  v30 = *(_DWORD *)(a2 + 24);
  if ( !v30 )
    goto LABEL_79;
  v31 = v30 - 1;
  v32 = *(_QWORD *)(a2 + 8);
  v33 = v31 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
  v34 = *(_DWORD *)(a2 + 16) + 1;
  v35 = (__int64 *)(v32 + 16LL * v33);
  v36 = *v35;
  if ( *v35 != *v18 )
  {
    v37 = 1;
    v38 = 0;
    while ( v36 != -4096 )
    {
      if ( !v38 && v36 == -8192 )
        v38 = v35;
      v33 = v31 & (v37 + v33);
      v35 = (__int64 *)(v32 + 16LL * v33);
      v36 = *v35;
      if ( *v18 == *v35 )
        goto LABEL_43;
      ++v37;
    }
    goto LABEL_34;
  }
LABEL_43:
  *(_DWORD *)(a2 + 16) = v34;
  if ( *v35 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v41 = *v18;
  *((_DWORD *)v35 + 2) = 0;
  *v35 = v41;
LABEL_46:
  v19[2] = (__int64)v4;
  *v19 = v56;
  v19[1] = v57;
  v19[3] = v58;
  return v58;
}
