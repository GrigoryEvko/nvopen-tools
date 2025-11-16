// Function: sub_1C574B0
// Address: 0x1c574b0
//
__int64 __fastcall sub_1C574B0(const __m128i *a1, __int64 a2)
{
  const __m128i *v3; // rbx
  __int64 *v4; // r14
  unsigned int v5; // r8d
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r10
  unsigned int v10; // r9d
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r10
  __m128i v14; // xmm1
  unsigned int v15; // esi
  __int64 *v16; // r13
  __int64 *v17; // r15
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // r8
  unsigned int v21; // edx
  int v22; // eax
  __int64 *v23; // r9
  __int64 v24; // rdi
  int v25; // r11d
  __int64 *v26; // r10
  __int64 *v27; // r11
  __int64 *v28; // rdi
  int v29; // eax
  int v30; // eax
  __int64 v31; // rax
  int v33; // r11d
  int v34; // eax
  __int64 v35; // rax
  int v36; // ecx
  int v37; // ecx
  __int64 v38; // r9
  unsigned int v39; // edx
  __int64 v40; // r8
  int v41; // ebx
  __int64 *v42; // r11
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
    v15 = *(_DWORD *)(a2 + 24);
    v16 = (__int64 *)v3[-1].m128i_i64[0];
    v17 = (__int64 *)v3;
    if ( v15 )
    {
      v5 = v15 - 1;
      v6 = *(_QWORD *)(a2 + 8);
      v7 = (v15 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v4 == *v8 )
      {
LABEL_3:
        v10 = *((_DWORD *)v8 + 2);
        goto LABEL_4;
      }
      v33 = 1;
      v23 = 0;
      while ( v9 != -8 )
      {
        if ( !v23 && v9 == -16 )
          v23 = v8;
        v7 = v5 & (v33 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v4 == *v8 )
          goto LABEL_3;
        ++v33;
      }
      if ( !v23 )
        v23 = v8;
      v34 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v22 = v34 + 1;
      if ( 4 * v22 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a2 + 20) - v22 > v15 >> 3 )
          goto LABEL_32;
        sub_1468630(a2, v15);
        v43 = *(_DWORD *)(a2 + 24);
        if ( !v43 )
        {
LABEL_81:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a2 + 8);
        v26 = 0;
        v46 = 1;
        v47 = v44 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v22 = *(_DWORD *)(a2 + 16) + 1;
        v23 = (__int64 *)(v45 + 16LL * v47);
        v48 = *v23;
        if ( *v4 == *v23 )
          goto LABEL_32;
        while ( v48 != -8 )
        {
          if ( !v26 && v48 == -16 )
            v26 = v23;
          v47 = v44 & (v46 + v47);
          v23 = (__int64 *)(v45 + 16LL * v47);
          v48 = *v23;
          if ( *v4 == *v23 )
            goto LABEL_32;
          ++v46;
        }
        goto LABEL_13;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_1468630(a2, 2 * v15);
    v18 = *(_DWORD *)(a2 + 24);
    if ( !v18 )
      goto LABEL_81;
    v19 = v18 - 1;
    v20 = *(_QWORD *)(a2 + 8);
    v21 = v19 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
    v22 = *(_DWORD *)(a2 + 16) + 1;
    v23 = (__int64 *)(v20 + 16LL * v21);
    v24 = *v23;
    if ( *v4 == *v23 )
      goto LABEL_32;
    v25 = 1;
    v26 = 0;
    while ( v24 != -8 )
    {
      if ( !v26 && v24 == -16 )
        v26 = v23;
      v21 = v19 & (v25 + v21);
      v23 = (__int64 *)(v20 + 16LL * v21);
      v24 = *v23;
      if ( *v4 == *v23 )
        goto LABEL_32;
      ++v25;
    }
LABEL_13:
    if ( v26 )
      v23 = v26;
LABEL_32:
    *(_DWORD *)(a2 + 16) = v22;
    if ( *v23 != -8 )
      --*(_DWORD *)(a2 + 20);
    v35 = *v4;
    *((_DWORD *)v23 + 2) = 0;
    *v23 = v35;
    v15 = *(_DWORD *)(a2 + 24);
    if ( !v15 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_36;
    }
    v6 = *(_QWORD *)(a2 + 8);
    v5 = v15 - 1;
    v10 = 0;
LABEL_4:
    v11 = v5 & (((unsigned int)*v16 >> 9) ^ ((unsigned int)*v16 >> 4));
    v12 = (__int64 *)(v6 + 16LL * v11);
    v13 = *v12;
    if ( *v16 != *v12 )
      break;
LABEL_5:
    v3 -= 2;
    if ( *((_DWORD *)v12 + 2) <= v10 )
      goto LABEL_25;
    v14 = _mm_loadu_si128(v3 + 1);
    v3[2] = _mm_loadu_si128(v3);
    v3[3] = v14;
  }
  v55 = 1;
  v27 = 0;
  while ( v13 != -8 )
  {
    if ( !v27 && v13 == -16 )
      v27 = v12;
    v11 = v5 & (v55 + v11);
    v12 = (__int64 *)(v6 + 16LL * v11);
    v13 = *v12;
    if ( *v16 == *v12 )
      goto LABEL_5;
    ++v55;
  }
  v28 = v27;
  if ( !v27 )
    v28 = v12;
  v29 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v30 = v29 + 1;
  if ( 4 * v30 >= 3 * v15 )
  {
LABEL_36:
    sub_1468630(a2, 2 * v15);
    v36 = *(_DWORD *)(a2 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a2 + 8);
      v39 = v37 & (((unsigned int)*v16 >> 9) ^ ((unsigned int)*v16 >> 4));
      v30 = *(_DWORD *)(a2 + 16) + 1;
      v28 = (__int64 *)(v38 + 16LL * v39);
      v40 = *v28;
      if ( *v28 == *v16 )
        goto LABEL_22;
      v41 = 1;
      v42 = 0;
      while ( v40 != -8 )
      {
        if ( !v42 && v40 == -16 )
          v42 = v28;
        v39 = v37 & (v41 + v39);
        v28 = (__int64 *)(v38 + 16LL * v39);
        v40 = *v28;
        if ( *v16 == *v28 )
          goto LABEL_22;
        ++v41;
      }
LABEL_40:
      if ( v42 )
        v28 = v42;
      goto LABEL_22;
    }
LABEL_80:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( v15 - (v30 + *(_DWORD *)(a2 + 20)) <= v15 >> 3 )
  {
    sub_1468630(a2, v15);
    v49 = *(_DWORD *)(a2 + 24);
    if ( v49 )
    {
      v50 = v49 - 1;
      v51 = *(_QWORD *)(a2 + 8);
      v42 = 0;
      v52 = 1;
      v53 = v50 & (((unsigned int)*v16 >> 9) ^ ((unsigned int)*v16 >> 4));
      v30 = *(_DWORD *)(a2 + 16) + 1;
      v28 = (__int64 *)(v51 + 16LL * v53);
      v54 = *v28;
      if ( *v28 == *v16 )
        goto LABEL_22;
      while ( v54 != -8 )
      {
        if ( !v42 && v54 == -16 )
          v42 = v28;
        v53 = v50 & (v52 + v53);
        v28 = (__int64 *)(v51 + 16LL * v53);
        v54 = *v28;
        if ( *v16 == *v28 )
          goto LABEL_22;
        ++v52;
      }
      goto LABEL_40;
    }
    goto LABEL_80;
  }
LABEL_22:
  *(_DWORD *)(a2 + 16) = v30;
  if ( *v28 != -8 )
    --*(_DWORD *)(a2 + 20);
  v31 = *v16;
  *((_DWORD *)v28 + 2) = 0;
  *v28 = v31;
LABEL_25:
  v17[2] = (__int64)v4;
  *v17 = v56;
  v17[1] = v57;
  v17[3] = v58;
  return v58;
}
