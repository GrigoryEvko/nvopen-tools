// Function: sub_1910330
// Address: 0x1910330
//
_DWORD *__fastcall sub_1910330(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r8
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // r12d
  unsigned int v10; // ecx
  int *v11; // rax
  int v12; // edx
  _DWORD *v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r14
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rax
  _DWORD *v24; // rdx
  _QWORD *v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rax
  unsigned int v29; // edx
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // edx
  int v33; // r11d
  int *v34; // r10
  int v35; // edi
  int v36; // ecx
  int v37; // r9d
  int v38; // eax
  int v39; // esi
  __int64 v40; // r8
  unsigned int v41; // edx
  int v42; // edi
  int v43; // r10d
  int *v44; // r9
  int v45; // eax
  int v46; // edx
  __int64 v47; // r8
  __int64 v48; // r12
  int v49; // edi
  int v50; // esi
  __int64 v51; // [rsp+0h] [rbp-60h]
  __m128i v52; // [rsp+10h] [rbp-50h] BYREF
  __m128i v53; // [rsp+20h] [rbp-40h]

  v3 = a1 + 376;
  v7 = *(_DWORD *)(a1 + 400);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 376);
    goto LABEL_67;
  }
  v8 = *(_QWORD *)(a1 + 384);
  v9 = 37 * a3;
  v10 = (v7 - 1) & (37 * a3);
  v11 = (int *)(v8 + 40LL * v10);
  v12 = *v11;
  if ( *v11 == a3 )
    goto LABEL_3;
  v33 = 1;
  v34 = 0;
  while ( v12 != -1 )
  {
    if ( v12 == -2 && !v34 )
      v34 = v11;
    v10 = (v7 - 1) & (v33 + v10);
    v11 = (int *)(v8 + 40LL * v10);
    v12 = *v11;
    if ( *v11 == a3 )
      goto LABEL_3;
    ++v33;
  }
  v35 = *(_DWORD *)(a1 + 392);
  if ( v34 )
    v11 = v34;
  ++*(_QWORD *)(a1 + 376);
  v36 = v35 + 1;
  if ( 4 * (v35 + 1) >= 3 * v7 )
  {
LABEL_67:
    sub_190FC70(v3, 2 * v7);
    v38 = *(_DWORD *)(a1 + 400);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 384);
      v41 = (v38 - 1) & (37 * a3);
      v36 = *(_DWORD *)(a1 + 392) + 1;
      v11 = (int *)(v40 + 40LL * v41);
      v42 = *v11;
      if ( *v11 == a3 )
        goto LABEL_61;
      v43 = 1;
      v44 = 0;
      while ( v42 != -1 )
      {
        if ( v42 == -2 && !v44 )
          v44 = v11;
        v41 = v39 & (v43 + v41);
        v11 = (int *)(v40 + 40LL * v41);
        v42 = *v11;
        if ( *v11 == a3 )
          goto LABEL_61;
        ++v43;
      }
LABEL_71:
      if ( v44 )
        v11 = v44;
      goto LABEL_61;
    }
LABEL_92:
    ++*(_DWORD *)(a1 + 392);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 396) - v36 <= v7 >> 3 )
  {
    sub_190FC70(v3, v7);
    v45 = *(_DWORD *)(a1 + 400);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a1 + 384);
      v44 = 0;
      LODWORD(v48) = (v45 - 1) & v9;
      v36 = *(_DWORD *)(a1 + 392) + 1;
      v49 = 1;
      v11 = (int *)(v47 + 40LL * (unsigned int)v48);
      v50 = *v11;
      if ( *v11 == a3 )
        goto LABEL_61;
      while ( v50 != -1 )
      {
        if ( v50 == -2 && !v44 )
          v44 = v11;
        v48 = v46 & (unsigned int)(v48 + v49);
        v11 = (int *)(v47 + 40 * v48);
        v50 = *v11;
        if ( *v11 == a3 )
          goto LABEL_61;
        ++v49;
      }
      goto LABEL_71;
    }
    goto LABEL_92;
  }
LABEL_61:
  *(_DWORD *)(a1 + 392) = v36;
  if ( *v11 != -1 )
    --*(_DWORD *)(a1 + 396);
  *v11 = a3;
  *((_QWORD *)v11 + 1) = 0;
  *((_QWORD *)v11 + 2) = 0;
  *((_QWORD *)v11 + 3) = 0;
  *((_QWORD *)v11 + 4) = 0;
LABEL_3:
  v52 = _mm_loadu_si128((const __m128i *)(v11 + 2));
  v13 = (_DWORD *)v52.m128i_i64[0];
  v53 = _mm_loadu_si128((const __m128i *)(v11 + 6));
  if ( !v52.m128i_i64[0] )
    return v13;
  v14 = *(_QWORD *)(a1 + 24);
  v15 = *(unsigned int *)(v14 + 48);
  if ( !(_DWORD)v15 )
    goto LABEL_45;
  v16 = *(_QWORD *)(v14 + 32);
  v17 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (__int64 *)(v16 + 16LL * v17);
  v19 = *v18;
  if ( a2 != *v18 )
  {
    v32 = 1;
    while ( v19 != -8 )
    {
      v37 = v32 + 1;
      v17 = (v15 - 1) & (v32 + v17);
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( a2 == *v18 )
        goto LABEL_6;
      v32 = v37;
    }
    goto LABEL_45;
  }
LABEL_6:
  if ( v18 == (__int64 *)(v16 + 16 * v15) )
  {
LABEL_45:
    v20 = 0;
    sub_190B2A0(a1, (__int64)&v52);
    goto LABEL_46;
  }
  v20 = v18[1];
  v21 = sub_190B2A0(a1, (__int64)&v52);
  v13 = (_DWORD *)v21;
  if ( v21 == v20 || !v20 )
    goto LABEL_46;
  if ( !v21 )
    goto LABEL_20;
  if ( v21 == *(_QWORD *)(v20 + 8) )
    goto LABEL_46;
  if ( *(_QWORD *)(v21 + 8) == v20 || *(_DWORD *)(v21 + 16) >= *(_DWORD *)(v20 + 16) )
    goto LABEL_19;
  if ( *(_BYTE *)(v14 + 72) )
  {
LABEL_52:
    if ( *(_DWORD *)(v20 + 48) >= v13[12] && *(_DWORD *)(v20 + 52) <= v13[13] )
      goto LABEL_46;
LABEL_19:
    v13 = 0;
    goto LABEL_20;
  }
  v22 = *(_DWORD *)(v14 + 76) + 1;
  *(_DWORD *)(v14 + 76) = v22;
  if ( v22 > 0x20 )
  {
    sub_15CC640(v14);
    goto LABEL_52;
  }
  v23 = v20;
  do
  {
    v24 = (_DWORD *)v23;
    v23 = *(_QWORD *)(v23 + 8);
  }
  while ( v23 && v13[4] <= *(_DWORD *)(v23 + 16) );
  if ( v13 != v24 )
    goto LABEL_19;
LABEL_46:
  v13 = (_DWORD *)v52.m128i_i64[0];
  if ( *(_BYTE *)(v52.m128i_i64[0] + 16) <= 0x10u )
    return v13;
LABEL_20:
  v25 = (_QWORD *)v53.m128i_i64[0];
  if ( v53.m128i_i64[0] )
  {
    while ( 1 )
    {
      v26 = *(_QWORD *)(a1 + 24);
      v27 = sub_190B2A0(a1, (__int64)v25);
      if ( v27 == v20 || !v20 )
        goto LABEL_30;
      if ( !v27 )
        goto LABEL_33;
      if ( v27 == *(_QWORD *)(v20 + 8) )
        goto LABEL_30;
      if ( v20 == *(_QWORD *)(v27 + 8) || *(_DWORD *)(v27 + 16) >= *(_DWORD *)(v20 + 16) )
        goto LABEL_33;
      if ( *(_BYTE *)(v26 + 72) )
      {
        if ( *(_DWORD *)(v20 + 48) < *(_DWORD *)(v27 + 48) || *(_DWORD *)(v20 + 52) > *(_DWORD *)(v27 + 52) )
          goto LABEL_33;
        goto LABEL_30;
      }
      v29 = *(_DWORD *)(v26 + 76) + 1;
      *(_DWORD *)(v26 + 76) = v29;
      if ( v29 > 0x20 )
      {
        v51 = v27;
        sub_15CC640(v26);
        if ( *(_DWORD *)(v20 + 48) < *(_DWORD *)(v51 + 48) || *(_DWORD *)(v20 + 52) > *(_DWORD *)(v51 + 52) )
          goto LABEL_33;
        goto LABEL_30;
      }
      v30 = v20;
      do
      {
        v31 = v30;
        v30 = *(_QWORD *)(v30 + 8);
      }
      while ( v30 && *(_DWORD *)(v27 + 16) <= *(_DWORD *)(v30 + 16) );
      if ( v27 == v31 )
      {
LABEL_30:
        if ( *(_BYTE *)(*v25 + 16LL) <= 0x10u )
          return (_DWORD *)*v25;
        if ( !v13 )
          v13 = (_DWORD *)*v25;
LABEL_33:
        v25 = (_QWORD *)v25[2];
        if ( !v25 )
          return v13;
      }
      else
      {
        v25 = (_QWORD *)v25[2];
        if ( !v25 )
          return v13;
      }
    }
  }
  return v13;
}
