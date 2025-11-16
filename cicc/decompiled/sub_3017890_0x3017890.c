// Function: sub_3017890
// Address: 0x3017890
//
_QWORD *__fastcall sub_3017890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r8
  int v11; // r11d
  __int64 *v12; // rdx
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // esi
  int v17; // r15d
  __int64 v18; // r10
  __int64 v19; // r8
  int v20; // r11d
  _QWORD *v21; // rdx
  unsigned int v22; // edi
  _QWORD *v23; // rax
  __int64 v24; // rcx
  _QWORD *result; // rax
  int v26; // eax
  int v27; // ecx
  int v28; // eax
  int v29; // esi
  __int64 v30; // r8
  unsigned int v31; // eax
  int v32; // ecx
  __int64 v33; // rdi
  int v34; // r10d
  _QWORD *v35; // r9
  int v36; // eax
  int v37; // eax
  int v38; // esi
  __int64 v39; // rdi
  unsigned int v40; // eax
  __int64 v41; // r8
  int v42; // r10d
  __int64 *v43; // r9
  int v44; // eax
  int v45; // eax
  __int64 v46; // rdi
  int v47; // r9d
  unsigned int v48; // r12d
  _QWORD *v49; // r8
  __int64 v50; // rsi
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdi
  int v54; // r9d
  unsigned int v55; // r15d
  __int64 *v56; // r8
  __int64 v57; // rsi

  v4 = a1 + 64;
  v9 = *(_DWORD *)(a1 + 88);
  if ( v9 )
  {
    v10 = *(_QWORD *)(a1 + 72);
    v11 = 1;
    v12 = 0;
    v13 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
    {
LABEL_3:
      v16 = *(_DWORD *)(a1 + 120);
      v17 = *((_DWORD *)v14 + 2);
      v18 = a1 + 96;
      if ( v16 )
        goto LABEL_4;
LABEL_20:
      ++*(_QWORD *)(a1 + 96);
      goto LABEL_21;
    }
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      v13 = (v9 - 1) & (v11 + v13);
      v14 = (__int64 *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == a2 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
    v26 = *(_DWORD *)(a1 + 80);
    ++*(_QWORD *)(a1 + 64);
    v27 = v26 + 1;
    if ( 4 * (v26 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 84) - v27 > v9 >> 3 )
        goto LABEL_17;
      sub_30132A0(v4, v9);
      v51 = *(_DWORD *)(a1 + 88);
      if ( v51 )
      {
        v52 = v51 - 1;
        v53 = *(_QWORD *)(a1 + 72);
        v54 = 1;
        v55 = v52 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v56 = 0;
        v27 = *(_DWORD *)(a1 + 80) + 1;
        v12 = (__int64 *)(v53 + 16LL * v55);
        v57 = *v12;
        if ( *v12 != a2 )
        {
          while ( v57 != -4096 )
          {
            if ( v57 == -8192 && !v56 )
              v56 = v12;
            v55 = v52 & (v54 + v55);
            v12 = (__int64 *)(v53 + 16LL * v55);
            v57 = *v12;
            if ( *v12 == a2 )
              goto LABEL_17;
            ++v54;
          }
          if ( v56 )
            v12 = v56;
        }
        goto LABEL_17;
      }
LABEL_81:
      ++*(_DWORD *)(a1 + 80);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 64);
  }
  sub_30132A0(v4, 2 * v9);
  v37 = *(_DWORD *)(a1 + 88);
  if ( !v37 )
    goto LABEL_81;
  v38 = v37 - 1;
  v39 = *(_QWORD *)(a1 + 72);
  v40 = (v37 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v27 = *(_DWORD *)(a1 + 80) + 1;
  v12 = (__int64 *)(v39 + 16LL * v40);
  v41 = *v12;
  if ( *v12 != a2 )
  {
    v42 = 1;
    v43 = 0;
    while ( v41 != -4096 )
    {
      if ( !v43 && v41 == -8192 )
        v43 = v12;
      v40 = v38 & (v42 + v40);
      v12 = (__int64 *)(v39 + 16LL * v40);
      v41 = *v12;
      if ( *v12 == a2 )
        goto LABEL_17;
      ++v42;
    }
    if ( v43 )
      v12 = v43;
  }
LABEL_17:
  *(_DWORD *)(a1 + 80) = v27;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v12 = a2;
  v17 = 0;
  v18 = a1 + 96;
  *((_DWORD *)v12 + 2) = 0;
  v16 = *(_DWORD *)(a1 + 120);
  if ( !v16 )
    goto LABEL_20;
LABEL_4:
  v19 = *(_QWORD *)(a1 + 104);
  v20 = 1;
  v21 = 0;
  v22 = (v16 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v23 = (_QWORD *)(v19 + 24LL * v22);
  v24 = *v23;
  if ( *v23 == a3 )
  {
LABEL_5:
    result = v23 + 1;
    goto LABEL_6;
  }
  while ( v24 != -4096 )
  {
    if ( !v21 && v24 == -8192 )
      v21 = v23;
    v22 = (v16 - 1) & (v20 + v22);
    v23 = (_QWORD *)(v19 + 24LL * v22);
    v24 = *v23;
    if ( *v23 == a3 )
      goto LABEL_5;
    ++v20;
  }
  if ( !v21 )
    v21 = v23;
  v36 = *(_DWORD *)(a1 + 112);
  ++*(_QWORD *)(a1 + 96);
  v32 = v36 + 1;
  if ( 4 * (v36 + 1) >= 3 * v16 )
  {
LABEL_21:
    sub_3017690(v18, 2 * v16);
    v28 = *(_DWORD *)(a1 + 120);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 104);
      v31 = (v28 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v32 = *(_DWORD *)(a1 + 112) + 1;
      v21 = (_QWORD *)(v30 + 24LL * v31);
      v33 = *v21;
      if ( *v21 != a3 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v21;
          v31 = v29 & (v34 + v31);
          v21 = (_QWORD *)(v30 + 24LL * v31);
          v33 = *v21;
          if ( *v21 == a3 )
            goto LABEL_38;
          ++v34;
        }
        if ( v35 )
          v21 = v35;
      }
      goto LABEL_38;
    }
    goto LABEL_82;
  }
  if ( v16 - *(_DWORD *)(a1 + 116) - v32 <= v16 >> 3 )
  {
    sub_3017690(v18, v16);
    v44 = *(_DWORD *)(a1 + 120);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 104);
      v47 = 1;
      v48 = v45 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v49 = 0;
      v32 = *(_DWORD *)(a1 + 112) + 1;
      v21 = (_QWORD *)(v46 + 24LL * v48);
      v50 = *v21;
      if ( *v21 != a3 )
      {
        while ( v50 != -4096 )
        {
          if ( v50 == -8192 && !v49 )
            v49 = v21;
          v48 = v45 & (v47 + v48);
          v21 = (_QWORD *)(v46 + 24LL * v48);
          v50 = *v21;
          if ( *v21 == a3 )
            goto LABEL_38;
          ++v47;
        }
        if ( v49 )
          v21 = v49;
      }
      goto LABEL_38;
    }
LABEL_82:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
LABEL_38:
  *(_DWORD *)(a1 + 112) = v32;
  if ( *v21 != -4096 )
    --*(_DWORD *)(a1 + 116);
  *v21 = a3;
  result = v21 + 1;
  *((_DWORD *)v21 + 2) = 0;
  v21[2] = 0;
LABEL_6:
  *(_DWORD *)result = v17;
  result[1] = a4;
  return result;
}
