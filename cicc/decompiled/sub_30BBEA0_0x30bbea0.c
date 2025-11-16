// Function: sub_30BBEA0
// Address: 0x30bbea0
//
bool __fastcall sub_30BBEA0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // edi
  __int64 v10; // rcx
  int v11; // r11d
  _QWORD *v12; // rdx
  unsigned int v13; // r10d
  _QWORD *v14; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // r9
  int v17; // r15d
  _QWORD *v18; // r10
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // r14
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // eax
  int v29; // edx
  __int64 v30; // rsi
  int v31; // r9d
  _QWORD *v32; // r8
  int v33; // eax
  int v34; // eax
  int v35; // esi
  __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // r8
  int v39; // r10d
  _QWORD *v40; // r9
  int v41; // eax
  int v42; // eax
  __int64 v43; // rdi
  int v44; // r9d
  unsigned int v45; // r15d
  _QWORD *v46; // r8
  __int64 v47; // rsi
  int v48; // eax
  int v49; // eax
  __int64 v50; // rsi
  int v51; // r8d
  unsigned int v52; // r12d
  _QWORD *v53; // rdi
  __int64 v54; // rcx

  v6 = *a1;
  v7 = *(_DWORD *)(*a1 + 120);
  v8 = *a1 + 96;
  if ( v7 )
  {
    v9 = v7 - 1;
    v10 = *(_QWORD *)(v6 + 104);
    v11 = 1;
    v12 = 0;
    v13 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = (_QWORD *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
    {
LABEL_3:
      v16 = v14[1];
      goto LABEL_4;
    }
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      v13 = v9 & (v11 + v13);
      v14 = (_QWORD *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
    v23 = *(_DWORD *)(v6 + 112);
    ++*(_QWORD *)(v6 + 96);
    v24 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(v6 + 116) - v24 <= v7 >> 3 )
      {
        sub_30BBCC0(v8, v7);
        v41 = *(_DWORD *)(v6 + 120);
        if ( !v41 )
          goto LABEL_81;
        v42 = v41 - 1;
        v43 = *(_QWORD *)(v6 + 104);
        v44 = 1;
        v45 = v42 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v46 = 0;
        v24 = *(_DWORD *)(v6 + 112) + 1;
        v12 = (_QWORD *)(v43 + 16LL * v45);
        v47 = *v12;
        if ( a2 != *v12 )
        {
          while ( v47 != -4096 )
          {
            if ( v47 == -8192 && !v46 )
              v46 = v12;
            v45 = v42 & (v44 + v45);
            v12 = (_QWORD *)(v43 + 16LL * v45);
            v47 = *v12;
            if ( a2 == *v12 )
              goto LABEL_16;
            ++v44;
          }
          if ( v46 )
            v12 = v46;
        }
      }
      goto LABEL_16;
    }
  }
  else
  {
    ++*(_QWORD *)(v6 + 96);
  }
  sub_30BBCC0(v8, 2 * v7);
  v34 = *(_DWORD *)(v6 + 120);
  if ( !v34 )
    goto LABEL_81;
  v35 = v34 - 1;
  v36 = *(_QWORD *)(v6 + 104);
  v37 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = *(_DWORD *)(v6 + 112) + 1;
  v12 = (_QWORD *)(v36 + 16LL * v37);
  v38 = *v12;
  if ( a2 != *v12 )
  {
    v39 = 1;
    v40 = 0;
    while ( v38 != -4096 )
    {
      if ( !v40 && v38 == -8192 )
        v40 = v12;
      v37 = v35 & (v39 + v37);
      v12 = (_QWORD *)(v36 + 16LL * v37);
      v38 = *v12;
      if ( a2 == *v12 )
        goto LABEL_16;
      ++v39;
    }
    if ( v40 )
      v12 = v40;
  }
LABEL_16:
  *(_DWORD *)(v6 + 112) = v24;
  if ( *v12 != -4096 )
    --*(_DWORD *)(v6 + 116);
  *v12 = a2;
  v12[1] = 0;
  v6 = *a1;
  v7 = *(_DWORD *)(*a1 + 120);
  v8 = *a1 + 96;
  if ( !v7 )
  {
    ++*(_QWORD *)(v6 + 96);
    goto LABEL_20;
  }
  v10 = *(_QWORD *)(v6 + 104);
  v9 = v7 - 1;
  v16 = 0;
LABEL_4:
  v17 = 1;
  v18 = 0;
  v19 = v9 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v20 = (_QWORD *)(v10 + 16LL * v19);
  v21 = *v20;
  if ( a3 == *v20 )
    return v16 < v20[1];
  while ( v21 != -4096 )
  {
    if ( !v18 && v21 == -8192 )
      v18 = v20;
    v19 = v9 & (v17 + v19);
    v20 = (_QWORD *)(v10 + 16LL * v19);
    v21 = *v20;
    if ( a3 == *v20 )
      return v16 < v20[1];
    ++v17;
  }
  if ( !v18 )
    v18 = v20;
  v33 = *(_DWORD *)(v6 + 112);
  ++*(_QWORD *)(v6 + 96);
  v29 = v33 + 1;
  if ( 4 * (v33 + 1) >= 3 * v7 )
  {
LABEL_20:
    sub_30BBCC0(v8, 2 * v7);
    v25 = *(_DWORD *)(v6 + 120);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v6 + 104);
      v28 = (v25 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v29 = *(_DWORD *)(v6 + 112) + 1;
      v18 = (_QWORD *)(v27 + 16LL * v28);
      v30 = *v18;
      if ( a3 != *v18 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( !v32 && v30 == -8192 )
            v32 = v18;
          v28 = v26 & (v31 + v28);
          v18 = (_QWORD *)(v27 + 16LL * v28);
          v30 = *v18;
          if ( a3 == *v18 )
            goto LABEL_37;
          ++v31;
        }
        if ( v32 )
          v18 = v32;
      }
      goto LABEL_37;
    }
    goto LABEL_81;
  }
  if ( v7 - *(_DWORD *)(v6 + 116) - v29 <= v7 >> 3 )
  {
    sub_30BBCC0(v8, v7);
    v48 = *(_DWORD *)(v6 + 120);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(v6 + 104);
      v51 = 1;
      v52 = v49 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v29 = *(_DWORD *)(v6 + 112) + 1;
      v53 = 0;
      v18 = (_QWORD *)(v50 + 16LL * v52);
      v54 = *v18;
      if ( a3 != *v18 )
      {
        while ( v54 != -4096 )
        {
          if ( v54 == -8192 && !v53 )
            v53 = v18;
          v52 = v49 & (v51 + v52);
          v18 = (_QWORD *)(v50 + 16LL * v52);
          v54 = *v18;
          if ( a3 == *v18 )
            goto LABEL_37;
          ++v51;
        }
        if ( v53 )
          v18 = v53;
      }
      goto LABEL_37;
    }
LABEL_81:
    ++*(_DWORD *)(v6 + 112);
    BUG();
  }
LABEL_37:
  *(_DWORD *)(v6 + 112) = v29;
  if ( *v18 != -4096 )
    --*(_DWORD *)(v6 + 116);
  *v18 = a3;
  v18[1] = 0;
  return 0;
}
