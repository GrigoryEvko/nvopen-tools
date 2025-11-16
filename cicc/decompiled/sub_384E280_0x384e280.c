// Function: sub_384E280
// Address: 0x384e280
//
__int64 __fastcall sub_384E280(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rbx
  unsigned int v9; // esi
  __int64 v10; // r9
  unsigned int v11; // ecx
  __int64 v12; // rdx
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // r11
  int v16; // r15d
  unsigned int v17; // r8d
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 result; // rax
  int v21; // ecx
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 v24; // rdi
  int v25; // r15d
  __int64 *v26; // r8
  int v27; // eax
  int v28; // edx
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // ecx
  int v33; // edx
  __int64 v34; // r8
  int v35; // r10d
  __int64 *v36; // r9
  int v37; // r11d
  __int64 *v38; // r10
  int v39; // ecx
  unsigned int v40; // r8d
  int v41; // eax
  int v42; // ecx
  __int64 v43; // rsi
  unsigned int v44; // eax
  __int64 v45; // rdi
  int v46; // r11d
  __int64 *v47; // r10
  int v48; // eax
  int v49; // eax
  __int64 v50; // rsi
  int v51; // r10d
  unsigned int v52; // r14d
  __int64 *v53; // rdi
  __int64 v54; // rcx
  int v55; // eax
  int v56; // ecx
  __int64 v57; // rdi
  int v58; // r9d
  unsigned int v59; // r14d
  __int64 *v60; // r8
  __int64 v61; // rsi

  v5 = *(_QWORD **)(a1 + 16);
  v6 = v5;
  if ( a2 != *v5 )
  {
    LODWORD(v7) = 0;
    do
    {
      v7 = (unsigned int)(v7 + 1);
      v6 = &v5[v7];
    }
    while ( *v6 != a2 );
  }
  *v6 = a3;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(_DWORD *)(v8 + 32);
  v10 = v8 + 8;
  if ( !v9 )
  {
    ++*(_QWORD *)(v8 + 8);
    goto LABEL_44;
  }
  v11 = v9 - 1;
  v12 = *(_QWORD *)(v8 + 16);
  v13 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( a2 == *v14 )
  {
LABEL_6:
    v16 = *((_DWORD *)v14 + 2);
    goto LABEL_7;
  }
  v25 = 1;
  v26 = 0;
  while ( v15 != -8 )
  {
    if ( v15 == -16 && !v26 )
      v26 = v14;
    v13 = v11 & (v25 + v13);
    v14 = (__int64 *)(v12 + 16LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
      goto LABEL_6;
    ++v25;
  }
  if ( !v26 )
    v26 = v14;
  v27 = *(_DWORD *)(v8 + 24);
  ++*(_QWORD *)(v8 + 8);
  v28 = v27 + 1;
  if ( 4 * (v27 + 1) >= 3 * v9 )
  {
LABEL_44:
    sub_13C67E0(v8 + 8, 2 * v9);
    v41 = *(_DWORD *)(v8 + 32);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(v8 + 16);
      v10 = v8 + 8;
      v44 = (v41 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = *(_DWORD *)(v8 + 24) + 1;
      v26 = (__int64 *)(v43 + 16LL * v44);
      v45 = *v26;
      if ( a2 != *v26 )
      {
        v46 = 1;
        v47 = 0;
        while ( v45 != -8 )
        {
          if ( v45 == -16 && !v47 )
            v47 = v26;
          v44 = v42 & (v46 + v44);
          v26 = (__int64 *)(v43 + 16LL * v44);
          v45 = *v26;
          if ( a2 == *v26 )
            goto LABEL_18;
          ++v46;
        }
        if ( v47 )
          v26 = v47;
      }
      goto LABEL_18;
    }
    goto LABEL_93;
  }
  if ( v9 - *(_DWORD *)(v8 + 28) - v28 <= v9 >> 3 )
  {
    sub_13C67E0(v8 + 8, v9);
    v48 = *(_DWORD *)(v8 + 32);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(v8 + 16);
      v51 = 1;
      v52 = v49 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = v8 + 8;
      v28 = *(_DWORD *)(v8 + 24) + 1;
      v53 = 0;
      v26 = (__int64 *)(v50 + 16LL * v52);
      v54 = *v26;
      if ( a2 != *v26 )
      {
        while ( v54 != -8 )
        {
          if ( !v53 && v54 == -16 )
            v53 = v26;
          v52 = v49 & (v51 + v52);
          v26 = (__int64 *)(v50 + 16LL * v52);
          v54 = *v26;
          if ( a2 == *v26 )
            goto LABEL_18;
          ++v51;
        }
        if ( v53 )
          v26 = v53;
      }
      goto LABEL_18;
    }
LABEL_93:
    ++*(_DWORD *)(v8 + 24);
    BUG();
  }
LABEL_18:
  *(_DWORD *)(v8 + 24) = v28;
  if ( *v26 != -8 )
    --*(_DWORD *)(v8 + 28);
  *v26 = a2;
  *((_DWORD *)v26 + 2) = 0;
  v9 = *(_DWORD *)(v8 + 32);
  if ( !v9 )
  {
    ++*(_QWORD *)(v8 + 8);
    v16 = 0;
    goto LABEL_22;
  }
  v12 = *(_QWORD *)(v8 + 16);
  v11 = v9 - 1;
  v16 = 0;
LABEL_7:
  v17 = v11 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = (__int64 *)(v12 + 16LL * v17);
  v19 = *v18;
  if ( a3 == *v18 )
    goto LABEL_8;
  v37 = 1;
  v38 = 0;
  while ( v19 != -8 )
  {
    if ( v19 == -16 && !v38 )
      v38 = v18;
    v17 = v11 & (v37 + v17);
    v18 = (__int64 *)(v12 + 16LL * v17);
    v19 = *v18;
    if ( a3 == *v18 )
      goto LABEL_8;
    ++v37;
  }
  v39 = *(_DWORD *)(v8 + 24);
  if ( v38 )
    v18 = v38;
  ++*(_QWORD *)(v8 + 8);
  v33 = v39 + 1;
  if ( 4 * (v39 + 1) >= 3 * v9 )
  {
LABEL_22:
    sub_13C67E0(v10, 2 * v9);
    v29 = *(_DWORD *)(v8 + 32);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v8 + 16);
      v32 = (v29 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v33 = *(_DWORD *)(v8 + 24) + 1;
      v18 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v18;
      if ( a3 != *v18 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -8 )
        {
          if ( v34 == -16 && !v36 )
            v36 = v18;
          v32 = v30 & (v35 + v32);
          v18 = (__int64 *)(v31 + 16LL * v32);
          v34 = *v18;
          if ( a3 == *v18 )
            goto LABEL_36;
          ++v35;
        }
        if ( v36 )
          v18 = v36;
      }
      goto LABEL_36;
    }
    goto LABEL_94;
  }
  if ( v9 - (v33 + *(_DWORD *)(v8 + 28)) <= v9 >> 3 )
  {
    sub_13C67E0(v10, v9);
    v55 = *(_DWORD *)(v8 + 32);
    if ( v55 )
    {
      v56 = v55 - 1;
      v57 = *(_QWORD *)(v8 + 16);
      v58 = 1;
      v59 = (v55 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v60 = 0;
      v33 = *(_DWORD *)(v8 + 24) + 1;
      v18 = (__int64 *)(v57 + 16LL * v59);
      v61 = *v18;
      if ( a3 != *v18 )
      {
        while ( v61 != -8 )
        {
          if ( !v60 && v61 == -16 )
            v60 = v18;
          v59 = v56 & (v58 + v59);
          v18 = (__int64 *)(v57 + 16LL * v59);
          v61 = *v18;
          if ( a3 == *v18 )
            goto LABEL_36;
          ++v58;
        }
        if ( v60 )
          v18 = v60;
      }
      goto LABEL_36;
    }
LABEL_94:
    ++*(_DWORD *)(v8 + 24);
    BUG();
  }
LABEL_36:
  *(_DWORD *)(v8 + 24) = v33;
  if ( *v18 != -8 )
    --*(_DWORD *)(v8 + 28);
  *v18 = a3;
  *((_DWORD *)v18 + 2) = 0;
LABEL_8:
  *((_DWORD *)v18 + 2) = v16;
  result = *(unsigned int *)(v8 + 32);
  if ( (_DWORD)result )
  {
    v21 = result - 1;
    v22 = *(_QWORD *)(v8 + 16);
    v23 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v22 + 16LL * v23;
    v24 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
    {
LABEL_10:
      *(_QWORD *)result = -16;
      --*(_DWORD *)(v8 + 24);
      ++*(_DWORD *)(v8 + 28);
    }
    else
    {
      result = 1;
      while ( v24 != -8 )
      {
        v40 = result + 1;
        v23 = v21 & (result + v23);
        result = v22 + 16LL * v23;
        v24 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_10;
        result = v40;
      }
    }
  }
  return result;
}
