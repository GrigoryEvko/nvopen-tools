// Function: sub_DF47E0
// Address: 0xdf47e0
//
__int64 __fastcall sub_DF47E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r12
  char v8; // r9
  __int64 v9; // rcx
  int v10; // r10d
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r13d
  unsigned int i; // eax
  _QWORD *v22; // r9
  unsigned int v23; // eax
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // r9
  __int64 (__fastcall *v27)(__int64); // rax
  char v28; // si
  __int64 v29; // rdi
  int v30; // r8d
  unsigned int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // r10
  unsigned int v34; // r8d
  int v35; // r11d
  __int64 v36; // r9
  unsigned int v37; // edx
  int v38; // ecx
  unsigned int v39; // esi
  int v40; // r13d
  __int64 v41; // rdi
  int v42; // ecx
  unsigned int v43; // edx
  __int64 v44; // rsi
  __int64 v45; // rdi
  int v46; // ecx
  unsigned int v47; // edx
  __int64 v48; // rsi
  int v49; // r10d
  __int64 v50; // r8
  int v51; // ecx
  int v52; // ecx
  int v53; // r10d
  unsigned __int8 v54; // [rsp+Fh] [rbp-21h]
  unsigned __int8 v55; // [rsp+Fh] [rbp-21h]

  v7 = *a1;
  v8 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v8 )
  {
    v9 = v7 + 16;
    v10 = 7;
  }
  else
  {
    v16 = *(unsigned int *)(v7 + 24);
    v9 = *(_QWORD *)(v7 + 16);
    if ( !(_DWORD)v16 )
      goto LABEL_16;
    v10 = v16 - 1;
  }
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v9 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( a2 == *(_QWORD *)v12 )
    goto LABEL_4;
  v25 = 1;
  while ( v13 != -4096 )
  {
    v40 = v25 + 1;
    v11 = v10 & (v25 + v11);
    v12 = v9 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( a2 == *(_QWORD *)v12 )
      goto LABEL_4;
    v25 = v40;
  }
  if ( v8 )
  {
    v24 = 128;
    goto LABEL_17;
  }
  v16 = *(unsigned int *)(v7 + 24);
LABEL_16:
  v24 = 16 * v16;
LABEL_17:
  v12 = v9 + v24;
LABEL_4:
  v14 = 128;
  if ( !v8 )
    v14 = 16LL * *(unsigned int *)(v7 + 24);
  if ( v12 != v9 + v14 )
    return *(unsigned __int8 *)(v12 + 8);
  v17 = a1[1];
  v18 = *(unsigned int *)(v17 + 24);
  v19 = *(_QWORD *)(v17 + 8);
  if ( (_DWORD)v18 )
  {
    v20 = 1;
    for ( i = (v18 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v18 - 1) & v23 )
    {
      v22 = (_QWORD *)(v19 + 24LL * i);
      if ( a2 == *v22 && a3 == v22[1] )
        break;
      if ( *v22 == -4096 && v22[1] == -4096 )
        goto LABEL_23;
      v23 = v20 + i;
      ++v20;
    }
  }
  else
  {
LABEL_23:
    v22 = (_QWORD *)(v19 + 24 * v18);
  }
  v26 = *(_QWORD *)(v22[2] + 24LL);
  v27 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v26 + 16LL);
  if ( v27 == sub_D32150 )
    result = sub_DF3010(v26 + 8, a3, a4, a1);
  else
    result = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v27)(v26, a3, a4, a1);
  v28 = *(_BYTE *)(v7 + 8) & 1;
  if ( v28 )
  {
    v29 = v7 + 16;
    v30 = 7;
    v31 = ((unsigned __int8)((unsigned int)a2 >> 9) ^ (unsigned __int8)((unsigned int)a2 >> 4)) & 7;
    v32 = v7 + 16 + 16LL * (((unsigned __int8)((unsigned int)a2 >> 9) ^ (unsigned __int8)((unsigned int)a2 >> 4)) & 7);
    v33 = *(_QWORD *)v32;
    if ( a2 == *(_QWORD *)v32 )
      return *(unsigned __int8 *)(v32 + 8);
  }
  else
  {
    v34 = *(_DWORD *)(v7 + 24);
    if ( !v34 )
    {
      v37 = *(_DWORD *)(v7 + 8);
      ++*(_QWORD *)v7;
      v36 = 0;
      v38 = (v37 >> 1) + 1;
      goto LABEL_40;
    }
    v30 = v34 - 1;
    v29 = *(_QWORD *)(v7 + 16);
    v31 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v32 = v29 + 16LL * v31;
    v33 = *(_QWORD *)v32;
    if ( a2 == *(_QWORD *)v32 )
      return *(unsigned __int8 *)(v32 + 8);
  }
  v35 = 1;
  v36 = 0;
  while ( v33 != -4096 )
  {
    if ( v33 == -8192 && !v36 )
      v36 = v32;
    v31 = v30 & (v35 + v31);
    v32 = v29 + 16LL * v31;
    v33 = *(_QWORD *)v32;
    if ( a2 == *(_QWORD *)v32 )
      return *(unsigned __int8 *)(v32 + 8);
    ++v35;
  }
  if ( !v36 )
    v36 = v32;
  v37 = *(_DWORD *)(v7 + 8);
  ++*(_QWORD *)v7;
  v38 = (v37 >> 1) + 1;
  if ( v28 )
  {
    v39 = 24;
    v34 = 8;
    goto LABEL_41;
  }
  v34 = *(_DWORD *)(v7 + 24);
LABEL_40:
  v39 = 3 * v34;
LABEL_41:
  if ( 4 * v38 >= v39 )
  {
    v54 = result;
    sub_BBCB10(v7, 2 * v34);
    result = v54;
    if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
    {
      v41 = v7 + 16;
      v42 = 7;
    }
    else
    {
      v51 = *(_DWORD *)(v7 + 24);
      v41 = *(_QWORD *)(v7 + 16);
      if ( !v51 )
        goto LABEL_82;
      v42 = v51 - 1;
    }
    v43 = v42 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v36 = v41 + 16LL * v43;
    v44 = *(_QWORD *)v36;
    if ( a2 != *(_QWORD *)v36 )
    {
      v53 = 1;
      v50 = 0;
      while ( v44 != -4096 )
      {
        if ( !v50 && v44 == -8192 )
          v50 = v36;
        v43 = v42 & (v53 + v43);
        v36 = v41 + 16LL * v43;
        v44 = *(_QWORD *)v36;
        if ( a2 == *(_QWORD *)v36 )
          goto LABEL_52;
        ++v53;
      }
      goto LABEL_58;
    }
LABEL_52:
    v37 = *(_DWORD *)(v7 + 8);
    goto LABEL_43;
  }
  if ( v34 - *(_DWORD *)(v7 + 12) - v38 <= v34 >> 3 )
  {
    v55 = result;
    sub_BBCB10(v7, v34);
    result = v55;
    if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
    {
      v45 = v7 + 16;
      v46 = 7;
      goto LABEL_55;
    }
    v52 = *(_DWORD *)(v7 + 24);
    v45 = *(_QWORD *)(v7 + 16);
    if ( v52 )
    {
      v46 = v52 - 1;
LABEL_55:
      v47 = v46 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = v45 + 16LL * v47;
      v48 = *(_QWORD *)v36;
      if ( a2 != *(_QWORD *)v36 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -4096 )
        {
          if ( v48 == -8192 && !v50 )
            v50 = v36;
          v47 = v46 & (v49 + v47);
          v36 = v45 + 16LL * v47;
          v48 = *(_QWORD *)v36;
          if ( a2 == *(_QWORD *)v36 )
            goto LABEL_52;
          ++v49;
        }
LABEL_58:
        if ( v50 )
          v36 = v50;
        goto LABEL_52;
      }
      goto LABEL_52;
    }
LABEL_82:
    *(_DWORD *)(v7 + 8) = (2 * (*(_DWORD *)(v7 + 8) >> 1) + 2) | *(_DWORD *)(v7 + 8) & 1;
    BUG();
  }
LABEL_43:
  *(_DWORD *)(v7 + 8) = (2 * (v37 >> 1) + 2) | v37 & 1;
  if ( *(_QWORD *)v36 != -4096 )
    --*(_DWORD *)(v7 + 12);
  *(_QWORD *)v36 = a2;
  *(_BYTE *)(v36 + 8) = result;
  return result;
}
