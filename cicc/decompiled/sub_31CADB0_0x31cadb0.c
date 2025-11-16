// Function: sub_31CADB0
// Address: 0x31cadb0
//
__int64 __fastcall sub_31CADB0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  unsigned int v14; // esi
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 result; // rax
  int v19; // r10d
  __int64 *v20; // r9
  int v21; // eax
  int v22; // edx
  int v23; // r10d
  _QWORD *v24; // r9
  int v25; // eax
  int v26; // edx
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  __int64 v30; // rsi
  int v31; // r10d
  __int64 *v32; // r8
  int v33; // eax
  __int64 v34; // rsi
  int v35; // r8d
  __int64 *v36; // rdi
  unsigned int v37; // r13d
  __int64 v38; // rcx
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rdi
  unsigned int v42; // eax
  __int64 v43; // rsi
  int v44; // r10d
  _QWORD *v45; // r8
  int v46; // eax
  int v47; // eax
  __int64 v48; // rsi
  int v49; // r8d
  _QWORD *v50; // rdi
  unsigned int v51; // r14d
  __int64 v52; // rcx

  v1 = a1[2];
  if ( !v1 )
    goto LABEL_5;
  v3 = *(_QWORD *)(v1 + 24);
  if ( !v3 )
    goto LABEL_5;
  v4 = a1[165];
  v5 = *(_DWORD *)(v4 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
    goto LABEL_41;
  }
  v6 = *(_QWORD *)(v4 + 8);
  v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = (_QWORD *)(v6 + 8LL * v7);
  v9 = *v8;
  if ( v3 == *v8 )
    goto LABEL_5;
  v23 = 1;
  v24 = 0;
  while ( v9 != -4096 )
  {
    if ( v24 || v9 != -8192 )
      v8 = v24;
    v7 = (v5 - 1) & (v23 + v7);
    v9 = *(_QWORD *)(v6 + 8LL * v7);
    if ( v3 == v9 )
      goto LABEL_5;
    ++v23;
    v24 = v8;
    v8 = (_QWORD *)(v6 + 8LL * v7);
  }
  v25 = *(_DWORD *)(v4 + 16);
  if ( !v24 )
    v24 = v8;
  ++*(_QWORD *)v4;
  v26 = v25 + 1;
  if ( 4 * (v25 + 1) >= 3 * v5 )
  {
LABEL_41:
    sub_CF4090(v4, 2 * v5);
    v39 = *(_DWORD *)(v4 + 24);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(v4 + 8);
      v42 = (v39 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v24 = (_QWORD *)(v41 + 8LL * v42);
      v43 = *v24;
      v26 = *(_DWORD *)(v4 + 16) + 1;
      if ( v3 != *v24 )
      {
        v44 = 1;
        v45 = 0;
        while ( v43 != -4096 )
        {
          if ( v43 == -8192 && !v45 )
            v45 = v24;
          v42 = v40 & (v44 + v42);
          v24 = (_QWORD *)(v41 + 8LL * v42);
          v43 = *v24;
          if ( v3 == *v24 )
            goto LABEL_23;
          ++v44;
        }
        if ( v45 )
          v24 = v45;
      }
      goto LABEL_23;
    }
    goto LABEL_85;
  }
  if ( v5 - *(_DWORD *)(v4 + 20) - v26 <= v5 >> 3 )
  {
    sub_CF4090(v4, v5);
    v46 = *(_DWORD *)(v4 + 24);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(v4 + 8);
      v49 = 1;
      v50 = 0;
      v51 = v47 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v24 = (_QWORD *)(v48 + 8LL * v51);
      v52 = *v24;
      v26 = *(_DWORD *)(v4 + 16) + 1;
      if ( v3 != *v24 )
      {
        while ( v52 != -4096 )
        {
          if ( v52 == -8192 && !v50 )
            v50 = v24;
          v51 = v47 & (v49 + v51);
          v24 = (_QWORD *)(v48 + 8LL * v51);
          v52 = *v24;
          if ( v3 == *v24 )
            goto LABEL_23;
          ++v49;
        }
        if ( v50 )
          v24 = v50;
      }
      goto LABEL_23;
    }
LABEL_85:
    ++*(_DWORD *)(v4 + 16);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(v4 + 16) = v26;
  if ( *v24 != -4096 )
    --*(_DWORD *)(v4 + 20);
  *v24 = v3;
LABEL_5:
  v10 = *a1;
  v11 = sub_ACA8A0(*(__int64 ***)(*a1 + 8));
  sub_BD84D0(v10, v11);
  v12 = *a1;
  v13 = a1[165];
  v14 = *(_DWORD *)(v13 + 24);
  if ( !v14 )
  {
    ++*(_QWORD *)v13;
    goto LABEL_27;
  }
  v15 = *(_QWORD *)(v13 + 8);
  v16 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v17 = (__int64 *)(v15 + 8LL * v16);
  result = *v17;
  if ( v12 == *v17 )
    return result;
  v19 = 1;
  v20 = 0;
  while ( result != -4096 )
  {
    if ( v20 || result != -8192 )
      v17 = v20;
    v16 = (v14 - 1) & (v19 + v16);
    result = *(_QWORD *)(v15 + 8LL * v16);
    if ( v12 == result )
      return result;
    ++v19;
    v20 = v17;
    v17 = (__int64 *)(v15 + 8LL * v16);
  }
  v21 = *(_DWORD *)(v13 + 16);
  if ( !v20 )
    v20 = v17;
  ++*(_QWORD *)v13;
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v14 )
  {
LABEL_27:
    sub_CF4090(v13, 2 * v14);
    v27 = *(_DWORD *)(v13 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v13 + 8);
      result = (v27 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v20 = (__int64 *)(v29 + 8 * result);
      v30 = *v20;
      v22 = *(_DWORD *)(v13 + 16) + 1;
      if ( v12 != *v20 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( !v32 && v30 == -8192 )
            v32 = v20;
          result = v28 & (unsigned int)(v31 + result);
          v20 = (__int64 *)(v29 + 8LL * (unsigned int)result);
          v30 = *v20;
          if ( v12 == *v20 )
            goto LABEL_14;
          ++v31;
        }
        if ( v32 )
          v20 = v32;
      }
      goto LABEL_14;
    }
    goto LABEL_84;
  }
  result = v14 - *(_DWORD *)(v13 + 20) - v22;
  if ( (unsigned int)result <= v14 >> 3 )
  {
    sub_CF4090(v13, v14);
    v33 = *(_DWORD *)(v13 + 24);
    if ( v33 )
    {
      result = (unsigned int)(v33 - 1);
      v34 = *(_QWORD *)(v13 + 8);
      v35 = 1;
      v36 = 0;
      v37 = result & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v20 = (__int64 *)(v34 + 8LL * v37);
      v38 = *v20;
      v22 = *(_DWORD *)(v13 + 16) + 1;
      if ( v12 != *v20 )
      {
        while ( v38 != -4096 )
        {
          if ( v38 == -8192 && !v36 )
            v36 = v20;
          v37 = result & (v35 + v37);
          v20 = (__int64 *)(v34 + 8LL * v37);
          v38 = *v20;
          if ( v12 == *v20 )
            goto LABEL_14;
          ++v35;
        }
        if ( v36 )
          v20 = v36;
      }
      goto LABEL_14;
    }
LABEL_84:
    ++*(_DWORD *)(v13 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v13 + 16) = v22;
  if ( *v20 != -4096 )
    --*(_DWORD *)(v13 + 20);
  *v20 = v12;
  return result;
}
