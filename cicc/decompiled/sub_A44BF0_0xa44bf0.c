// Function: sub_A44BF0
// Address: 0xa44bf0
//
unsigned __int64 __fastcall sub_A44BF0(__int64 a1, char *a2)
{
  __int64 v2; // r13
  unsigned int v4; // r8d
  __int64 v5; // r9
  char *v6; // rdi
  int v7; // r14d
  char **v8; // rcx
  __int64 v9; // rdx
  __int64 *v10; // rax
  char *v11; // r11
  _DWORD *v12; // rcx
  unsigned __int64 result; // rax
  int v14; // eax
  __int64 *v15; // r12
  __int64 *v16; // r14
  __int64 v17; // rsi
  unsigned int v18; // esi
  char *v19; // rdi
  __int64 v20; // r8
  char **v21; // rdx
  int v22; // r11d
  unsigned int v23; // ecx
  __int64 *v24; // rax
  char *v25; // r10
  _DWORD *v26; // r12
  _BYTE *v27; // rsi
  _BYTE *v28; // rsi
  int v29; // eax
  int v30; // esi
  __int64 v31; // r9
  int v32; // ecx
  char *v33; // r8
  int v34; // eax
  int v35; // eax
  int v36; // esi
  __int64 v37; // r9
  char **v38; // r10
  int v39; // r11d
  char *v40; // r8
  int v41; // eax
  int v42; // esi
  __int64 v43; // r9
  unsigned int v44; // eax
  int v45; // r11d
  char **v46; // r10
  int v47; // eax
  int v48; // esi
  __int64 v49; // r9
  char **v50; // r10
  int v51; // r11d
  unsigned int v52; // eax
  char *v53; // r8
  int v54; // r11d
  char *v55; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 24;
  v4 = *(_DWORD *)(a1 + 48);
  v55 = a2;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_55;
  }
  v5 = *(_QWORD *)(a1 + 32);
  v6 = a2;
  v7 = 1;
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v5 + 16 * v9);
  v11 = (char *)*v10;
  if ( a2 != (char *)*v10 )
  {
    while ( v11 != (char *)-4096LL )
    {
      if ( !v8 && v11 == (char *)-8192LL )
        v8 = (char **)v10;
      v9 = (v4 - 1) & (v7 + (_DWORD)v9);
      v10 = (__int64 *)(v5 + 16LL * (unsigned int)v9);
      v11 = (char *)*v10;
      if ( a2 == (char *)*v10 )
        goto LABEL_3;
      ++v7;
    }
    if ( !v8 )
      v8 = (char **)v10;
    v14 = *(_DWORD *)(a1 + 40);
    ++*(_QWORD *)(a1 + 24);
    v9 = (unsigned int)(v14 + 1);
    if ( 4 * (int)v9 < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 44) - (unsigned int)v9 > v4 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 40) = v9;
        if ( *v8 != (char *)-4096LL )
          --*(_DWORD *)(a1 + 44);
        *v8 = v6;
        a2 = v55;
        v12 = v8 + 1;
        *v12 = 0;
        goto LABEL_18;
      }
      sub_A44A10(v2, v4);
      v47 = *(_DWORD *)(a1 + 48);
      if ( v47 )
      {
        v6 = v55;
        v48 = v47 - 1;
        v49 = *(_QWORD *)(a1 + 32);
        v50 = 0;
        v51 = 1;
        v52 = (v47 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
        v9 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
        v8 = (char **)(v49 + 16LL * v52);
        v53 = *v8;
        if ( *v8 != v55 )
        {
          while ( v53 != (char *)-4096LL )
          {
            if ( v53 == (char *)-8192LL && !v50 )
              v50 = v8;
            v52 = v48 & (v51 + v52);
            v8 = (char **)(v49 + 16LL * v52);
            v53 = *v8;
            if ( v55 == *v8 )
              goto LABEL_15;
            ++v51;
          }
          if ( v50 )
            v8 = v50;
        }
        goto LABEL_15;
      }
LABEL_90:
      ++*(_DWORD *)(a1 + 40);
      BUG();
    }
LABEL_55:
    sub_A44A10(v2, 2 * v4);
    v41 = *(_DWORD *)(a1 + 48);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(a1 + 32);
      v44 = (v41 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
      v9 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
      v8 = (char **)(v43 + 16LL * v44);
      v6 = *v8;
      if ( v55 != *v8 )
      {
        v45 = 1;
        v46 = 0;
        while ( v6 != (char *)-4096LL )
        {
          if ( !v46 && v6 == (char *)-8192LL )
            v46 = v8;
          v44 = v42 & (v45 + v44);
          v8 = (char **)(v43 + 16LL * v44);
          v6 = *v8;
          if ( v55 == *v8 )
            goto LABEL_15;
          ++v45;
        }
        v6 = v55;
        if ( v46 )
          v8 = v46;
      }
      goto LABEL_15;
    }
    goto LABEL_90;
  }
LABEL_3:
  v12 = v10 + 1;
  result = *((unsigned int *)v10 + 2);
  if ( (_DWORD)result )
    return result;
LABEL_18:
  if ( a2[8] == 15 && (a2[9] & 4) == 0 )
  {
    *v12 = -1;
    a2 = v55;
  }
  v15 = (__int64 *)*((_QWORD *)a2 + 2);
  v16 = &v15[*((unsigned int *)a2 + 3)];
  while ( v16 != v15 )
  {
    v17 = *v15++;
    sub_A44BF0(a1, v17, v9, v12);
  }
  v18 = *(_DWORD *)(a1 + 48);
  if ( v18 )
  {
    v19 = v55;
    v20 = *(_QWORD *)(a1 + 32);
    v21 = 0;
    v22 = 1;
    v23 = (v18 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
    v24 = (__int64 *)(v20 + 16LL * v23);
    v25 = (char *)*v24;
    if ( v55 == (char *)*v24 )
    {
LABEL_25:
      v26 = v24 + 1;
      result = (unsigned int)(*((_DWORD *)v24 + 2) - 1);
      if ( (unsigned int)result <= 0xFFFFFFFD )
        return result;
      goto LABEL_26;
    }
    while ( v25 != (char *)-4096LL )
    {
      if ( !v21 && v25 == (char *)-8192LL )
        v21 = (char **)v24;
      v23 = (v18 - 1) & (v22 + v23);
      v24 = (__int64 *)(v20 + 16LL * v23);
      v25 = (char *)*v24;
      if ( v55 == (char *)*v24 )
        goto LABEL_25;
      ++v22;
    }
    if ( !v21 )
      v21 = (char **)v24;
    v34 = *(_DWORD *)(a1 + 40);
    ++*(_QWORD *)(a1 + 24);
    v32 = v34 + 1;
    if ( 4 * (v34 + 1) < 3 * v18 )
    {
      result = v18 - *(_DWORD *)(a1 + 44) - v32;
      if ( (unsigned int)result > v18 >> 3 )
        goto LABEL_35;
      sub_A44A10(v2, v18);
      v35 = *(_DWORD *)(a1 + 48);
      if ( v35 )
      {
        v19 = v55;
        v36 = v35 - 1;
        v37 = *(_QWORD *)(a1 + 32);
        v38 = 0;
        v39 = 1;
        v32 = *(_DWORD *)(a1 + 40) + 1;
        result = (v35 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
        v21 = (char **)(v37 + 16 * result);
        v40 = *v21;
        if ( *v21 != v55 )
        {
          while ( v40 != (char *)-4096LL )
          {
            if ( v40 == (char *)-8192LL && !v38 )
              v38 = v21;
            result = v36 & (unsigned int)(v39 + result);
            v21 = (char **)(v37 + 16LL * (unsigned int)result);
            v40 = *v21;
            if ( v55 == *v21 )
              goto LABEL_35;
            ++v39;
          }
LABEL_51:
          if ( v38 )
            v21 = v38;
          goto LABEL_35;
        }
        goto LABEL_35;
      }
LABEL_91:
      ++*(_DWORD *)(a1 + 40);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  sub_A44A10(v2, 2 * v18);
  v29 = *(_DWORD *)(a1 + 48);
  if ( !v29 )
    goto LABEL_91;
  v19 = v55;
  v30 = v29 - 1;
  v31 = *(_QWORD *)(a1 + 32);
  v32 = *(_DWORD *)(a1 + 40) + 1;
  result = (v29 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
  v21 = (char **)(v31 + 16 * result);
  v33 = *v21;
  if ( *v21 != v55 )
  {
    v54 = 1;
    v38 = 0;
    while ( v33 != (char *)-4096LL )
    {
      if ( !v38 && v33 == (char *)-8192LL )
        v38 = v21;
      result = v30 & (unsigned int)(v54 + result);
      v21 = (char **)(v31 + 16LL * (unsigned int)result);
      v33 = *v21;
      if ( v55 == *v21 )
        goto LABEL_35;
      ++v54;
    }
    goto LABEL_51;
  }
LABEL_35:
  *(_DWORD *)(a1 + 40) = v32;
  if ( *v21 != (char *)-4096LL )
    --*(_DWORD *)(a1 + 44);
  *v21 = v19;
  v26 = v21 + 1;
  *((_DWORD *)v21 + 2) = 0;
LABEL_26:
  v27 = *(_BYTE **)(a1 + 64);
  if ( v27 == *(_BYTE **)(a1 + 72) )
  {
    result = (unsigned __int64)sub_918210(a1 + 56, v27, &v55);
    v28 = *(_BYTE **)(a1 + 64);
  }
  else
  {
    if ( v27 )
    {
      result = (unsigned __int64)v55;
      *(_QWORD *)v27 = v55;
      v27 = *(_BYTE **)(a1 + 64);
    }
    v28 = v27 + 8;
    *(_QWORD *)(a1 + 64) = v28;
  }
  *v26 = (__int64)&v28[-*(_QWORD *)(a1 + 56)] >> 3;
  return result;
}
