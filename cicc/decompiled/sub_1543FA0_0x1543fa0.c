// Function: sub_1543FA0
// Address: 0x1543fa0
//
unsigned __int64 __fastcall sub_1543FA0(__int64 a1, char *a2)
{
  __int64 v2; // r13
  unsigned int v4; // esi
  char *v5; // rcx
  __int64 v6; // r8
  unsigned int v7; // edx
  unsigned __int64 result; // rax
  char *v9; // rdi
  int v10; // r11d
  char *v11; // r10
  int v12; // edi
  int v13; // edi
  __int64 *v14; // r12
  __int64 *v15; // r14
  __int64 v16; // rsi
  unsigned int v17; // esi
  char *v18; // rcx
  __int64 v19; // rdi
  unsigned int v20; // edx
  char *v21; // r9
  unsigned __int64 v22; // r12
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  int v26; // edx
  __int64 v27; // rdi
  _BYTE *v28; // rsi
  _BYTE *v29; // rsi
  int v30; // eax
  int v31; // esi
  __int64 v32; // r9
  unsigned int v33; // edx
  int v34; // r11d
  char *v35; // r10
  int v36; // eax
  int v37; // esi
  __int64 v38; // r9
  char *v39; // r10
  int v40; // r11d
  unsigned int v41; // edx
  char *v42; // r8
  int v43; // r10d
  int v44; // eax
  int v45; // eax
  int v46; // esi
  __int64 v47; // r8
  unsigned __int64 v48; // r9
  int v49; // r10d
  __int64 v50; // rdi
  int v51; // r10d
  char *v52; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 24;
  v52 = a2;
  v4 = *(_DWORD *)(a1 + 48);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_36;
  }
  v5 = v52;
  v6 = *(_QWORD *)(a1 + 32);
  v7 = (v4 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
  result = v6 + 16LL * v7;
  v9 = *(char **)result;
  if ( v52 != *(char **)result )
  {
    v10 = 1;
    v11 = 0;
    while ( v9 != (char *)-8LL )
    {
      if ( !v11 && v9 == (char *)-16LL )
        v11 = (char *)result;
      v7 = (v4 - 1) & (v10 + v7);
      result = v6 + 16LL * v7;
      v9 = *(char **)result;
      if ( v52 == *(char **)result )
        goto LABEL_3;
      ++v10;
    }
    v12 = *(_DWORD *)(a1 + 40);
    if ( v11 )
      result = (unsigned __int64)v11;
    ++*(_QWORD *)(a1 + 24);
    v13 = v12 + 1;
    if ( 4 * v13 < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 44) - v13 > v4 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 40) = v13;
        if ( *(_QWORD *)result != -8 )
          --*(_DWORD *)(a1 + 44);
        *(_DWORD *)(result + 8) = 0;
        *(_QWORD *)result = v5;
        v5 = v52;
        goto LABEL_14;
      }
      sub_1543DE0(v2, v4);
      v36 = *(_DWORD *)(a1 + 48);
      if ( v36 )
      {
        v5 = v52;
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 32);
        v39 = 0;
        v40 = 1;
        v13 = *(_DWORD *)(a1 + 40) + 1;
        v41 = (v36 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        result = v38 + 16LL * v41;
        v42 = *(char **)result;
        if ( *(char **)result != v52 )
        {
          while ( v42 != (char *)-8LL )
          {
            if ( v42 == (char *)-16LL && !v39 )
              v39 = (char *)result;
            v41 = v37 & (v40 + v41);
            result = v38 + 16LL * v41;
            v42 = *(char **)result;
            if ( v52 == *(char **)result )
              goto LABEL_11;
            ++v40;
          }
          if ( v39 )
            result = (unsigned __int64)v39;
        }
        goto LABEL_11;
      }
LABEL_93:
      ++*(_DWORD *)(a1 + 40);
      BUG();
    }
LABEL_36:
    sub_1543DE0(v2, 2 * v4);
    v30 = *(_DWORD *)(a1 + 48);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 32);
      v13 = *(_DWORD *)(a1 + 40) + 1;
      v33 = (v30 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      result = v32 + 16LL * v33;
      v5 = *(char **)result;
      if ( v52 != *(char **)result )
      {
        v34 = 1;
        v35 = 0;
        while ( v5 != (char *)-8LL )
        {
          if ( !v35 && v5 == (char *)-16LL )
            v35 = (char *)result;
          v33 = v31 & (v34 + v33);
          result = v32 + 16LL * v33;
          v5 = *(char **)result;
          if ( v52 == *(char **)result )
            goto LABEL_11;
          ++v34;
        }
        v5 = v52;
        if ( v35 )
          result = (unsigned __int64)v35;
      }
      goto LABEL_11;
    }
    goto LABEL_93;
  }
LABEL_3:
  if ( *(_DWORD *)(result + 8) )
    return result;
LABEL_14:
  if ( v5[8] == 13 && (v5[9] & 4) == 0 )
    *(_DWORD *)(result + 8) = -1;
  v14 = (__int64 *)*((_QWORD *)v5 + 2);
  v15 = &v14[*((unsigned int *)v5 + 3)];
  while ( v15 != v14 )
  {
    v16 = *v14++;
    sub_1543FA0(a1, v16);
  }
  v17 = *(_DWORD *)(a1 + 48);
  if ( !v17 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_24;
  }
  v18 = v52;
  v19 = *(_QWORD *)(a1 + 32);
  v20 = (v17 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
  result = v19 + 16LL * v20;
  v21 = *(char **)result;
  if ( v52 != *(char **)result )
  {
    v43 = 1;
    v22 = 0;
    while ( v21 != (char *)-8LL )
    {
      if ( !v22 && v21 == (char *)-16LL )
        v22 = result;
      v20 = (v17 - 1) & (v43 + v20);
      result = v19 + 16LL * v20;
      v21 = *(char **)result;
      if ( v52 == *(char **)result )
        goto LABEL_21;
      ++v43;
    }
    if ( !v22 )
      v22 = result;
    v44 = *(_DWORD *)(a1 + 40);
    ++*(_QWORD *)(a1 + 24);
    v26 = v44 + 1;
    if ( 4 * (v44 + 1) < 3 * v17 )
    {
      result = v17 - *(_DWORD *)(a1 + 44) - v26;
      if ( (unsigned int)result > v17 >> 3 )
        goto LABEL_26;
      sub_1543DE0(v2, v17);
      v45 = *(_DWORD *)(a1 + 48);
      if ( v45 )
      {
        v18 = v52;
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 32);
        v48 = 0;
        v49 = 1;
        result = (v45 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        v26 = *(_DWORD *)(a1 + 40) + 1;
        v22 = v47 + 16 * result;
        v50 = *(_QWORD *)v22;
        if ( *(char **)v22 != v52 )
        {
          while ( v50 != -8 )
          {
            if ( v50 == -16 && !v48 )
              v48 = v22;
            result = v46 & (unsigned int)(v49 + result);
            v22 = v47 + 16LL * (unsigned int)result;
            v50 = *(_QWORD *)v22;
            if ( v52 == *(char **)v22 )
              goto LABEL_26;
            ++v49;
          }
LABEL_58:
          if ( v48 )
            v22 = v48;
          goto LABEL_26;
        }
        goto LABEL_26;
      }
      goto LABEL_94;
    }
LABEL_24:
    sub_1543DE0(v2, 2 * v17);
    v23 = *(_DWORD *)(a1 + 48);
    if ( v23 )
    {
      v18 = v52;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 32);
      result = (v23 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      v26 = *(_DWORD *)(a1 + 40) + 1;
      v22 = v25 + 16 * result;
      v27 = *(_QWORD *)v22;
      if ( *(char **)v22 != v52 )
      {
        v51 = 1;
        v48 = 0;
        while ( v27 != -8 )
        {
          if ( !v48 && v27 == -16 )
            v48 = v22;
          result = v24 & (unsigned int)(v51 + result);
          v22 = v25 + 16LL * (unsigned int)result;
          v27 = *(_QWORD *)v22;
          if ( v52 == *(char **)v22 )
            goto LABEL_26;
          ++v51;
        }
        goto LABEL_58;
      }
LABEL_26:
      *(_DWORD *)(a1 + 40) = v26;
      if ( *(_QWORD *)v22 != -8 )
        --*(_DWORD *)(a1 + 44);
      *(_QWORD *)v22 = v18;
      *(_DWORD *)(v22 + 8) = 0;
      goto LABEL_29;
    }
LABEL_94:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_21:
  if ( (unsigned int)(*(_DWORD *)(result + 8) - 1) <= 0xFFFFFFFD )
    return result;
  v22 = result;
LABEL_29:
  v28 = *(_BYTE **)(a1 + 64);
  if ( v28 == *(_BYTE **)(a1 + 72) )
  {
    result = (unsigned __int64)sub_1277EB0(a1 + 56, v28, &v52);
    v29 = *(_BYTE **)(a1 + 64);
  }
  else
  {
    if ( v28 )
    {
      result = (unsigned __int64)v52;
      *(_QWORD *)v28 = v52;
      v28 = *(_BYTE **)(a1 + 64);
    }
    v29 = v28 + 8;
    *(_QWORD *)(a1 + 64) = v29;
  }
  *(_DWORD *)(v22 + 8) = (__int64)&v29[-*(_QWORD *)(a1 + 56)] >> 3;
  return result;
}
