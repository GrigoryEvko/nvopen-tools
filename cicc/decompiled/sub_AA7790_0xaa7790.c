// Function: sub_AA7790
// Address: 0xaa7790
//
__int64 *__fastcall sub_AA7790(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r12
  char v6; // di
  __int64 v7; // r8
  int v8; // esi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  unsigned int v13; // esi
  unsigned int v14; // edx
  __int64 *v15; // r9
  int v16; // eax
  unsigned int v17; // r8d
  __int64 v18; // rdi
  int v19; // r10d
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 v27; // rax
  int v28; // r8d
  __int64 *v29; // rdi
  int v30; // ecx
  int v31; // ecx
  int v32; // r8d

  v4 = sub_AA48A0(a1);
  v5 = *(_QWORD *)v4;
  v6 = *(_BYTE *)(*(_QWORD *)v4 + 3520LL) & 1;
  if ( v6 )
  {
    v7 = v5 + 3528;
    v8 = 3;
  }
  else
  {
    v13 = *(_DWORD *)(v5 + 3536);
    v7 = *(_QWORD *)(v5 + 3528);
    if ( !v13 )
    {
      v14 = *(_DWORD *)(v5 + 3520);
      v15 = 0;
      ++*(_QWORD *)(v5 + 3512);
      v16 = (v14 >> 1) + 1;
LABEL_8:
      v17 = 3 * v13;
      goto LABEL_9;
    }
    v8 = v13 - 1;
  }
  v9 = v8 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a1 == *v10 )
  {
LABEL_4:
    v10[1] = a2;
    return v10 + 1;
  }
  v19 = 1;
  v15 = 0;
  while ( v11 != -4096 )
  {
    if ( !v15 && v11 == -8192 )
      v15 = v10;
    v9 = v8 & (v19 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a1 == *v10 )
      goto LABEL_4;
    ++v19;
  }
  v14 = *(_DWORD *)(v5 + 3520);
  v17 = 12;
  v13 = 4;
  if ( !v15 )
    v15 = v10;
  ++*(_QWORD *)(v5 + 3512);
  v16 = (v14 >> 1) + 1;
  if ( !v6 )
  {
    v13 = *(_DWORD *)(v5 + 3536);
    goto LABEL_8;
  }
LABEL_9:
  v18 = v5 + 3512;
  if ( 4 * v16 >= v17 )
  {
    sub_AA7370(v18, 2 * v13);
    if ( (*(_BYTE *)(v5 + 3520) & 1) != 0 )
    {
      v20 = v5 + 3528;
      v21 = 3;
    }
    else
    {
      v30 = *(_DWORD *)(v5 + 3536);
      v20 = *(_QWORD *)(v5 + 3528);
      if ( !v30 )
        goto LABEL_52;
      v21 = v30 - 1;
    }
    v22 = v21 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v15 = (__int64 *)(v20 + 16LL * v22);
    v23 = *v15;
    if ( a1 != *v15 )
    {
      v32 = 1;
      v29 = 0;
      while ( v23 != -4096 )
      {
        if ( !v29 && v23 == -8192 )
          v29 = v15;
        v22 = v21 & (v32 + v22);
        v15 = (__int64 *)(v20 + 16LL * v22);
        v23 = *v15;
        if ( a1 == *v15 )
          goto LABEL_23;
        ++v32;
      }
      goto LABEL_29;
    }
LABEL_23:
    v14 = *(_DWORD *)(v5 + 3520);
    goto LABEL_11;
  }
  if ( v13 - *(_DWORD *)(v5 + 3524) - v16 <= v13 >> 3 )
  {
    sub_AA7370(v18, v13);
    if ( (*(_BYTE *)(v5 + 3520) & 1) != 0 )
    {
      v24 = v5 + 3528;
      v25 = 3;
      goto LABEL_26;
    }
    v31 = *(_DWORD *)(v5 + 3536);
    v24 = *(_QWORD *)(v5 + 3528);
    if ( v31 )
    {
      v25 = v31 - 1;
LABEL_26:
      v26 = v25 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v15 = (__int64 *)(v24 + 16LL * v26);
      v27 = *v15;
      if ( a1 != *v15 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v29 )
            v29 = v15;
          v26 = v25 & (v28 + v26);
          v15 = (__int64 *)(v24 + 16LL * v26);
          v27 = *v15;
          if ( a1 == *v15 )
            goto LABEL_23;
          ++v28;
        }
LABEL_29:
        if ( v29 )
          v15 = v29;
        goto LABEL_23;
      }
      goto LABEL_23;
    }
LABEL_52:
    *(_DWORD *)(v5 + 3520) = (2 * (*(_DWORD *)(v5 + 3520) >> 1) + 2) | *(_DWORD *)(v5 + 3520) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(v5 + 3520) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *v15 != -4096 )
    --*(_DWORD *)(v5 + 3524);
  *v15 = a1;
  v15[1] = 0;
  v15[1] = a2;
  return v15 + 1;
}
