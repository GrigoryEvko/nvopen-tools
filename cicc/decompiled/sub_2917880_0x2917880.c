// Function: sub_2917880
// Address: 0x2917880
//
__int64 __fastcall sub_2917880(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 v9; // r14
  char v10; // cl
  __int64 v11; // rax
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // r15
  __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rsi
  unsigned int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // r10
  const void *v22; // rsi
  const void *v23; // rdi
  size_t v24; // rdx
  __int64 v25; // rdi
  _QWORD *v26; // rax
  int v27; // ecx
  unsigned int v28; // eax
  __int64 v29; // rdi
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // r15
  int v34; // r8d
  int v35; // r11d
  int v36; // ecx
  unsigned int v37; // edx
  int v38; // esi
  unsigned int v39; // eax
  int v40; // edx
  int v41; // esi
  __int64 v42; // rsi

  v4 = *a1;
  v5 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)(v4 + 28) )
  {
    v6 = *(_QWORD **)(v4 + 8);
    v7 = &v6[*(unsigned int *)(v4 + 20)];
    if ( v6 != v7 )
    {
      while ( v5 != *v6 )
      {
        if ( v7 == ++v6 )
          goto LABEL_8;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(v4, *(_QWORD *)(a2 - 64)) )
  {
    return 1;
  }
LABEL_8:
  v9 = a1[1];
  v10 = *(_BYTE *)(v9 + 8) & 1;
  if ( v10 )
  {
    v11 = v9 + 16;
    v12 = 7;
  }
  else
  {
    v32 = *(unsigned int *)(v9 + 24);
    v11 = *(_QWORD *)(v9 + 16);
    if ( !(_DWORD)v32 )
      goto LABEL_43;
    v12 = v32 - 1;
  }
  v13 = v12 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v14 = (__int64 *)(v11 + 40LL * v13);
  v15 = *v14;
  if ( v5 == *v14 )
    goto LABEL_11;
  v34 = 1;
  while ( v15 != -4096 )
  {
    v13 = v12 & (v34 + v13);
    v14 = (__int64 *)(v11 + 40LL * v13);
    v15 = *v14;
    if ( v5 == *v14 )
      goto LABEL_11;
    ++v34;
  }
  if ( v10 )
  {
    v33 = 320;
    goto LABEL_44;
  }
  v32 = *(unsigned int *)(v9 + 24);
LABEL_43:
  v33 = 40 * v32;
LABEL_44:
  v14 = (__int64 *)(v11 + v33);
LABEL_11:
  v16 = *(_QWORD *)v9;
  if ( v10 )
  {
    if ( v14 != (__int64 *)(v11 + 320) )
    {
      v17 = 7;
      LODWORD(v18) = 8;
      goto LABEL_14;
    }
    return 0;
  }
  v18 = *(unsigned int *)(v9 + 24);
  if ( v14 == (__int64 *)(v11 + 40 * v18) )
    return 0;
  if ( !(_DWORD)v18 )
  {
    *(_QWORD *)v9 = v16 + 1;
    goto LABEL_25;
  }
  v17 = (unsigned int)(v18 - 1);
LABEL_14:
  v19 = v17 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = v11 + 40LL * v19;
  v21 = *(_QWORD *)v20;
  if ( a2 != *(_QWORD *)v20 )
  {
    v35 = 1;
    v29 = 0;
    while ( v21 != -4096 )
    {
      if ( v21 == -8192 && !v29 )
        v29 = v20;
      v19 = v17 & (v35 + v19);
      v20 = v11 + 40LL * v19;
      v21 = *(_QWORD *)v20;
      if ( a2 == *(_QWORD *)v20 )
        goto LABEL_15;
      ++v35;
    }
    v31 = *(_DWORD *)(v9 + 8);
    if ( !v29 )
      v29 = v20;
    *(_QWORD *)v9 = v16 + 1;
    v36 = (v31 >> 1) + 1;
    v16 = (unsigned int)(4 * v36);
    if ( (unsigned int)v16 < 3 * (int)v18 )
    {
      v37 = v18 - *(_DWORD *)(v9 + 12) - v36;
      v20 = (unsigned int)v18 >> 3;
      if ( v37 <= (unsigned int)v20 )
      {
        sub_2916630(v9, v18);
        if ( (*(_BYTE *)(v9 + 8) & 1) != 0 )
        {
          v16 = v9 + 16;
          v38 = 7;
          goto LABEL_59;
        }
        v41 = *(_DWORD *)(v9 + 24);
        v16 = *(_QWORD *)(v9 + 16);
        if ( v41 )
        {
          v38 = v41 - 1;
LABEL_59:
          v39 = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v29 = v16 + 40LL * v39;
          v20 = *(_QWORD *)v29;
          if ( a2 != *(_QWORD *)v29 )
          {
            v40 = 1;
            v17 = 0;
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v17 )
                v17 = v29;
              v39 = v38 & (v40 + v39);
              v29 = v16 + 40LL * v39;
              v20 = *(_QWORD *)v29;
              if ( a2 == *(_QWORD *)v29 )
                goto LABEL_29;
              ++v40;
            }
            if ( v17 )
              v29 = v17;
          }
LABEL_29:
          v31 = *(_DWORD *)(v9 + 8);
          goto LABEL_30;
        }
LABEL_87:
        *(_DWORD *)(v9 + 8) = (2 * (*(_DWORD *)(v9 + 8) >> 1) + 2) | *(_DWORD *)(v9 + 8) & 1;
        BUG();
      }
      goto LABEL_30;
    }
LABEL_25:
    sub_2916630(v9, 2 * v18);
    if ( (*(_BYTE *)(v9 + 8) & 1) != 0 )
    {
      v17 = v9 + 16;
      v20 = 7;
    }
    else
    {
      v27 = *(_DWORD *)(v9 + 24);
      v17 = *(_QWORD *)(v9 + 16);
      if ( !v27 )
        goto LABEL_87;
      v20 = (unsigned int)(v27 - 1);
    }
    v28 = v20 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v29 = v17 + 40LL * v28;
    v30 = *(_QWORD *)v29;
    if ( a2 == *(_QWORD *)v29 )
      goto LABEL_29;
    v16 = 1;
    v42 = 0;
    while ( v30 != -4096 )
    {
      if ( !v42 && v30 == -8192 )
        v42 = v29;
      v28 = v20 & (v16 + v28);
      v29 = v17 + 40LL * v28;
      v30 = *(_QWORD *)v29;
      if ( a2 == *(_QWORD *)v29 )
        goto LABEL_29;
      v16 = (unsigned int)(v16 + 1);
    }
    if ( !v42 )
      goto LABEL_29;
    v31 = *(_DWORD *)(v9 + 8);
    v29 = v42;
LABEL_30:
    v24 = 2 * (v31 >> 1) + 2;
    *(_DWORD *)(v9 + 8) = v24 | v31 & 1;
    if ( *(_QWORD *)v29 != -4096 )
      --*(_DWORD *)(v9 + 12);
    *(_QWORD *)v29 = a2;
    *(_OWORD *)(v29 + 8) = 0;
    *(_OWORD *)(v29 + 24) = 0;
    if ( v14[3] != v14[2] )
      goto LABEL_16;
    return 0;
  }
LABEL_15:
  v22 = *(const void **)(v20 + 16);
  v23 = (const void *)v14[2];
  v24 = v14[3] - (_QWORD)v23;
  if ( v24 != *(_QWORD *)(v20 + 24) - (_QWORD)v22 )
  {
LABEL_16:
    v25 = *a1;
    if ( !*(_BYTE *)(*a1 + 28) )
      goto LABEL_38;
    v26 = *(_QWORD **)(v25 + 8);
    v20 = *(unsigned int *)(v25 + 20);
    v24 = (size_t)&v26[v20];
    if ( v26 != (_QWORD *)v24 )
    {
      while ( v5 != *v26 )
      {
        if ( (_QWORD *)v24 == ++v26 )
          goto LABEL_37;
      }
      return 1;
    }
LABEL_37:
    if ( (unsigned int)v20 < *(_DWORD *)(v25 + 16) )
    {
      *(_DWORD *)(v25 + 20) = v20 + 1;
      *(_QWORD *)v24 = v5;
      ++*(_QWORD *)v25;
    }
    else
    {
LABEL_38:
      sub_C8CC70(v25, v5, v24, v20, v16, v17);
    }
    return 1;
  }
  if ( !v24 )
    return 0;
  if ( memcmp(v23, v22, v24) )
    goto LABEL_16;
  return 0;
}
