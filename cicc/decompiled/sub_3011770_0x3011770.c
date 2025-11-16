// Function: sub_3011770
// Address: 0x3011770
//
_QWORD *__fastcall sub_3011770(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r12
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned __int64 *v8; // rdx
  int v9; // r11d
  unsigned int v10; // ecx
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // r9
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // r8
  unsigned int v16; // esi
  __int64 v17; // r9
  int v18; // r11d
  unsigned int v19; // edi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdi
  _QWORD *result; // rax
  int v25; // ecx
  int v26; // ecx
  int v27; // eax
  int v28; // ecx
  int v29; // eax
  int v30; // edi
  __int64 v31; // r10
  unsigned int v32; // esi
  unsigned __int64 v33; // rax
  int v34; // r9d
  unsigned __int64 *v35; // r8
  int v36; // eax
  int v37; // esi
  unsigned int v38; // edx
  __int64 v39; // rdi
  __int64 v40; // r10
  int v41; // eax
  int v42; // edx
  int v43; // edi
  __int64 v44; // r14
  __int64 v45; // rsi
  int v46; // eax
  int v47; // esi
  __int64 v48; // r8
  int v49; // edi
  __int64 v50; // r14
  unsigned __int64 *v51; // r9
  unsigned __int64 v52; // rax

  v4 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_42;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = (v6 - 1) & (37 * v4);
  v11 = (unsigned __int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( v4 == *v11 )
  {
LABEL_3:
    v13 = v11 + 1;
    goto LABEL_4;
  }
  while ( v12 != -4096 )
  {
    if ( !v8 && v12 == -8192 )
      v8 = v11;
    v10 = (v6 - 1) & (v9 + v10);
    v11 = (unsigned __int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v4 == *v11 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v27 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v28 = v27 + 1;
  if ( 4 * (v27 + 1) >= 3 * v6 )
  {
LABEL_42:
    sub_3011340(a1, 2 * v6);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 8);
      v32 = (v29 - 1) & (37 * v4);
      v28 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (unsigned __int64 *)(v31 + 16LL * v32);
      v33 = *v8;
      if ( v4 != *v8 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v8;
          v32 = v30 & (v34 + v32);
          v8 = (unsigned __int64 *)(v31 + 16LL * v32);
          v33 = *v8;
          if ( v4 == *v8 )
            goto LABEL_38;
          ++v34;
        }
        if ( v35 )
          v8 = v35;
      }
      goto LABEL_38;
    }
    goto LABEL_89;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v28 <= v6 >> 3 )
  {
    sub_3011340(a1, v6);
    v46 = *(_DWORD *)(a1 + 24);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 8);
      v49 = 1;
      LODWORD(v50) = (v46 - 1) & (37 * v4);
      v51 = 0;
      v28 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (unsigned __int64 *)(v48 + 16LL * (unsigned int)v50);
      v52 = *v8;
      if ( v4 != *v8 )
      {
        while ( v52 != -4096 )
        {
          if ( v52 == -8192 && !v51 )
            v51 = v8;
          v50 = v47 & (unsigned int)(v50 + v49);
          v8 = (unsigned __int64 *)(v48 + 16 * v50);
          v52 = *v8;
          if ( v4 == *v8 )
            goto LABEL_38;
          ++v49;
        }
        if ( v51 )
          v8 = v51;
      }
      goto LABEL_38;
    }
LABEL_89:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_38:
  *(_DWORD *)(a1 + 16) = v28;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v13 = v8 + 1;
  v8[1] = 0;
LABEL_4:
  v14 = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v15 = a1 + 32;
  *v13 = v14;
  v16 = *(_DWORD *)(a1 + 56);
  if ( !v16 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_50;
  }
  v17 = *(_QWORD *)(a1 + 40);
  v18 = 1;
  v19 = (v16 - 1) & (37 * v14);
  v20 = v17 + 72LL * v19;
  v21 = 0;
  v22 = *(_QWORD *)v20;
  if ( v14 != *(_QWORD *)v20 )
  {
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !v21 )
        v21 = v20;
      v19 = (v16 - 1) & (v18 + v19);
      v20 = v17 + 72LL * v19;
      v22 = *(_QWORD *)v20;
      if ( v14 == *(_QWORD *)v20 )
        goto LABEL_6;
      ++v18;
    }
    v25 = *(_DWORD *)(a1 + 48);
    if ( !v21 )
      v21 = v20;
    ++*(_QWORD *)(a1 + 32);
    v26 = v25 + 1;
    if ( 4 * v26 < 3 * v16 )
    {
      if ( v16 - *(_DWORD *)(a1 + 52) - v26 > v16 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(a1 + 48) = v26;
        if ( *(_QWORD *)v21 != -4096 )
          --*(_DWORD *)(a1 + 52);
        *(_QWORD *)v21 = v14;
        v23 = v21 + 8;
        *(_QWORD *)(v21 + 8) = 0;
        *(_QWORD *)(v21 + 16) = v21 + 40;
        *(_QWORD *)(v21 + 24) = 4;
        *(_DWORD *)(v21 + 32) = 0;
        *(_BYTE *)(v21 + 36) = 1;
        goto LABEL_21;
      }
      sub_3011530(a1 + 32, v16);
      v41 = *(_DWORD *)(a1 + 56);
      if ( v41 )
      {
        v42 = v41 - 1;
        v15 = *(_QWORD *)(a1 + 40);
        v43 = 1;
        LODWORD(v44) = (v41 - 1) & (37 * v14);
        v17 = 0;
        v26 = *(_DWORD *)(a1 + 48) + 1;
        v21 = v15 + 72LL * (unsigned int)v44;
        v45 = *(_QWORD *)v21;
        if ( v14 != *(_QWORD *)v21 )
        {
          while ( v45 != -4096 )
          {
            if ( v45 == -8192 && !v17 )
              v17 = v21;
            v44 = v42 & (unsigned int)(v44 + v43);
            v21 = v15 + 72 * v44;
            v45 = *(_QWORD *)v21;
            if ( v14 == *(_QWORD *)v21 )
              goto LABEL_18;
            ++v43;
          }
          if ( v17 )
            v21 = v17;
        }
        goto LABEL_18;
      }
LABEL_90:
      ++*(_DWORD *)(a1 + 48);
      BUG();
    }
LABEL_50:
    sub_3011530(a1 + 32, 2 * v16);
    v36 = *(_DWORD *)(a1 + 56);
    if ( v36 )
    {
      v37 = v36 - 1;
      v17 = *(_QWORD *)(a1 + 40);
      v38 = (v36 - 1) & (37 * v14);
      v26 = *(_DWORD *)(a1 + 48) + 1;
      v21 = v17 + 72LL * v38;
      v39 = *(_QWORD *)v21;
      if ( v14 != *(_QWORD *)v21 )
      {
        v15 = 1;
        v40 = 0;
        while ( v39 != -4096 )
        {
          if ( !v40 && v39 == -8192 )
            v40 = v21;
          v38 = v37 & (v15 + v38);
          v21 = v17 + 72LL * v38;
          v39 = *(_QWORD *)v21;
          if ( v14 == *(_QWORD *)v21 )
            goto LABEL_18;
          v15 = (unsigned int)(v15 + 1);
        }
        if ( v40 )
          v21 = v40;
      }
      goto LABEL_18;
    }
    goto LABEL_90;
  }
LABEL_6:
  v23 = v20 + 8;
  if ( !*(_BYTE *)(v20 + 36) )
    return sub_C8CC70(v23, v4, v20, v22, v15, v17);
LABEL_21:
  result = *(_QWORD **)(v23 + 8);
  v22 = *(unsigned int *)(v23 + 20);
  v20 = (__int64)&result[v22];
  if ( result == (_QWORD *)v20 )
  {
LABEL_26:
    if ( (unsigned int)v22 >= *(_DWORD *)(v23 + 16) )
      return sub_C8CC70(v23, v4, v20, v22, v15, v17);
    *(_DWORD *)(v23 + 20) = v22 + 1;
    *(_QWORD *)v20 = v4;
    ++*(_QWORD *)v23;
  }
  else
  {
    while ( v4 != *result )
    {
      if ( (_QWORD *)v20 == ++result )
        goto LABEL_26;
    }
  }
  return result;
}
