// Function: sub_2B54920
// Address: 0x2b54920
//
__int64 __fastcall sub_2B54920(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  char v9; // cl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // edx
  __int64 v13; // rax
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 v17; // r8
  int v18; // edx
  unsigned int v19; // edi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rcx
  int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // r10d
  __int64 v28; // r12
  _QWORD *v29; // rdx
  __int64 v30; // rcx
  int v31; // edx
  unsigned int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rcx
  int v35; // edx
  unsigned int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rdi
  int v39; // edx
  int v40; // edx

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 15;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v17 = 0;
      v18 = (v16 >> 1) + 1;
LABEL_8:
      v19 = 3 * v15;
      goto LABEL_9;
    }
    v11 = v15 - 1;
  }
  v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = v10 + 16LL * v12;
  a6 = *(_QWORD *)v13;
  if ( v8 == *(_QWORD *)v13 )
    return *(_QWORD *)(a1 + 272) + 16LL * *(unsigned int *)(v13 + 8);
  v27 = 1;
  v17 = 0;
  while ( a6 != -4096 )
  {
    if ( !v17 && a6 == -8192 )
      v17 = v13;
    v12 = v11 & (v27 + v12);
    v13 = v10 + 16LL * v12;
    a6 = *(_QWORD *)v13;
    if ( v8 == *(_QWORD *)v13 )
      return *(_QWORD *)(a1 + 272) + 16LL * *(unsigned int *)(v13 + 8);
    ++v27;
  }
  if ( !v17 )
    v17 = v13;
  v16 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v18 = (v16 >> 1) + 1;
  if ( !v9 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v19 = 48;
  v15 = 16;
LABEL_9:
  if ( v19 <= 4 * v18 )
  {
    sub_2281F90(a1, 2 * v15);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v30 = a1 + 16;
      v31 = 15;
    }
    else
    {
      v39 = *(_DWORD *)(a1 + 24);
      v30 = *(_QWORD *)(a1 + 16);
      if ( !v39 )
        goto LABEL_60;
      v31 = v39 - 1;
    }
    v32 = v31 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v17 = v30 + 16LL * v32;
    v33 = *(_QWORD *)v17;
    if ( v8 != *(_QWORD *)v17 )
    {
      a6 = 1;
      v38 = 0;
      while ( v33 != -4096 )
      {
        if ( !v38 && v33 == -8192 )
          v38 = v17;
        v32 = v31 & (a6 + v32);
        v17 = v30 + 16LL * v32;
        v33 = *(_QWORD *)v17;
        if ( v8 == *(_QWORD *)v17 )
          goto LABEL_30;
        a6 = (unsigned int)(a6 + 1);
      }
      goto LABEL_36;
    }
LABEL_30:
    v16 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v15 - *(_DWORD *)(a1 + 12) - v18 <= v15 >> 3 )
  {
    sub_2281F90(a1, v15);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v34 = a1 + 16;
      v35 = 15;
      goto LABEL_33;
    }
    v40 = *(_DWORD *)(a1 + 24);
    v34 = *(_QWORD *)(a1 + 16);
    if ( v40 )
    {
      v35 = v40 - 1;
LABEL_33:
      v36 = v35 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v17 = v34 + 16LL * v36;
      v37 = *(_QWORD *)v17;
      if ( v8 != *(_QWORD *)v17 )
      {
        a6 = 1;
        v38 = 0;
        while ( v37 != -4096 )
        {
          if ( !v38 && v37 == -8192 )
            v38 = v17;
          v36 = v35 & (a6 + v36);
          v17 = v34 + 16LL * v36;
          v37 = *(_QWORD *)v17;
          if ( v8 == *(_QWORD *)v17 )
            goto LABEL_30;
          a6 = (unsigned int)(a6 + 1);
        }
LABEL_36:
        if ( v38 )
          v17 = v38;
        goto LABEL_30;
      }
      goto LABEL_30;
    }
LABEL_60:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
  if ( *(_QWORD *)v17 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_DWORD *)(v17 + 8) = 0;
  *(_QWORD *)v17 = v8;
  *(_DWORD *)(v17 + 8) = *(_DWORD *)(a1 + 280);
  v20 = *(unsigned int *)(a1 + 280);
  v21 = *(unsigned int *)(a1 + 284);
  v22 = *(_DWORD *)(a1 + 280);
  if ( v20 >= v21 )
  {
    v28 = *a2;
    if ( v21 < v20 + 1 )
    {
      sub_C8D5F0(a1 + 272, (const void *)(a1 + 288), v20 + 1, 0x10u, v20 + 1, a6);
      v20 = *(unsigned int *)(a1 + 280);
    }
    v29 = (_QWORD *)(*(_QWORD *)(a1 + 272) + 16 * v20);
    *v29 = v28;
    v29[1] = 0;
    v23 = *(_QWORD *)(a1 + 272);
    v26 = (unsigned int)(*(_DWORD *)(a1 + 280) + 1);
    *(_DWORD *)(a1 + 280) = v26;
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 272);
    v24 = v23 + 16 * v20;
    if ( v24 )
    {
      v25 = *a2;
      *(_DWORD *)(v24 + 8) = 0;
      *(_QWORD *)v24 = v25;
      v22 = *(_DWORD *)(a1 + 280);
      v23 = *(_QWORD *)(a1 + 272);
    }
    v26 = (unsigned int)(v22 + 1);
    *(_DWORD *)(a1 + 280) = v26;
  }
  return v23 + 16 * v26 - 16;
}
