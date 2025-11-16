// Function: sub_3170BD0
// Address: 0x3170bd0
//
void __fastcall sub_3170BD0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r14d
  __int64 *v9; // rcx
  unsigned int v10; // r8d
  __int64 *v11; // rax
  __int64 v12; // r10
  bool v13; // cc
  unsigned __int64 v14; // rdi
  const void **v15; // r8
  int v16; // r10d
  __int64 *v17; // r14
  unsigned int v18; // ecx
  __int64 *v19; // rbx
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // edx
  unsigned int v25; // eax
  int v26; // edi
  int v27; // edi
  int v28; // r8d
  __int64 v29; // r9
  __int64 *v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rsi
  int v33; // esi
  int v34; // esi
  __int64 v35; // r8
  unsigned int v36; // ebx
  __int64 v37; // rax
  int v38; // edi
  int v39; // r9d
  int v40; // r9d
  __int64 v41; // r10
  unsigned int v42; // eax
  __int64 v43; // rdi
  int v44; // esi
  __int64 *v45; // rcx
  int v46; // edi
  int v47; // edi
  __int64 v48; // r9
  __int64 *v49; // rsi
  __int64 v50; // r15
  int v51; // eax
  __int64 v52; // rcx

  if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 368), **(_QWORD **)(a1 + 376), a2) )
    return;
  v4 = *(_QWORD *)(a2 + 16);
  if ( !v4 )
    return;
  while ( !(unsigned __int8)sub_B19F20(*(_QWORD *)(a1 + 368), **(char ***)(a1 + 376), v4) )
  {
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      return;
  }
  v5 = *(_QWORD *)(a1 + 400);
  v6 = *(_DWORD *)(a1 + 416);
  v7 = a1 + 392;
  if ( !*(_BYTE *)(a1 + 344) )
  {
    if ( v6 )
    {
      v8 = 1;
      v9 = 0;
      v10 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (__int64 *)(v5 + 32LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
      {
LABEL_10:
        if ( !*((_BYTE *)v11 + 24) )
          return;
        v13 = *((_DWORD *)v11 + 4) <= 0x40u;
        *((_BYTE *)v11 + 24) = 0;
        if ( v13 )
          return;
        v14 = v11[1];
        if ( !v14 )
          return;
        goto LABEL_13;
      }
      while ( v12 != -4096 )
      {
        if ( !v9 && v12 == -8192 )
          v9 = v11;
        v10 = (v6 - 1) & (v8 + v10);
        v11 = (__int64 *)(v5 + 32LL * v10);
        v12 = *v11;
        if ( a2 == *v11 )
          goto LABEL_10;
        ++v8;
      }
      if ( !v9 )
        v9 = v11;
      v21 = *(_DWORD *)(a1 + 408);
      ++*(_QWORD *)(a1 + 392);
      v22 = v21 + 1;
      if ( 4 * v22 < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 412) - v22 > v6 >> 3 )
        {
LABEL_33:
          *(_DWORD *)(a1 + 408) = v22;
          if ( *v9 != -4096 )
            --*(_DWORD *)(a1 + 412);
          *v9 = a2;
          v9[3] = 0;
          *(_OWORD *)(v9 + 1) = 0;
          return;
        }
        sub_3170990(v7, v6);
        v33 = *(_DWORD *)(a1 + 416);
        if ( v33 )
        {
          v34 = v33 - 1;
          v35 = *(_QWORD *)(a1 + 400);
          v36 = v34 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v9 = (__int64 *)(v35 + 32LL * v36);
          v37 = *v9;
          if ( a2 != *v9 )
          {
            v38 = 1;
            v30 = 0;
            while ( v37 != -4096 )
            {
              if ( !v30 && v37 == -8192 )
                v30 = v9;
              v36 = v34 & (v38 + v36);
              v9 = (__int64 *)(v35 + 32LL * v36);
              v37 = *v9;
              if ( a2 == *v9 )
                goto LABEL_55;
              ++v38;
            }
LABEL_60:
            v22 = *(_DWORD *)(a1 + 408) + 1;
            if ( v30 )
              v9 = v30;
            goto LABEL_33;
          }
          goto LABEL_55;
        }
        goto LABEL_97;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 392);
    }
    sub_3170990(v7, 2 * v6);
    v26 = *(_DWORD *)(a1 + 416);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = 1;
      v29 = *(_QWORD *)(a1 + 400);
      v30 = 0;
      v31 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v29 + 32LL * v31);
      v32 = *v9;
      if ( a2 != *v9 )
      {
        while ( v32 != -4096 )
        {
          if ( !v30 && v32 == -8192 )
            v30 = v9;
          v31 = v27 & (v28 + v31);
          v9 = (__int64 *)(v29 + 32LL * v31);
          v32 = *v9;
          if ( a2 == *v9 )
            goto LABEL_55;
          ++v28;
        }
        goto LABEL_60;
      }
LABEL_55:
      v22 = *(_DWORD *)(a1 + 408) + 1;
      goto LABEL_33;
    }
LABEL_97:
    ++*(_DWORD *)(a1 + 408);
    BUG();
  }
  v15 = (const void **)(a1 + 352);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 392);
    goto LABEL_63;
  }
  v16 = 1;
  v17 = 0;
  v18 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (__int64 *)(v5 + 32LL * v18);
  v20 = *v19;
  if ( a2 != *v19 )
  {
    while ( v20 != -4096 )
    {
      if ( v20 == -8192 && !v17 )
        v17 = v19;
      v18 = (v6 - 1) & (v16 + v18);
      v19 = (__int64 *)(v5 + 32LL * v18);
      v20 = *v19;
      if ( a2 == *v19 )
        goto LABEL_16;
      ++v16;
    }
    v23 = *(_DWORD *)(a1 + 408);
    if ( !v17 )
      v17 = v19;
    ++*(_QWORD *)(a1 + 392);
    v24 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 412) - v24 <= v6 >> 3 )
      {
        sub_3170990(v7, v6);
        v46 = *(_DWORD *)(a1 + 416);
        if ( !v46 )
          goto LABEL_97;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 400);
        v49 = 0;
        v15 = (const void **)(a1 + 352);
        LODWORD(v50) = v47 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v24 = *(_DWORD *)(a1 + 408) + 1;
        v51 = 1;
        v17 = (__int64 *)(v48 + 32LL * (unsigned int)v50);
        v52 = *v17;
        if ( a2 != *v17 )
        {
          while ( v52 != -4096 )
          {
            if ( !v49 && v52 == -8192 )
              v49 = v17;
            v50 = v47 & (unsigned int)(v50 + v51);
            v17 = (__int64 *)(v48 + 32 * v50);
            v52 = *v17;
            if ( a2 == *v17 )
              goto LABEL_46;
            ++v51;
          }
          if ( v49 )
            v17 = v49;
        }
      }
      goto LABEL_46;
    }
LABEL_63:
    sub_3170990(v7, 2 * v6);
    v39 = *(_DWORD *)(a1 + 416);
    if ( !v39 )
      goto LABEL_97;
    v40 = v39 - 1;
    v15 = (const void **)(a1 + 352);
    v41 = *(_QWORD *)(a1 + 400);
    v42 = v40 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v24 = *(_DWORD *)(a1 + 408) + 1;
    v17 = (__int64 *)(v41 + 32LL * v42);
    v43 = *v17;
    if ( a2 != *v17 )
    {
      v44 = 1;
      v45 = 0;
      while ( v43 != -4096 )
      {
        if ( !v45 && v43 == -8192 )
          v45 = v17;
        v42 = v40 & (v44 + v42);
        v17 = (__int64 *)(v41 + 32LL * v42);
        v43 = *v17;
        if ( a2 == *v17 )
          goto LABEL_46;
        ++v44;
      }
      if ( v45 )
        v17 = v45;
    }
LABEL_46:
    *(_DWORD *)(a1 + 408) = v24;
    if ( *v17 != -4096 )
      --*(_DWORD *)(a1 + 412);
    *v17 = a2;
    v25 = *(_DWORD *)(a1 + 360);
    *((_DWORD *)v17 + 4) = v25;
    if ( v25 > 0x40 )
      sub_C43780((__int64)(v17 + 1), v15);
    else
      v17[1] = *(_QWORD *)(a1 + 352);
    *((_BYTE *)v17 + 24) = 1;
    return;
  }
LABEL_16:
  if ( !*((_BYTE *)v19 + 24) )
    return;
  if ( *((_DWORD *)v19 + 4) > 0x40u )
  {
    if ( sub_C43C50((__int64)(v19 + 1), (const void **)(a1 + 352)) )
      return;
    v14 = v19[1];
    *((_BYTE *)v19 + 24) = 0;
    if ( !v14 )
      return;
LABEL_13:
    j_j___libc_free_0_0(v14);
    return;
  }
  if ( v19[1] != *(_QWORD *)(a1 + 352) )
    *((_BYTE *)v19 + 24) = 0;
}
