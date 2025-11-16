// Function: sub_AD51C0
// Address: 0xad51c0
//
__int64 __fastcall sub_AD51C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r8
  int v11; // r11d
  __int64 *v12; // rdx
  unsigned int v13; // r9d
  __int64 *v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r13
  int v19; // eax
  int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  int v24; // esi
  __int64 v25; // r8
  int v26; // esi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // r8
  unsigned int v35; // eax
  __int64 v36; // rdi
  int v37; // r10d
  __int64 *v38; // r9
  int v39; // eax
  int v40; // eax
  __int64 v41; // rdi
  __int64 *v42; // r8
  unsigned int v43; // r14d
  int v44; // r9d
  int v45; // edx
  int v46; // r10d

  v5 = sub_BD3990(a3);
  if ( *(_BYTE *)v5 >= 4u )
    v5 = 0;
  v6 = sub_BD5C60(a1, a2, v4);
  v7 = *(_QWORD *)v6;
  v8 = *(unsigned int *)(*(_QWORD *)v6 + 2080LL);
  v9 = *(_QWORD *)v6 + 2056LL;
  if ( !(_DWORD)v8 )
  {
    ++*(_QWORD *)(v7 + 2056);
    goto LABEL_34;
  }
  v10 = *(_QWORD *)(v7 + 2064);
  v11 = 1;
  v12 = 0;
  v13 = (v8 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( v5 != *v14 )
  {
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      v13 = (v8 - 1) & (v11 + v13);
      v14 = (__int64 *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( v5 == *v14 )
        goto LABEL_5;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
    v19 = *(_DWORD *)(v7 + 2072);
    ++*(_QWORD *)(v7 + 2056);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < (unsigned int)(3 * v8) )
    {
      if ( (int)v8 - *(_DWORD *)(v7 + 2076) - v20 > (unsigned int)v8 >> 3 )
      {
LABEL_17:
        *(_DWORD *)(v7 + 2072) = v20;
        if ( *v12 != -4096 )
          --*(_DWORD *)(v7 + 2076);
        *v12 = v5;
        v17 = v12 + 1;
        v12[1] = 0;
        goto LABEL_20;
      }
      sub_ACC9A0(v9, v8);
      v39 = *(_DWORD *)(v7 + 2080);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(v7 + 2064);
        v42 = 0;
        v43 = v40 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v44 = 1;
        v20 = *(_DWORD *)(v7 + 2072) + 1;
        v12 = (__int64 *)(v41 + 16LL * v43);
        v8 = *v12;
        if ( v5 != *v12 )
        {
          while ( v8 != -4096 )
          {
            if ( !v42 && v8 == -8192 )
              v42 = v12;
            v43 = v40 & (v44 + v43);
            v12 = (__int64 *)(v41 + 16LL * v43);
            v8 = *v12;
            if ( v5 == *v12 )
              goto LABEL_17;
            ++v44;
          }
          if ( v42 )
            v12 = v42;
        }
        goto LABEL_17;
      }
LABEL_61:
      ++*(_DWORD *)(v7 + 2072);
      BUG();
    }
LABEL_34:
    sub_ACC9A0(v9, 2 * v8);
    v33 = *(_DWORD *)(v7 + 2080);
    if ( v33 )
    {
      v8 = (unsigned int)(v33 - 1);
      v34 = *(_QWORD *)(v7 + 2064);
      v35 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v20 = *(_DWORD *)(v7 + 2072) + 1;
      v12 = (__int64 *)(v34 + 16LL * v35);
      v36 = *v12;
      if ( v5 != *v12 )
      {
        v37 = 1;
        v38 = 0;
        while ( v36 != -4096 )
        {
          if ( !v38 && v36 == -8192 )
            v38 = v12;
          v35 = v8 & (v37 + v35);
          v12 = (__int64 *)(v34 + 16LL * v35);
          v36 = *v12;
          if ( v5 == *v12 )
            goto LABEL_17;
          ++v37;
        }
        if ( v38 )
          v12 = v38;
      }
      goto LABEL_17;
    }
    goto LABEL_61;
  }
LABEL_5:
  v16 = v14[1];
  v17 = v14 + 1;
  if ( v16 )
    return sub_AD4C90(v16, *(__int64 ***)(a1 + 8), 0);
LABEL_20:
  v21 = (__int64 *)sub_BD5C60(a1, v8, v12);
  v22 = *(_QWORD *)(a1 - 32);
  v23 = *v21;
  v24 = *(_DWORD *)(v23 + 2080);
  v25 = *(_QWORD *)(v23 + 2064);
  if ( v24 )
  {
    v26 = v24 - 1;
    v27 = v26 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v28 = (__int64 *)(v25 + 16LL * v27);
    v29 = *v28;
    if ( v22 == *v28 )
    {
LABEL_22:
      *v28 = -8192;
      --*(_DWORD *)(v23 + 2072);
      ++*(_DWORD *)(v23 + 2076);
    }
    else
    {
      v45 = 1;
      while ( v29 != -4096 )
      {
        v46 = v45 + 1;
        v27 = v26 & (v45 + v27);
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( v22 == *v28 )
          goto LABEL_22;
        v45 = v46;
      }
    }
  }
  *v17 = a1;
  if ( *(_QWORD *)(a1 - 32) )
  {
    v30 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = v5;
  if ( v5 )
  {
    v31 = *(_QWORD *)(v5 + 16);
    *(_QWORD *)(a1 - 24) = v31;
    if ( v31 )
      *(_QWORD *)(v31 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = v5 + 16;
    *(_QWORD *)(v5 + 16) = a1 - 32;
  }
  v32 = *(_QWORD *)(v5 + 8);
  if ( *(_QWORD *)(a1 + 8) != v32 )
    *(_QWORD *)(a1 + 8) = v32;
  return 0;
}
