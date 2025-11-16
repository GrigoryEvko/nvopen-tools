// Function: sub_35E8960
// Address: 0x35e8960
//
__int64 __fastcall sub_35E8960(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r14d
  __int64 v7; // r8
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r10
  __int64 v12; // r12
  unsigned int v13; // esi
  __int64 v14; // rdi
  int v15; // r14d
  __int64 v16; // r8
  __int64 *v17; // rdx
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r10
  int v22; // eax
  int v23; // ecx
  int v24; // eax
  int v25; // ecx
  int v26; // eax
  int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // r10d
  _QWORD *v32; // r9
  int v33; // eax
  int v34; // esi
  __int64 v35; // r8
  unsigned int v36; // eax
  __int64 v37; // rdi
  int v38; // r10d
  __int64 *v39; // r9
  int v40; // eax
  int v41; // eax
  __int64 v42; // rdi
  __int64 *v43; // r8
  unsigned int v44; // r13d
  int v45; // r9d
  __int64 v46; // rsi
  int v47; // eax
  int v48; // eax
  __int64 v49; // rdi
  _QWORD *v50; // r8
  unsigned int v51; // r13d
  int v52; // r9d
  __int64 v53; // rsi

  v4 = a1 + 208;
  v5 = *(_DWORD *)(a1 + 232);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 208);
    goto LABEL_34;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 216);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
  {
LABEL_3:
    v12 = v10[1];
    goto LABEL_4;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v22 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  v23 = v22 + 1;
  if ( 4 * (v22 + 1) >= 3 * v5 )
  {
LABEL_34:
    sub_2E48800(v4, 2 * v5);
    v26 = *(_DWORD *)(a1 + 232);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 216);
      v29 = (v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 224) + 1;
      v8 = (_QWORD *)(v28 + 16LL * v29);
      v30 = *v8;
      if ( *v8 != a2 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( !v32 && v30 == -8192 )
            v32 = v8;
          v29 = v27 & (v31 + v29);
          v8 = (_QWORD *)(v28 + 16LL * v29);
          v30 = *v8;
          if ( *v8 == a2 )
            goto LABEL_17;
          ++v31;
        }
        if ( v32 )
          v8 = v32;
      }
      goto LABEL_17;
    }
    goto LABEL_81;
  }
  if ( v5 - *(_DWORD *)(a1 + 228) - v23 <= v5 >> 3 )
  {
    sub_2E48800(v4, v5);
    v47 = *(_DWORD *)(a1 + 232);
    if ( v47 )
    {
      v48 = v47 - 1;
      v49 = *(_QWORD *)(a1 + 216);
      v50 = 0;
      v51 = v48 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v52 = 1;
      v23 = *(_DWORD *)(a1 + 224) + 1;
      v8 = (_QWORD *)(v49 + 16LL * v51);
      v53 = *v8;
      if ( *v8 != a2 )
      {
        while ( v53 != -4096 )
        {
          if ( v53 == -8192 && !v50 )
            v50 = v8;
          v51 = v48 & (v52 + v51);
          v8 = (_QWORD *)(v49 + 16LL * v51);
          v53 = *v8;
          if ( *v8 == a2 )
            goto LABEL_17;
          ++v52;
        }
        if ( v50 )
          v8 = v50;
      }
      goto LABEL_17;
    }
LABEL_81:
    ++*(_DWORD *)(a1 + 224);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 224) = v23;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 228);
  *v8 = a2;
  v12 = 0;
  v8[1] = 0;
LABEL_4:
  v13 = *(_DWORD *)(a1 + 264);
  v14 = a1 + 240;
  if ( !v13 )
  {
    ++*(_QWORD *)(a1 + 240);
    goto LABEL_42;
  }
  v15 = 1;
  v16 = *(_QWORD *)(a1 + 248);
  v17 = 0;
  v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v19 = (__int64 *)(v16 + 16LL * v18);
  v20 = *v19;
  if ( *v19 == v12 )
    return *((unsigned int *)v19 + 2);
  while ( v20 != -4096 )
  {
    if ( !v17 && v20 == -8192 )
      v17 = v19;
    v18 = (v13 - 1) & (v15 + v18);
    v19 = (__int64 *)(v16 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == v12 )
      return *((unsigned int *)v19 + 2);
    ++v15;
  }
  if ( !v17 )
    v17 = v19;
  v24 = *(_DWORD *)(a1 + 256);
  ++*(_QWORD *)(a1 + 240);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v13 )
  {
LABEL_42:
    sub_354C5D0(v14, 2 * v13);
    v33 = *(_DWORD *)(a1 + 264);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 248);
      v36 = (v33 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v25 = *(_DWORD *)(a1 + 256) + 1;
      v17 = (__int64 *)(v35 + 16LL * v36);
      v37 = *v17;
      if ( *v17 != v12 )
      {
        v38 = 1;
        v39 = 0;
        while ( v37 != -4096 )
        {
          if ( !v39 && v37 == -8192 )
            v39 = v17;
          v36 = v34 & (v38 + v36);
          v17 = (__int64 *)(v35 + 16LL * v36);
          v37 = *v17;
          if ( *v17 == v12 )
            goto LABEL_30;
          ++v38;
        }
        if ( v39 )
          v17 = v39;
      }
      goto LABEL_30;
    }
    goto LABEL_82;
  }
  if ( v13 - *(_DWORD *)(a1 + 260) - v25 <= v13 >> 3 )
  {
    sub_354C5D0(v14, v13);
    v40 = *(_DWORD *)(a1 + 264);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 248);
      v43 = 0;
      v44 = v41 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v45 = 1;
      v25 = *(_DWORD *)(a1 + 256) + 1;
      v17 = (__int64 *)(v42 + 16LL * v44);
      v46 = *v17;
      if ( *v17 != v12 )
      {
        while ( v46 != -4096 )
        {
          if ( v46 == -8192 && !v43 )
            v43 = v17;
          v44 = v41 & (v45 + v44);
          v17 = (__int64 *)(v42 + 16LL * v44);
          v46 = *v17;
          if ( *v17 == v12 )
            goto LABEL_30;
          ++v45;
        }
        if ( v43 )
          v17 = v43;
      }
      goto LABEL_30;
    }
LABEL_82:
    ++*(_DWORD *)(a1 + 256);
    BUG();
  }
LABEL_30:
  *(_DWORD *)(a1 + 256) = v25;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 260);
  *v17 = v12;
  *((_DWORD *)v17 + 2) = 0;
  return 0;
}
