// Function: sub_1910080
// Address: 0x1910080
//
__int64 __fastcall sub_1910080(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // ecx
  int *v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  int *v14; // rax
  int v16; // r15d
  int *v17; // r10
  int v18; // edi
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // edx
  int v24; // edi
  int v25; // r10d
  int *v26; // r9
  int v27; // eax
  int v28; // edx
  __int64 v29; // r8
  __int64 v30; // r14
  int v31; // edi
  int v32; // esi
  int *v33; // r11

  v4 = a4 + 376;
  v8 = *(_DWORD *)(a4 + 400);
  if ( !v8 )
  {
    ++*(_QWORD *)(a4 + 376);
    goto LABEL_20;
  }
  v9 = *(_QWORD *)(a4 + 384);
  v10 = (v8 - 1) & (37 * a2);
  v11 = (int *)(v9 + 40LL * v10);
  v12 = *v11;
  if ( *v11 != a2 )
  {
    v16 = 1;
    v17 = 0;
    while ( v12 != -1 )
    {
      if ( v17 || v12 != -2 )
        v11 = v17;
      v10 = (v8 - 1) & (v16 + v10);
      v33 = (int *)(v9 + 40LL * v10);
      v12 = *v33;
      if ( *v33 == a2 )
      {
        v13 = *((_QWORD *)v33 + 2);
        v11 = (int *)(v9 + 40LL * v10);
        goto LABEL_4;
      }
      ++v16;
      v17 = v11;
      v11 = (int *)(v9 + 40LL * v10);
    }
    v18 = *(_DWORD *)(a4 + 392);
    if ( v17 )
      v11 = v17;
    ++*(_QWORD *)(a4 + 376);
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a4 + 396) - v19 > v8 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(a4 + 392) = v19;
        if ( *v11 != -1 )
          --*(_DWORD *)(a4 + 396);
        *v11 = a2;
        v13 = 0;
        *((_QWORD *)v11 + 1) = 0;
        *((_QWORD *)v11 + 2) = 0;
        *((_QWORD *)v11 + 3) = 0;
        *((_QWORD *)v11 + 4) = 0;
        goto LABEL_4;
      }
      sub_190FC70(v4, v8);
      v27 = *(_DWORD *)(a4 + 400);
      if ( v27 )
      {
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a4 + 384);
        v26 = 0;
        LODWORD(v30) = (v27 - 1) & (37 * a2);
        v19 = *(_DWORD *)(a4 + 392) + 1;
        v31 = 1;
        v11 = (int *)(v29 + 40LL * (unsigned int)v30);
        v32 = *v11;
        if ( *v11 == a2 )
          goto LABEL_16;
        while ( v32 != -1 )
        {
          if ( v32 == -2 && !v26 )
            v26 = v11;
          v30 = v28 & (unsigned int)(v30 + v31);
          v11 = (int *)(v29 + 40 * v30);
          v32 = *v11;
          if ( *v11 == a2 )
            goto LABEL_16;
          ++v31;
        }
        goto LABEL_24;
      }
      goto LABEL_46;
    }
LABEL_20:
    sub_190FC70(v4, 2 * v8);
    v20 = *(_DWORD *)(a4 + 400);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a4 + 384);
      v23 = (v20 - 1) & (37 * a2);
      v19 = *(_DWORD *)(a4 + 392) + 1;
      v11 = (int *)(v22 + 40LL * v23);
      v24 = *v11;
      if ( *v11 == a2 )
        goto LABEL_16;
      v25 = 1;
      v26 = 0;
      while ( v24 != -1 )
      {
        if ( !v26 && v24 == -2 )
          v26 = v11;
        v23 = v21 & (v25 + v23);
        v11 = (int *)(v22 + 40LL * v23);
        v24 = *v11;
        if ( *v11 == a2 )
          goto LABEL_16;
        ++v25;
      }
LABEL_24:
      if ( v26 )
        v11 = v26;
      goto LABEL_16;
    }
LABEL_46:
    ++*(_DWORD *)(a4 + 392);
    BUG();
  }
  v13 = *((_QWORD *)v11 + 2);
LABEL_4:
  v14 = v11 + 2;
  while ( a3 == v13 )
  {
    v14 = (int *)*((_QWORD *)v14 + 2);
    if ( !v14 )
      return 1;
    v13 = *((_QWORD *)v14 + 1);
  }
  return 0;
}
