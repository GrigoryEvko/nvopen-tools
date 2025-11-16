// Function: sub_E70580
// Address: 0xe70580
//
__int64 __fastcall sub_E70580(__int64 a1, int a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r14d
  int *v7; // rdx
  __int64 v8; // r8
  unsigned int v9; // ecx
  int *v10; // rax
  int v11; // r10d
  _QWORD *v12; // r12
  unsigned int *v13; // rax
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdx
  unsigned int *v18; // rax
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // eax
  int v23; // edi
  int v24; // r10d
  int *v25; // r9
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  int *v29; // r8
  unsigned int v30; // r13d
  int v31; // r9d
  int v32; // esi

  v4 = a1 + 1440;
  v5 = *(_DWORD *)(a1 + 1464);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 1440);
    goto LABEL_23;
  }
  v6 = 1;
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 1448);
  v9 = (v5 - 1) & (37 * a2);
  v10 = (int *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    while ( v11 != -1 )
    {
      if ( !v7 && v11 == -2 )
        v7 = v10;
      v9 = (v5 - 1) & (v6 + v9);
      v10 = (int *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_3;
      ++v6;
    }
    if ( !v7 )
      v7 = v10;
    v15 = *(_DWORD *)(a1 + 1456);
    ++*(_QWORD *)(a1 + 1440);
    v16 = v15 + 1;
    if ( 4 * (v15 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 1460) - v16 > v5 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 1456) = v16;
        if ( *v7 != -1 )
          --*(_DWORD *)(a1 + 1460);
        *((_QWORD *)v7 + 1) = 0;
        *v7 = a2;
        v12 = v7 + 2;
        goto LABEL_18;
      }
      sub_E700D0(v4, v5);
      v26 = *(_DWORD *)(a1 + 1464);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a1 + 1448);
        v29 = 0;
        v30 = v27 & (37 * a2);
        v31 = 1;
        v16 = *(_DWORD *)(a1 + 1456) + 1;
        v7 = (int *)(v28 + 16LL * v30);
        v32 = *v7;
        if ( *v7 != a2 )
        {
          while ( v32 != -1 )
          {
            if ( !v29 && v32 == -2 )
              v29 = v7;
            v30 = v27 & (v31 + v30);
            v7 = (int *)(v28 + 16LL * v30);
            v32 = *v7;
            if ( *v7 == a2 )
              goto LABEL_15;
            ++v31;
          }
          if ( v29 )
            v7 = v29;
        }
        goto LABEL_15;
      }
LABEL_47:
      ++*(_DWORD *)(a1 + 1456);
      BUG();
    }
LABEL_23:
    sub_E700D0(v4, 2 * v5);
    v19 = *(_DWORD *)(a1 + 1464);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 1448);
      v22 = (v19 - 1) & (37 * a2);
      v16 = *(_DWORD *)(a1 + 1456) + 1;
      v7 = (int *)(v21 + 16LL * v22);
      v23 = *v7;
      if ( *v7 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -1 )
        {
          if ( !v25 && v23 == -2 )
            v25 = v7;
          v22 = v20 & (v24 + v22);
          v7 = (int *)(v21 + 16LL * v22);
          v23 = *v7;
          if ( *v7 == a2 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v7 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
LABEL_3:
  v12 = v10 + 2;
  v13 = (unsigned int *)*((_QWORD *)v10 + 1);
  if ( v13 )
    return *v13;
LABEL_18:
  v17 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 272) += 4LL;
  v18 = (unsigned int *)((v17 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a1 + 200) >= (unsigned __int64)(v18 + 1) && v17 )
    *(_QWORD *)(a1 + 192) = v18 + 1;
  else
    v18 = (unsigned int *)sub_9D1E70(a1 + 192, 4, 4, 3);
  *v18 = 0;
  *v12 = v18;
  return *v18;
}
