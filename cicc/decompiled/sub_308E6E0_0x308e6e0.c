// Function: sub_308E6E0
// Address: 0x308e6e0
//
void __fastcall sub_308E6E0(__int64 a1, int a2)
{
  int v4; // eax
  __int64 v5; // rcx
  int v6; // eax
  unsigned int v7; // edx
  int v8; // esi
  unsigned int v9; // esi
  __int64 v10; // rdi
  unsigned int v11; // ecx
  int *v12; // rdx
  int v13; // eax
  int v14; // r11d
  int *v15; // r10
  int v16; // eax
  int v17; // edx
  int v18; // eax
  int v19; // eax
  __int64 v20; // rsi
  int v21; // r8d
  int *v22; // rdi
  unsigned int v23; // r13d
  int v24; // ecx
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // eax
  int v29; // esi
  int v30; // edi
  int v31; // r9d
  int *v32; // r8

  if ( a2 == -1 )
    return;
  if ( *(_DWORD *)(a1 + 312) )
  {
    v4 = *(_DWORD *)(a1 + 320);
    v5 = *(_QWORD *)(a1 + 304);
    if ( v4 )
    {
      v6 = v4 - 1;
      v7 = v6 & (37 * a2);
      v8 = *(_DWORD *)(v5 + 4LL * v7);
      if ( v8 == a2 )
        return;
      v30 = 1;
      while ( v8 != -1 )
      {
        v7 = v6 & (v30 + v7);
        v8 = *(_DWORD *)(v5 + 4LL * v7);
        if ( v8 == a2 )
          return;
        ++v30;
      }
    }
  }
  sub_3089D70(a1, a2);
  v9 = *(_DWORD *)(a1 + 320);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_22;
  }
  v10 = *(_QWORD *)(a1 + 304);
  v11 = (v9 - 1) & (37 * a2);
  v12 = (int *)(v10 + 4LL * v11);
  v13 = *v12;
  if ( a2 == *v12 )
    return;
  v14 = 1;
  v15 = 0;
  while ( v13 != -1 )
  {
    if ( v13 != -2 || v15 )
      v12 = v15;
    v11 = (v9 - 1) & (v14 + v11);
    v13 = *(_DWORD *)(v10 + 4LL * v11);
    if ( v13 == a2 )
      return;
    ++v14;
    v15 = v12;
    v12 = (int *)(v10 + 4LL * v11);
  }
  v16 = *(_DWORD *)(a1 + 312);
  if ( !v15 )
    v15 = v12;
  ++*(_QWORD *)(a1 + 296);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v9 )
  {
LABEL_22:
    sub_A08C50(a1 + 296, 2 * v9);
    v25 = *(_DWORD *)(a1 + 320);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 304);
      v28 = (v25 - 1) & (37 * a2);
      v15 = (int *)(v27 + 4LL * v28);
      v29 = *v15;
      v17 = *(_DWORD *)(a1 + 312) + 1;
      if ( *v15 != a2 )
      {
        v31 = 1;
        v32 = 0;
        while ( v29 != -1 )
        {
          if ( v29 == -2 && !v32 )
            v32 = v15;
          v28 = v26 & (v31 + v28);
          v15 = (int *)(v27 + 4LL * v28);
          v29 = *v15;
          if ( *v15 == a2 )
            goto LABEL_24;
          ++v31;
        }
        if ( v32 )
          v15 = v32;
      }
      goto LABEL_24;
    }
    goto LABEL_51;
  }
  if ( v9 - *(_DWORD *)(a1 + 316) - v17 <= v9 >> 3 )
  {
    sub_A08C50(a1 + 296, v9);
    v18 = *(_DWORD *)(a1 + 320);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 304);
      v21 = 1;
      v22 = 0;
      v23 = v19 & (37 * a2);
      v15 = (int *)(v20 + 4LL * v23);
      v24 = *v15;
      v17 = *(_DWORD *)(a1 + 312) + 1;
      if ( *v15 != a2 )
      {
        while ( v24 != -1 )
        {
          if ( !v22 && v24 == -2 )
            v22 = v15;
          v23 = v19 & (v21 + v23);
          v15 = (int *)(v20 + 4LL * v23);
          v24 = *v15;
          if ( a2 == *v15 )
            goto LABEL_24;
          ++v21;
        }
        if ( v22 )
          v15 = v22;
      }
      goto LABEL_24;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 312);
    BUG();
  }
LABEL_24:
  *(_DWORD *)(a1 + 312) = v17;
  if ( *v15 != -1 )
    --*(_DWORD *)(a1 + 316);
  *v15 = a2;
}
