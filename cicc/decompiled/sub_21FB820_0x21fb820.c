// Function: sub_21FB820
// Address: 0x21fb820
//
void __fastcall sub_21FB820(__int64 a1, int a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // ecx
  int *v7; // rdx
  int v8; // eax
  int v9; // eax
  int v10; // eax
  __int64 v11; // rsi
  unsigned int v12; // edx
  int v13; // ecx
  int v14; // edi
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  int *v19; // r10
  int v20; // esi
  int v21; // edx
  int v22; // r11d
  int v23; // eax
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  int *v28; // rdi
  unsigned int v29; // r13d
  int v30; // ecx
  int v31; // r9d
  int *v32; // r8

  if ( a2 == -1 )
    return;
  if ( *(_DWORD *)(a1 + 344) )
  {
    v9 = *(_DWORD *)(a1 + 352);
    if ( v9 )
    {
      v10 = v9 - 1;
      v11 = *(_QWORD *)(a1 + 336);
      v12 = v10 & (37 * a2);
      v13 = *(_DWORD *)(v11 + 4LL * v12);
      if ( v13 == a2 )
        return;
      v14 = 1;
      while ( v13 != -1 )
      {
        v12 = v10 & (v14 + v12);
        v13 = *(_DWORD *)(v11 + 4LL * v12);
        if ( v13 == a2 )
          return;
        ++v14;
      }
    }
  }
  sub_21F8CF0(a1, a2);
  v4 = *(_DWORD *)(a1 + 352);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 328);
    goto LABEL_13;
  }
  v5 = *(_QWORD *)(a1 + 336);
  v6 = (v4 - 1) & (37 * a2);
  v7 = (int *)(v5 + 4LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
    return;
  v22 = 1;
  v19 = 0;
  while ( v8 != -1 )
  {
    if ( v8 == -2 && !v19 )
      v19 = v7;
    v6 = (v4 - 1) & (v22 + v6);
    v7 = (int *)(v5 + 4LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
      return;
    ++v22;
  }
  v23 = *(_DWORD *)(a1 + 344);
  if ( !v19 )
    v19 = v7;
  ++*(_QWORD *)(a1 + 328);
  v21 = v23 + 1;
  if ( 4 * (v23 + 1) >= 3 * v4 )
  {
LABEL_13:
    sub_136B240(a1 + 328, 2 * v4);
    v15 = *(_DWORD *)(a1 + 352);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 336);
      v18 = (v15 - 1) & (37 * a2);
      v19 = (int *)(v17 + 4LL * v18);
      v20 = *v19;
      v21 = *(_DWORD *)(a1 + 344) + 1;
      if ( a2 != *v19 )
      {
        v31 = 1;
        v32 = 0;
        while ( v20 != -1 )
        {
          if ( v20 == -2 && !v32 )
            v32 = v19;
          v18 = v16 & (v31 + v18);
          v19 = (int *)(v17 + 4LL * v18);
          v20 = *v19;
          if ( *v19 == a2 )
            goto LABEL_15;
          ++v31;
        }
        if ( v32 )
          v19 = v32;
      }
      goto LABEL_15;
    }
    goto LABEL_50;
  }
  if ( v4 - *(_DWORD *)(a1 + 348) - v21 <= v4 >> 3 )
  {
    sub_136B240(a1 + 328, v4);
    v24 = *(_DWORD *)(a1 + 352);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 336);
      v27 = 1;
      v28 = 0;
      v29 = v25 & (37 * a2);
      v19 = (int *)(v26 + 4LL * v29);
      v30 = *v19;
      v21 = *(_DWORD *)(a1 + 344) + 1;
      if ( *v19 != a2 )
      {
        while ( v30 != -1 )
        {
          if ( v30 == -2 && !v28 )
            v28 = v19;
          v29 = v25 & (v27 + v29);
          v19 = (int *)(v26 + 4LL * v29);
          v30 = *v19;
          if ( a2 == *v19 )
            goto LABEL_15;
          ++v27;
        }
        if ( v28 )
          v19 = v28;
      }
      goto LABEL_15;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 344);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 344) = v21;
  if ( *v19 != -1 )
    --*(_DWORD *)(a1 + 348);
  *v19 = a2;
}
