// Function: sub_2F5FA90
// Address: 0x2f5fa90
//
__int64 __fastcall sub_2F5FA90(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int *v9; // r14
  int v10; // r11d
  unsigned int v11; // edx
  int *v12; // rax
  int v13; // r10d
  __int64 result; // rax
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  unsigned int v21; // esi
  int v22; // r9d
  unsigned int *v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  unsigned int v28; // r15d
  unsigned int *v29; // rdi
  unsigned int v30; // ecx

  v6 = a1 + 168;
  v7 = *(_DWORD *)(a1 + 192);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 168);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(a1 + 176);
  v9 = 0;
  v10 = 1;
  v11 = (v7 - 1) & (37 * a2);
  v12 = (int *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a2 )
    return *((_QWORD *)v12 + 1);
  while ( v13 != -1 )
  {
    if ( !v9 && v13 == -2 )
      v9 = (unsigned int *)v12;
    v11 = (v7 - 1) & (v10 + v11);
    v12 = (int *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a2 )
      return *((_QWORD *)v12 + 1);
    ++v10;
  }
  if ( !v9 )
    v9 = (unsigned int *)v12;
  v15 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v7 )
  {
LABEL_18:
    sub_2F5F8B0(v6, 2 * v7);
    v17 = *(_DWORD *)(a1 + 192);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 176);
      v20 = (v17 - 1) & (37 * a2);
      v16 = *(_DWORD *)(a1 + 184) + 1;
      v9 = (unsigned int *)(v19 + 16LL * v20);
      v21 = *v9;
      if ( *v9 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -1 )
        {
          if ( !v23 && v21 == -2 )
            v23 = v9;
          v20 = v18 & (v22 + v20);
          v9 = (unsigned int *)(v19 + 16LL * v20);
          v21 = *v9;
          if ( *v9 == a2 )
            goto LABEL_14;
          ++v22;
        }
        if ( v23 )
          v9 = v23;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v7 - *(_DWORD *)(a1 + 188) - v16 <= v7 >> 3 )
  {
    sub_2F5F8B0(v6, v7);
    v24 = *(_DWORD *)(a1 + 192);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 176);
      v27 = 1;
      v28 = v25 & (37 * a2);
      v16 = *(_DWORD *)(a1 + 184) + 1;
      v29 = 0;
      v9 = (unsigned int *)(v26 + 16LL * v28);
      v30 = *v9;
      if ( a2 != *v9 )
      {
        while ( v30 != -1 )
        {
          if ( v30 == -2 && !v29 )
            v29 = v9;
          v28 = v25 & (v27 + v28);
          v9 = (unsigned int *)(v26 + 16LL * v28);
          v30 = *v9;
          if ( *v9 == a2 )
            goto LABEL_14;
          ++v27;
        }
        if ( v29 )
          v9 = v29;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 184);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 184) = v16;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 188);
  *v9 = a2;
  *((_QWORD *)v9 + 1) = 0;
  result = sub_2FF6620(a3, a2);
  *((_QWORD *)v9 + 1) = result;
  return result;
}
