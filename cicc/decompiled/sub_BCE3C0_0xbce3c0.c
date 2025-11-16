// Function: sub_BCE3C0
// Address: 0xbce3c0
//
__int64 __fastcall sub_BCE3C0(__int64 *a1, int a2)
{
  __int64 v4; // rbx
  __int64 result; // rax
  __int64 *v6; // r14
  unsigned int v7; // esi
  __int64 v8; // rdi
  __int64 v9; // r8
  int *v10; // r14
  int v11; // r10d
  unsigned int v12; // ecx
  __int64 v13; // rax
  int v14; // edx
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  int v21; // esi
  int v22; // r9d
  int *v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int *v27; // rdi
  unsigned int v28; // r15d
  int v29; // r8d
  int v30; // ecx
  __int64 v31; // [rsp+8h] [rbp-38h]

  v4 = *a1;
  if ( !a2 )
  {
    result = *(_QWORD *)(v4 + 3096);
    v6 = (__int64 *)(v4 + 3096);
    goto LABEL_3;
  }
  v7 = *(_DWORD *)(v4 + 3128);
  v8 = v4 + 3104;
  if ( !v7 )
  {
    ++*(_QWORD *)(v4 + 3104);
    goto LABEL_25;
  }
  v9 = *(_QWORD *)(v4 + 3112);
  v10 = 0;
  v11 = 1;
  v12 = (v7 - 1) & (37 * a2);
  v13 = v9 + 16LL * v12;
  v14 = *(_DWORD *)v13;
  if ( *(_DWORD *)v13 == a2 )
  {
LABEL_7:
    v6 = (__int64 *)(v13 + 8);
    result = *(_QWORD *)(v13 + 8);
LABEL_3:
    if ( result )
      return result;
    goto LABEL_21;
  }
  while ( v14 != -1 )
  {
    if ( v14 == -2 && !v10 )
      v10 = (int *)v13;
    v12 = (v7 - 1) & (v11 + v12);
    v13 = v9 + 16LL * v12;
    v14 = *(_DWORD *)v13;
    if ( *(_DWORD *)v13 == a2 )
      goto LABEL_7;
    ++v11;
  }
  if ( !v10 )
    v10 = (int *)v13;
  v15 = *(_DWORD *)(v4 + 3120);
  ++*(_QWORD *)(v4 + 3104);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v7 )
  {
LABEL_25:
    sub_BCE1E0(v8, 2 * v7);
    v17 = *(_DWORD *)(v4 + 3128);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v4 + 3112);
      v20 = (v17 - 1) & (37 * a2);
      v16 = *(_DWORD *)(v4 + 3120) + 1;
      v10 = (int *)(v19 + 16LL * v20);
      v21 = *v10;
      if ( *v10 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -1 )
        {
          if ( !v23 && v21 == -2 )
            v23 = v10;
          v20 = v18 & (v22 + v20);
          v10 = (int *)(v19 + 16LL * v20);
          v21 = *v10;
          if ( *v10 == a2 )
            goto LABEL_18;
          ++v22;
        }
        if ( v23 )
          v10 = v23;
      }
      goto LABEL_18;
    }
    goto LABEL_48;
  }
  if ( v7 - *(_DWORD *)(v4 + 3124) - v16 <= v7 >> 3 )
  {
    sub_BCE1E0(v8, v7);
    v24 = *(_DWORD *)(v4 + 3128);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v4 + 3112);
      v27 = 0;
      v28 = v25 & (37 * a2);
      v29 = 1;
      v16 = *(_DWORD *)(v4 + 3120) + 1;
      v10 = (int *)(v26 + 16LL * v28);
      v30 = *v10;
      if ( *v10 != a2 )
      {
        while ( v30 != -1 )
        {
          if ( v30 == -2 && !v27 )
            v27 = v10;
          v28 = v25 & (v29 + v28);
          v10 = (int *)(v26 + 16LL * v28);
          v30 = *v10;
          if ( *v10 == a2 )
            goto LABEL_18;
          ++v29;
        }
        if ( v27 )
          v10 = v27;
      }
      goto LABEL_18;
    }
LABEL_48:
    ++*(_DWORD *)(v4 + 3120);
    BUG();
  }
LABEL_18:
  *(_DWORD *)(v4 + 3120) = v16;
  if ( *v10 != -1 )
    --*(_DWORD *)(v4 + 3124);
  *v10 = a2;
  v6 = (__int64 *)(v10 + 2);
  *v6 = 0;
LABEL_21:
  result = sub_A777F0(0x18u, (__int64 *)(v4 + 2640));
  if ( result )
  {
    v31 = result;
    sub_BCBCF0(result, (__int64)a1, a2);
    result = v31;
  }
  *v6 = result;
  return result;
}
