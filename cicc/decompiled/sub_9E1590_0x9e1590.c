// Function: sub_9E1590
// Address: 0x9e1590
//
__int64 __fastcall sub_9E1590(__int64 a1, int a2)
{
  __int64 v2; // r9
  unsigned int v4; // esi
  int v6; // r11d
  int *v7; // rdx
  __int64 v8; // r8
  unsigned int v9; // edi
  int *v10; // rax
  int v11; // ecx
  int v13; // eax
  int v14; // ecx
  int v15; // eax
  int v16; // eax
  __int64 v17; // r8
  unsigned int v18; // esi
  int v19; // edi
  int v20; // r10d
  int *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  int *v25; // r8
  unsigned int v26; // r13d
  int v27; // r9d
  int v28; // esi

  v2 = a1 + 448;
  v4 = *(_DWORD *)(a1 + 472);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 448);
    goto LABEL_18;
  }
  v6 = 1;
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 456);
  v9 = (v4 - 1) & (37 * a2);
  v10 = (int *)(v8 + 24LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    return *((_QWORD *)v10 + 1);
  while ( v11 != -1 )
  {
    if ( !v7 && v11 == -2 )
      v7 = v10;
    v9 = (v4 - 1) & (v6 + v9);
    v10 = (int *)(v8 + 24LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      return *((_QWORD *)v10 + 1);
    ++v6;
  }
  if ( !v7 )
    v7 = v10;
  v13 = *(_DWORD *)(a1 + 464);
  ++*(_QWORD *)(a1 + 448);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_9E0BD0(v2, 2 * v4);
    v15 = *(_DWORD *)(a1 + 472);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 456);
      v18 = v16 & (37 * a2);
      v14 = *(_DWORD *)(a1 + 464) + 1;
      v7 = (int *)(v17 + 24LL * v18);
      v19 = *v7;
      if ( *v7 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -1 )
        {
          if ( !v21 && v19 == -2 )
            v21 = v7;
          v18 = v16 & (v20 + v18);
          v7 = (int *)(v17 + 24LL * v18);
          v19 = *v7;
          if ( *v7 == a2 )
            goto LABEL_14;
          ++v20;
        }
        if ( v21 )
          v7 = v21;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v4 - *(_DWORD *)(a1 + 468) - v14 <= v4 >> 3 )
  {
    sub_9E0BD0(v2, v4);
    v22 = *(_DWORD *)(a1 + 472);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 456);
      v25 = 0;
      v26 = v23 & (37 * a2);
      v27 = 1;
      v14 = *(_DWORD *)(a1 + 464) + 1;
      v7 = (int *)(v24 + 24LL * v26);
      v28 = *v7;
      if ( *v7 != a2 )
      {
        while ( v28 != -1 )
        {
          if ( !v25 && v28 == -2 )
            v25 = v7;
          v26 = v23 & (v27 + v26);
          v7 = (int *)(v24 + 24LL * v26);
          v28 = *v7;
          if ( *v7 == a2 )
            goto LABEL_14;
          ++v27;
        }
        if ( v25 )
          v7 = v25;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 464);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 464) = v14;
  if ( *v7 != -1 )
    --*(_DWORD *)(a1 + 468);
  *v7 = a2;
  *((_QWORD *)v7 + 1) = 0;
  *((_QWORD *)v7 + 2) = 0;
  return *((_QWORD *)v7 + 1);
}
