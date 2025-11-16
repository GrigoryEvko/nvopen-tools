// Function: sub_D85100
// Address: 0xd85100
//
__int64 __fastcall sub_D85100(__int64 a1, int a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  int *v7; // r13
  int v8; // r11d
  unsigned int v9; // edx
  int *v10; // rax
  int v11; // r9d
  int v13; // eax
  int v14; // ecx
  __int64 v15; // rdi
  unsigned int v16; // eax
  int v17; // edx
  int v18; // esi
  __int64 v19; // rax
  int v20; // eax
  int v21; // eax
  int v22; // eax
  __int64 v23; // rsi
  int v24; // r8d
  unsigned int v25; // r14d
  int *v26; // rdi
  int v27; // ecx
  int v28; // r9d
  int *v29; // r8
  __int64 v30; // [rsp+10h] [rbp-30h]

  if ( !*(_QWORD *)(a1 + 8) )
    return v30;
  v4 = a1 + 56;
  v5 = *(_DWORD *)(a1 + 80);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_8;
  }
  v6 = *(_QWORD *)(a1 + 64);
  v7 = 0;
  v8 = 1;
  v9 = (v5 - 1) & (37 * a2);
  v10 = (int *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    return *((_QWORD *)v10 + 1);
  while ( v11 != 0x7FFFFFFF )
  {
    if ( !v7 && v11 == 0x80000000 )
      v7 = v10;
    v9 = (v5 - 1) & (v8 + v9);
    v10 = (int *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      return *((_QWORD *)v10 + 1);
    ++v8;
  }
  if ( !v7 )
    v7 = v10;
  v20 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  v17 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v5 )
  {
LABEL_8:
    sub_D84F20(v4, 2 * v5);
    v13 = *(_DWORD *)(a1 + 80);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 64);
      v16 = (v13 - 1) & (37 * a2);
      v17 = *(_DWORD *)(a1 + 72) + 1;
      v7 = (int *)(v15 + 16LL * v16);
      v18 = *v7;
      if ( *v7 != a2 )
      {
        v28 = 1;
        v29 = 0;
        while ( v18 != 0x7FFFFFFF )
        {
          if ( !v29 && v18 == 0x80000000 )
            v29 = v7;
          v16 = v14 & (v28 + v16);
          v7 = (int *)(v15 + 16LL * v16);
          v18 = *v7;
          if ( *v7 == a2 )
            goto LABEL_10;
          ++v28;
        }
        if ( v29 )
          v7 = v29;
      }
      goto LABEL_10;
    }
    goto LABEL_44;
  }
  if ( v5 - *(_DWORD *)(a1 + 76) - v17 <= v5 >> 3 )
  {
    sub_D84F20(v4, v5);
    v21 = *(_DWORD *)(a1 + 80);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 64);
      v24 = 1;
      v25 = v22 & (37 * a2);
      v26 = 0;
      v17 = *(_DWORD *)(a1 + 72) + 1;
      v7 = (int *)(v23 + 16LL * v25);
      v27 = *v7;
      if ( *v7 != a2 )
      {
        while ( v27 != 0x7FFFFFFF )
        {
          if ( v27 == 0x80000000 && !v26 )
            v26 = v7;
          v25 = v22 & (v24 + v25);
          v7 = (int *)(v23 + 16LL * v25);
          v27 = *v7;
          if ( *v7 == a2 )
            goto LABEL_10;
          ++v24;
        }
        if ( v26 )
          v7 = v26;
      }
      goto LABEL_10;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 72);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 72) = v17;
  if ( *v7 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 76);
  *v7 = a2;
  *((_QWORD *)v7 + 1) = 0;
  v19 = *(_QWORD *)(sub_EF9A70(*(_QWORD *)(a1 + 8) + 8LL, a2) + 8);
  *((_QWORD *)v7 + 1) = v19;
  return v19;
}
