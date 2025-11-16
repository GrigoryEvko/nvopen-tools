// Function: sub_13E7760
// Address: 0x13e7760
//
__int64 __fastcall sub_13E7760(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rax
  int v11; // r15d
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // esi
  int v16; // eax
  __int64 v17; // rdi
  unsigned int v18; // esi
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v2 = sub_157EB90(a2);
  v3 = sub_1632FA0(v2);
  v4 = sub_14AD280(a1, v3, 6);
  if ( v4 != sub_14AD280(v4, v3, 1) )
    return 0;
  v6 = *(_QWORD *)(a2 + 48);
  v7 = a2 + 40;
  if ( a2 + 40 == v6 )
    return 0;
  while ( 1 )
  {
    if ( !v6 )
      BUG();
    v9 = *(_BYTE *)(v6 - 8);
    if ( v9 == 54 || v9 == 55 )
    {
      v8 = **(_QWORD **)(v6 - 48);
      if ( *(_BYTE *)(v8 + 8) == 16 )
        v8 = **(_QWORD **)(v8 + 16);
      if ( !(*(_DWORD *)(v8 + 8) >> 8) )
      {
        v22 = sub_15F2050(v6 - 24);
        v23 = sub_1632FA0(v22);
        if ( v4 == sub_14AD280(*(_QWORD *)(v6 - 48), v23, 6) )
          return 1;
      }
      goto LABEL_8;
    }
    if ( v9 != 78 )
      goto LABEL_8;
    v10 = *(_QWORD *)(v6 - 48);
    if ( *(_BYTE *)(v10 + 16) )
      goto LABEL_8;
    if ( (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
      goto LABEL_8;
    v11 = *(_DWORD *)(v10 + 36);
    if ( (unsigned int)(v11 - 133) > 4 || ((1LL << ((unsigned __int8)v11 + 123)) & 0x15) == 0 )
      goto LABEL_8;
    v12 = v6 - 24;
    v13 = *(_DWORD *)(v6 - 4) & 0xFFFFFFF;
    v14 = *(_QWORD *)(v6 - 24 + 24 * (3 - v13));
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 <= 0x40 )
    {
      if ( *(_QWORD *)(v14 + 24) )
        goto LABEL_8;
    }
    else
    {
      v27 = *(_DWORD *)(v6 - 4) & 0xFFFFFFF;
      v16 = sub_16A57B0(v14 + 24);
      v13 = v27;
      if ( v15 != v16 )
        goto LABEL_8;
    }
    v17 = *(_QWORD *)(v12 + 24 * (2 - v13));
    if ( *(_BYTE *)(v17 + 16) == 13 )
    {
      v18 = *(_DWORD *)(v17 + 32);
      if ( v18 <= 0x40 )
      {
        if ( !*(_QWORD *)(v17 + 24) )
          goto LABEL_8;
      }
      else
      {
        v28 = v13;
        v19 = sub_16A57B0(v17 + 24);
        v13 = v28;
        if ( v18 == v19 )
          goto LABEL_8;
      }
      if ( !(*(_DWORD *)(**(_QWORD **)(v12 - 24 * v13) + 8LL) >> 8) )
      {
        v24 = sub_15F2050(v6 - 24);
        v25 = sub_1632FA0(v24);
        if ( v4 == sub_14AD280(*(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v6 - 4) & 0xFFFFFFF)), v25, 6) )
          return 1;
        v26 = *(_QWORD *)(v6 - 48);
        if ( *(_BYTE *)(v26 + 16) )
          BUG();
        v11 = *(_DWORD *)(v26 + 36);
      }
      if ( (v11 & 0xFFFFFFFD) == 0x85
        && !(*(_DWORD *)(**(_QWORD **)(v12 + 24 * (1LL - (*(_DWORD *)(v6 - 4) & 0xFFFFFFF))) + 8LL) >> 8) )
      {
        v20 = sub_15F2050(v6 - 24);
        v21 = sub_1632FA0(v20);
        if ( v4 == sub_14AD280(*(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v6 - 4) & 0xFFFFFFF))), v21, 6) )
          return 1;
      }
    }
LABEL_8:
    v6 = *(_QWORD *)(v6 + 8);
    if ( v7 == v6 )
      return 0;
  }
}
