// Function: sub_1C7E8C0
// Address: 0x1c7e8c0
//
char __fastcall sub_1C7E8C0(__int64 a1, __int64 a2, int a3, __int64 a4, _BYTE *a5)
{
  char v6; // al
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v11; // rax
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rax
  char v27; // al
  __int64 v28; // rdi
  _QWORD v29[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 != 55 )
  {
    if ( v6 != 78 )
      return 0;
    v11 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v11 + 16) && (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
    {
      v12 = *(_DWORD *)(v11 + 36);
      if ( sub_1C301F0(v12) || (unsigned __int8)sub_1C30260(v12) )
        return a3 != 5;
    }
    if ( (unsigned __int8)sub_1560260((_QWORD *)(a2 + 56), -1, 36) )
      return 0;
    if ( *(char *)(a2 + 23) >= 0 )
      goto LABEL_49;
    v13 = sub_1648A40(a2);
    v15 = v13 + v14;
    v16 = 0;
    if ( *(char *)(a2 + 23) < 0 )
      v16 = sub_1648A40(a2);
    if ( !(unsigned int)((v15 - v16) >> 4) )
    {
LABEL_49:
      v17 = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(v17 + 16) )
      {
        v29[0] = *(_QWORD *)(v17 + 112);
        if ( (unsigned __int8)sub_1560260(v29, -1, 36) )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560260((_QWORD *)(a2 + 56), -1, 36) )
      return 0;
    if ( *(char *)(a2 + 23) >= 0 )
      goto LABEL_50;
    v18 = sub_1648A40(a2);
    v20 = v18 + v19;
    v21 = 0;
    if ( *(char *)(a2 + 23) < 0 )
      v21 = sub_1648A40(a2);
    if ( !(unsigned int)((v20 - v21) >> 4) )
    {
LABEL_50:
      v22 = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(v22 + 16) )
      {
        v29[0] = *(_QWORD *)(v22 + 112);
        if ( (unsigned __int8)sub_1560260(v29, -1, 36) )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560260((_QWORD *)(a2 + 56), -1, 37) )
      return 0;
    if ( *(char *)(a2 + 23) >= 0
      || ((v23 = sub_1648A40(a2), v25 = v23 + v24, *(char *)(a2 + 23) >= 0) ? (v26 = 0) : (v26 = sub_1648A40(a2)),
          v26 == v25) )
    {
LABEL_43:
      v28 = *(_QWORD *)(a2 - 24);
      v27 = *(_BYTE *)(v28 + 16);
      if ( v27 )
      {
LABEL_40:
        if ( v27 == 20 )
          return sub_1CCC8B0();
        return 1;
      }
      v29[0] = *(_QWORD *)(v28 + 112);
      if ( (unsigned __int8)sub_1560260(v29, -1, 37) )
        return 0;
    }
    else
    {
      while ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) <= 1u )
      {
        v26 += 16;
        if ( v25 == v26 )
          goto LABEL_43;
      }
    }
    v27 = *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL);
    goto LABEL_40;
  }
  v7 = *(_QWORD **)a4;
  v8 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( v8 != *(_QWORD *)a4 )
  {
    do
    {
      if ( *v7 == a2 )
        *a5 = 1;
      ++v7;
    }
    while ( (_QWORD *)v8 != v7 );
  }
  if ( !a3 )
    return 1;
  v9 = **(_QWORD **)(a2 - 24);
  if ( *(_BYTE *)(v9 + 8) == 16 )
    v9 = **(_QWORD **)(v9 + 16);
  return *(_DWORD *)(v9 + 8) >> 8 == 0 || a3 == *(_DWORD *)(v9 + 8) >> 8;
}
