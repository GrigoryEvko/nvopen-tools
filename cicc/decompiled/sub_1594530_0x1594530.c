// Function: sub_1594530
// Address: 0x1594530
//
__int64 __fastcall sub_1594530(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // r15
  _QWORD *v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rbx
  __int64 v9; // r13
  char v10; // dl
  unsigned int v11; // edx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  char v14; // dl
  unsigned __int64 v15; // rdi
  __int64 v16; // [rsp-40h] [rbp-40h]
  __int64 v17; // [rsp-40h] [rbp-40h]

  if ( *(_WORD *)(a1 + 18) != 32 )
    return 0;
  v2 = (_QWORD *)a1;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v4 = (_QWORD *)(v3 + 24);
  v5 = sub_16348C0(a1) | 4;
  v6 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v2 = (_QWORD *)(v6 + *(_QWORD *)(a1 - 8));
  v7 = a1 - v6 + 24;
  v8 = -1;
  if ( v4 == v2 )
    return 1;
  while ( 1 )
  {
    v9 = *(_QWORD *)v7;
    v10 = *(_BYTE *)(*(_QWORD *)v7 + 16LL);
    if ( v10 == 9 )
    {
      if ( (v5 & 4) == 0 )
        goto LABEL_26;
      goto LABEL_15;
    }
    if ( v10 != 13 )
      return 0;
    if ( (v5 & 4) == 0 )
    {
LABEL_26:
      v15 = v5 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_27:
      v16 = v7;
      v13 = sub_1643D30(v15, *v4);
      v7 = v16;
      goto LABEL_16;
    }
    if ( v8 != -1 )
      break;
LABEL_15:
    v13 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v15 = 0;
      goto LABEL_27;
    }
LABEL_16:
    v14 = *(_BYTE *)(v13 + 8);
    if ( ((v14 - 14) & 0xFD) != 0 )
    {
      v5 = 0;
      if ( v14 == 13 )
        v5 = v13;
    }
    else
    {
      v8 = *(_QWORD *)(v13 + 32);
      v5 = *(_QWORD *)(v13 + 24) | 4LL;
    }
    v4 += 3;
    v7 += 24;
    if ( v2 == v4 )
      return 1;
  }
  v11 = *(_DWORD *)(v9 + 32);
  if ( v11 <= 0x40 )
  {
    v12 = *(_QWORD *)(v9 + 24);
    goto LABEL_14;
  }
  v17 = v7;
  if ( v11 - (unsigned int)sub_16A57B0(v9 + 24) <= 0x40 )
  {
    v7 = v17;
    v12 = **(_QWORD **)(v9 + 24);
LABEL_14:
    if ( v12 >= v8 )
      return 0;
    goto LABEL_15;
  }
  return 0;
}
