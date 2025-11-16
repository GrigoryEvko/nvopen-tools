// Function: sub_161C790
// Address: 0x161c790
//
__int64 __fastcall sub_161C790(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // r15
  bool v8; // cl
  __int64 v9; // rax
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // r8
  int v16; // [rsp+8h] [rbp-38h]
  bool v17; // [rsp+Fh] [rbp-31h]

  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 8 * (1 - v4));
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (2 - v4)) + 136LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = (unsigned __int8)(**(_BYTE **)(v5 - 8LL * *(unsigned int *)(v5 + 8)) - 4) < 0x1Fu;
  v9 = (unsigned int)v8 - 1 + 4;
  if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 8) )
    return a2;
  v10 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (v9 - v4)) + 136LL);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
  {
    v16 = *(_DWORD *)(v10 + 32);
    v17 = (unsigned __int8)(**(_BYTE **)(v5 - 8LL * *(unsigned int *)(v5 + 8)) - 4) < 0x1Fu;
    v11 = sub_16A57B0(v10 + 24);
    v8 = v17;
    if ( v16 != v11 )
      goto LABEL_6;
    return a2;
  }
  if ( !*(_QWORD *)(v10 + 24) )
    return a2;
LABEL_6:
  v12 = *(_QWORD *)(a2 - 8 * v4);
  if ( !v8 )
    return sub_161C5A0(a1, v12, v5, (__int64)v7, 0);
  v13 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (3 - v4)) + 136LL);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  return sub_161C680(a1, v12, v5, (__int64)v7, (__int64)v14, 0);
}
