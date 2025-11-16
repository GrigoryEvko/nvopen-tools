// Function: sub_B8D060
// Address: 0xb8d060
//
__int64 __fastcall sub_B8D060(_QWORD *a1, __int64 a2)
{
  bool v4; // cl
  __int64 *v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  _QWORD *v9; // rbx
  unsigned __int8 v10; // al
  _BYTE **v11; // rax
  bool v12; // si
  __int64 v13; // rax
  __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // rax
  _QWORD *v18; // r8
  __int64 *v19; // [rsp+0h] [rbp-40h]
  int v20; // [rsp+8h] [rbp-38h]

  v4 = (*(_BYTE *)(a2 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v5 = *(__int64 **)(a2 - 32);
  else
    v5 = (__int64 *)(a2 - 8LL * ((*(_BYTE *)(a2 - 16) >> 2) & 0xF) - 16);
  v6 = *v5;
  v7 = v5[1];
  v8 = *(_QWORD *)(v5[2] + 136);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_BYTE *)(v7 - 16);
  if ( (v10 & 2) != 0 )
    v11 = *(_BYTE ***)(v7 - 32);
  else
    v11 = (_BYTE **)(v7 - 8LL * ((v10 >> 2) & 0xF) - 16);
  v12 = (unsigned __int8)(**v11 - 5) < 0x20u;
  v13 = (unsigned int)v12 - 1 + 4;
  if ( v4 )
  {
    if ( (unsigned int)v13 >= *(_DWORD *)(a2 - 24) )
      return a2;
  }
  else if ( (unsigned int)v13 >= ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) )
  {
    return a2;
  }
  v15 = *(_QWORD *)(v5[v13] + 136);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
  {
    v20 = *(_DWORD *)(v15 + 32);
    v19 = v5;
    v16 = sub_C444A0(v15 + 24);
    v5 = v19;
    if ( v20 == v16 )
      return a2;
    if ( v12 )
      goto LABEL_14;
    return sub_B8CE70(a1, v6, v7, (__int64)v9, 0);
  }
  if ( !*(_QWORD *)(v15 + 24) )
    return a2;
  if ( !v12 )
    return sub_B8CE70(a1, v6, v7, (__int64)v9, 0);
LABEL_14:
  v17 = *(_QWORD *)(v5[3] + 136);
  v18 = *(_QWORD **)(v17 + 24);
  if ( *(_DWORD *)(v17 + 32) > 0x40u )
    v18 = (_QWORD *)*v18;
  return sub_B8CF50(a1, v6, v7, (__int64)v9, (__int64)v18, 0);
}
