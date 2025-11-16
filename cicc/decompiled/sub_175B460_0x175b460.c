// Function: sub_175B460
// Address: 0x175b460
//
__int64 __fastcall sub_175B460(__int64 a1, __int64 a2)
{
  int v2; // eax
  int v4; // eax
  __int64 *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // r12
  unsigned int v11; // r13d
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    v4 = v2 - 24;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *(unsigned __int16 *)(a2 + 18);
  }
  if ( v4 != 36 )
    return 0;
  v5 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(__int64 **)(a2 - 8)
     : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v6 = *v5;
  v7 = *(_QWORD *)(*v5 + 8);
  if ( !v7 || *(_QWORD *)(v7 + 8) )
    return 0;
  v8 = *(_BYTE *)(v6 + 16);
  if ( v8 == 48 )
  {
    v14 = *(_QWORD *)(v6 - 48);
    if ( !v14 )
      return 0;
    **(_QWORD **)a1 = v14;
    v10 = *(_QWORD *)(v6 - 24);
    if ( *(_BYTE *)(v10 + 16) != 13 )
      return 0;
  }
  else
  {
    if ( v8 != 5 )
      return 0;
    if ( *(_WORD *)(v6 + 18) != 24 )
      return 0;
    v9 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    if ( !v9 )
      return 0;
    **(_QWORD **)a1 = v9;
    v10 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v10 + 16) != 13 )
      return 0;
  }
  v11 = *(_DWORD *)(v10 + 32);
  if ( v11 > 0x40 )
  {
    if ( v11 - (unsigned int)sub_16A57B0(v10 + 24) > 0x40 )
      return 0;
    v12 = *(_QWORD **)(a1 + 8);
    v13 = **(_QWORD **)(v10 + 24);
  }
  else
  {
    v12 = *(_QWORD **)(a1 + 8);
    v13 = *(_QWORD *)(v10 + 24);
  }
  *v12 = v13;
  return 1;
}
