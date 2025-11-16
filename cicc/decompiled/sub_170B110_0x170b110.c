// Function: sub_170B110
// Address: 0x170b110
//
__int64 __fastcall sub_170B110(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // r13d
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 49 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    if ( !v4 )
      return 0;
    **(_QWORD **)a1 = v4;
    v5 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v5 + 16) != 13 )
      return 0;
  }
  else
  {
    if ( v2 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 25 )
      return 0;
    v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v9 )
      return 0;
    **(_QWORD **)a1 = v9;
    v5 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v5 + 16) != 13 )
      return 0;
  }
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
  {
    v7 = *(_QWORD **)(a1 + 8);
    v8 = *(_QWORD *)(v5 + 24);
  }
  else
  {
    if ( v6 - (unsigned int)sub_16A57B0(v5 + 24) > 0x40 )
      return 0;
    v7 = *(_QWORD **)(a1 + 8);
    v8 = **(_QWORD **)(v5 + 24);
  }
  *v7 = v8;
  return 1;
}
