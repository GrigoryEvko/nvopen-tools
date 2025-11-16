// Function: sub_1731FF0
// Address: 0x1731ff0
//
__int64 __fastcall sub_1731FF0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v8 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v9 = *(_BYTE *)(v8 + 16);
    if ( v9 == 51 )
    {
      if ( *(_QWORD *)(v8 - 48) != *(_QWORD *)a1 )
        return 0;
      v10 = *(_QWORD *)(v8 - 24);
      if ( !v10 )
        return 0;
    }
    else
    {
      if ( v9 != 5 )
        return 0;
      if ( *(_WORD *)(v8 + 18) != 27 )
        return 0;
      if ( *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)) != *(_QWORD *)a1 )
        return 0;
      v10 = *(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
      if ( !v10 )
        return 0;
    }
    **(_QWORD **)(a1 + 8) = v10;
    v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( v7 )
      goto LABEL_18;
    return 0;
  }
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 == 51 )
  {
    if ( *(_QWORD *)(v4 - 48) != *(_QWORD *)a1 )
      return 0;
    v6 = *(_QWORD *)(v4 - 24);
    if ( !v6 )
      return 0;
  }
  else
  {
    if ( v5 != 5 )
      return 0;
    if ( *(_WORD *)(v4 + 18) != 27 )
      return 0;
    if ( *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)) != *(_QWORD *)a1 )
      return 0;
    v6 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( !v6 )
      return 0;
  }
  **(_QWORD **)(a1 + 8) = v6;
  v7 = *(_QWORD *)(a2 - 24);
  if ( !v7 )
    return 0;
LABEL_18:
  **(_QWORD **)(a1 + 16) = v7;
  return 1;
}
