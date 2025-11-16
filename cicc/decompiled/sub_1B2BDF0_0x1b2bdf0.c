// Function: sub_1B2BDF0
// Address: 0x1b2bdf0
//
__int64 __fastcall sub_1B2BDF0(_DWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rdx
  unsigned __int8 v7; // al
  int v8; // eax
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  int v11; // eax
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  int v16; // eax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( (unsigned __int8)(*(_BYTE *)(v12 + 16) - 75) > 1u )
    {
      v14 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      v15 = *(_BYTE *)(v14 + 16);
      if ( v15 <= 0x17u || (unsigned __int8)(v15 - 75) > 1u )
        return 0;
      v16 = *(unsigned __int16 *)(v14 + 18);
      BYTE1(v16) &= ~0x80u;
      **a1 = v16;
      v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v7 = *(_BYTE *)(v6 + 16);
    }
    else
    {
      v13 = *(unsigned __int16 *)(v12 + 18);
      BYTE1(v13) &= ~0x80u;
      **a1 = v13;
      v6 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      v7 = *(_BYTE *)(v6 + 16);
      if ( v7 <= 0x17u )
        return 0;
    }
LABEL_7:
    if ( (unsigned __int8)(v7 - 75) <= 1u )
      goto LABEL_8;
    return 0;
  }
  v4 = *(_QWORD *)(a2 - 48);
  if ( (unsigned __int8)(*(_BYTE *)(v4 + 16) - 75) <= 1u )
  {
    v5 = *(unsigned __int16 *)(v4 + 18);
    BYTE1(v5) &= ~0x80u;
    **a1 = v5;
    v6 = *(_QWORD *)(a2 - 24);
    v7 = *(_BYTE *)(v6 + 16);
    if ( v7 <= 0x17u )
      return 0;
    goto LABEL_7;
  }
  v9 = *(_QWORD *)(a2 - 24);
  v10 = *(_BYTE *)(v9 + 16);
  if ( v10 <= 0x17u )
    return 0;
  if ( (unsigned __int8)(v10 - 75) > 1u )
    return 0;
  v11 = *(unsigned __int16 *)(v9 + 18);
  BYTE1(v11) &= ~0x80u;
  **a1 = v11;
  v6 = *(_QWORD *)(a2 - 48);
  if ( (unsigned __int8)(*(_BYTE *)(v6 + 16) - 75) > 1u )
    return 0;
LABEL_8:
  v8 = *(unsigned __int16 *)(v6 + 18);
  BYTE1(v8) &= ~0x80u;
  *a1[2] = v8;
  return 1;
}
