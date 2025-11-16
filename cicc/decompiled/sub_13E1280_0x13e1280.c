// Function: sub_13E1280
// Address: 0x13e1280
//
bool __fastcall sub_13E1280(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // eax
  int v7; // eax
  _QWORD *v8; // rdx
  __int64 v9; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 37 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    if ( !v4 )
      return 0;
    **(_QWORD **)a1 = v4;
    v5 = *(_QWORD *)(a2 - 24);
    v6 = *(unsigned __int8 *)(v5 + 16);
    if ( (unsigned __int8)v6 <= 0x17u )
      goto LABEL_14;
LABEL_7:
    v7 = v6 - 24;
    goto LABEL_8;
  }
  if ( v2 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 13 )
    return 0;
  v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v9 )
    return 0;
  **(_QWORD **)a1 = v9;
  v5 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v6 = *(unsigned __int8 *)(v5 + 16);
  if ( (unsigned __int8)v6 > 0x17u )
    goto LABEL_7;
LABEL_14:
  if ( (_BYTE)v6 != 5 )
    return 0;
  v7 = *(unsigned __int16 *)(v5 + 18);
LABEL_8:
  if ( v7 != 45 )
    return 0;
  if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
    v8 = *(_QWORD **)(v5 - 8);
  else
    v8 = (_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
  return *v8 == *(_QWORD *)(a1 + 8);
}
