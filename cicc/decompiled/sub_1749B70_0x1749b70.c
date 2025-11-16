// Function: sub_1749B70
// Address: 0x1749b70
//
_BOOL8 __fastcall sub_1749B70(__int64 a1, __int64 a2)
{
  int v2; // eax
  _QWORD **v4; // rdx
  _QWORD *v5; // rdx
  int v6; // edx
  _QWORD **v7; // rdi
  _QWORD **v8; // rdx

  v2 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    if ( (_BYTE)v2 != 61 )
      goto LABEL_9;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    if ( *(_WORD *)(a1 + 18) != 37 )
      goto LABEL_16;
  }
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v4 = *(_QWORD ***)(a1 - 8);
  else
    v4 = (_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v5 = *v4;
  if ( v5 )
    return *v5 == a2;
  if ( (unsigned __int8)v2 <= 0x17u )
  {
LABEL_16:
    v6 = *(unsigned __int16 *)(a1 + 18);
    if ( (_WORD)v6 == 38 )
      goto LABEL_17;
    goto LABEL_10;
  }
LABEL_9:
  v6 = (unsigned __int8)v2 - 24;
  if ( (_BYTE)v2 == 62 )
  {
LABEL_17:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v8 = *(_QWORD ***)(a1 - 8);
    else
      v8 = (_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v5 = *v8;
    if ( v5 )
      return *v5 == a2;
    if ( (unsigned __int8)v2 <= 0x17u )
      v6 = *(unsigned __int16 *)(a1 + 18);
    else
      v6 = v2 - 24;
  }
LABEL_10:
  if ( v6 != 36 )
    return 0;
  v7 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
     ? *(_QWORD ***)(a1 - 8)
     : (_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v5 = *v7;
  if ( !*v7 )
    return 0;
  return *v5 == a2;
}
