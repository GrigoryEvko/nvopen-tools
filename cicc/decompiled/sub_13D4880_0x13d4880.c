// Function: sub_13D4880
// Address: 0x13d4880
//
__int64 __fastcall sub_13D4880(_QWORD *a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rdx
  int v6; // eax
  __int64 v7; // rdx
  int v8; // eax
  int v9; // eax
  _QWORD *v10; // rdx
  int v11; // eax
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  int v14; // edx
  int v15; // edx
  _QWORD *v16; // rcx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    v3 = 0;
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return v3;
    v13 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v14 = *(unsigned __int8 *)(v13 + 16);
    if ( (unsigned __int8)v14 > 0x17u )
    {
      v15 = v14 - 24;
    }
    else
    {
      if ( (_BYTE)v14 != 5 )
        goto LABEL_21;
      v15 = *(unsigned __int16 *)(v13 + 18);
    }
    if ( v15 == 45 )
    {
      v16 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
          ? *(_QWORD **)(v13 - 8)
          : (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v3 = 1;
      if ( *v16 == *a1 )
        return v3;
    }
LABEL_21:
    v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v8 = *(unsigned __int8 *)(v7 + 16);
    if ( (unsigned __int8)v8 <= 0x17u )
      goto LABEL_22;
    goto LABEL_8;
  }
  v5 = *(_QWORD *)(a2 - 48);
  v6 = *(unsigned __int8 *)(v5 + 16);
  if ( (unsigned __int8)v6 > 0x17u )
  {
    v11 = v6 - 24;
  }
  else
  {
    if ( (_BYTE)v6 != 5 )
      goto LABEL_7;
    v11 = *(unsigned __int16 *)(v5 + 18);
  }
  if ( v11 == 45 )
  {
    v12 = (*(_BYTE *)(v5 + 23) & 0x40) != 0
        ? *(_QWORD **)(v5 - 8)
        : (_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
    v3 = 1;
    if ( *v12 == *a1 )
      return v3;
  }
LABEL_7:
  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(unsigned __int8 *)(v7 + 16);
  if ( (unsigned __int8)v8 <= 0x17u )
  {
LABEL_22:
    v3 = 0;
    if ( (_BYTE)v8 != 5 )
      return v3;
    v9 = *(unsigned __int16 *)(v7 + 18);
    goto LABEL_9;
  }
LABEL_8:
  v9 = v8 - 24;
LABEL_9:
  v3 = 0;
  if ( v9 == 45 )
  {
    if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
      v10 = *(_QWORD **)(v7 - 8);
    else
      v10 = (_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    LOBYTE(v3) = *v10 == *a1;
  }
  return v3;
}
