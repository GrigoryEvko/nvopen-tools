// Function: sub_179D860
// Address: 0x179d860
//
__int64 __fastcall sub_179D860(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rdx
  __int64 v5; // rax
  int v6; // eax
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  int v13; // eax
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    if ( v2 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v11 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v12 = *(_QWORD *)(v11 + 8);
    if ( !v12 || *(_QWORD *)(v12 + 8) )
      return 0;
    v13 = *(unsigned __int8 *)(v11 + 16);
    if ( (unsigned __int8)v13 <= 0x17u )
    {
      if ( (_BYTE)v13 != 5 )
        return 0;
      if ( (unsigned int)*(unsigned __int16 *)(v11 + 18) - 24 > 1 )
        return 0;
      v18 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      if ( !v18 )
        return 0;
      **(_QWORD **)a1 = v18;
      if ( *(_QWORD *)(v11 + 24 * (1LL - (*(_DWORD *)(v11 + 20) & 0xFFFFFFF))) != *(_QWORD *)(a1 + 8) )
        return 0;
    }
    else
    {
      if ( (unsigned int)(v13 - 48) > 1 )
        return 0;
      v14 = (*(_BYTE *)(v11 + 23) & 0x40) != 0
          ? *(__int64 **)(v11 - 8)
          : (__int64 *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      v15 = *v14;
      if ( !v15 )
        return 0;
      **(_QWORD **)a1 = v15;
      v16 = (*(_BYTE *)(v11 + 23) & 0x40) != 0 ? *(_QWORD *)(v11 - 8) : v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
      if ( *(_QWORD *)(v16 + 24) != *(_QWORD *)(a1 + 8) )
        return 0;
    }
    v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v10 + 16) == 13 )
      goto LABEL_28;
    return 0;
  }
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_QWORD *)(v4 + 8);
  if ( !v5 || *(_QWORD *)(v5 + 8) )
    return 0;
  v6 = *(unsigned __int8 *)(v4 + 16);
  if ( (unsigned __int8)v6 <= 0x17u )
  {
    if ( (_BYTE)v6 != 5 )
      return 0;
    if ( (unsigned int)*(unsigned __int16 *)(v4 + 18) - 24 > 1 )
      return 0;
    v17 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    if ( !v17 )
      return 0;
    **(_QWORD **)a1 = v17;
    if ( *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF))) != *(_QWORD *)(a1 + 8) )
      return 0;
  }
  else
  {
    if ( (unsigned int)(v6 - 48) > 1 )
      return 0;
    v7 = (*(_BYTE *)(v4 + 23) & 0x40) != 0
       ? *(__int64 **)(v4 - 8)
       : (__int64 *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    v8 = *v7;
    if ( !v8 )
      return 0;
    **(_QWORD **)a1 = v8;
    v9 = (*(_BYTE *)(v4 + 23) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
    if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(a1 + 8) )
      return 0;
  }
  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) != 13 )
    return 0;
LABEL_28:
  **(_QWORD **)(a1 + 16) = v10;
  return 1;
}
