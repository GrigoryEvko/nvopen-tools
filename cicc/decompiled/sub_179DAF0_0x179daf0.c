// Function: sub_179DAF0
// Address: 0x179daf0
//
__int64 __fastcall sub_179DAF0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rdx
  __int64 v5; // rax
  int v6; // eax
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // eax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    if ( v2 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v13 = *(_QWORD *)(v12 + 8);
    if ( !v13 || *(_QWORD *)(v13 + 8) )
      return 0;
    v14 = *(unsigned __int8 *)(v12 + 16);
    if ( (unsigned __int8)v14 <= 0x17u )
    {
      if ( (_BYTE)v14 != 5 )
        return 0;
      if ( (unsigned int)*(unsigned __int16 *)(v12 + 18) - 24 > 1 )
        return 0;
      v20 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
      if ( !v20 )
        return 0;
      **a1 = v20;
      v18 = *(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
      if ( !v18 )
        return 0;
    }
    else
    {
      if ( (unsigned int)(v14 - 48) > 1 )
        return 0;
      v15 = (*(_BYTE *)(v12 + 23) & 0x40) != 0
          ? *(__int64 **)(v12 - 8)
          : (__int64 *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
      v16 = *v15;
      if ( !v16 )
        return 0;
      **a1 = v16;
      v17 = (*(_BYTE *)(v12 + 23) & 0x40) != 0 ? *(_QWORD *)(v12 - 8) : v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
      v18 = *(_QWORD *)(v17 + 24);
      if ( !v18 )
        return 0;
    }
    *a1[1] = v18;
    v11 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v11 + 16) == 13 )
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
    v19 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    if ( !v19 )
      return 0;
    **a1 = v19;
    v10 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( !v10 )
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
    **a1 = v8;
    v9 = (*(_BYTE *)(v4 + 23) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
    v10 = *(_QWORD *)(v9 + 24);
    if ( !v10 )
      return 0;
  }
  *a1[1] = v10;
  v11 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v11 + 16) != 13 )
    return 0;
LABEL_28:
  *a1[2] = v11;
  return 1;
}
