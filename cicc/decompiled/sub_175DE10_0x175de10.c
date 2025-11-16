// Function: sub_175DE10
// Address: 0x175de10
//
__int64 __fastcall sub_175DE10(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  int v4; // eax
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rsi
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rsi

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 48);
  if ( !v2 )
    return 0;
  v3 = *(_QWORD *)(a2 - 24);
  if ( !v3 )
    return 0;
  v4 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v4) &= ~0x80u;
  if ( v4 == 36 )
  {
    v9 = *(_BYTE *)(v2 + 16);
    if ( v9 == 35 )
    {
      v10 = *(_QWORD *)(v2 - 48);
      if ( !v10 )
        return 0;
      v11 = *(_QWORD *)(v2 - 24);
      if ( !v11 )
        return 0;
    }
    else
    {
      if ( v9 != 5 )
        return 0;
      if ( *(_WORD *)(v2 + 18) != 11 )
        return 0;
      v10 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
      if ( !v10 )
        return 0;
      v11 = *(_QWORD *)(v2 + 24 * (1LL - (*(_DWORD *)(v2 + 20) & 0xFFFFFFF)));
      if ( !v11 )
        return 0;
    }
    if ( v3 == v10 )
    {
      **a1 = v3;
    }
    else
    {
      if ( v11 != v3 )
        return 0;
      **a1 = v10;
    }
    *a1[1] = v11;
    if ( *(_BYTE *)(v2 + 16) <= 0x17u )
      return 0;
    *a1[2] = v2;
    return 1;
  }
  else
  {
    if ( v4 != 34 )
      return 0;
    v6 = *(_BYTE *)(v3 + 16);
    if ( v6 == 35 )
    {
      v7 = *(_QWORD *)(v3 - 48);
      if ( !v7 )
        return 0;
      v8 = *(_QWORD *)(v3 - 24);
      if ( !v8 )
        return 0;
    }
    else
    {
      if ( v6 != 5 )
        return 0;
      if ( *(_WORD *)(v3 + 18) != 11 )
        return 0;
      v7 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
      if ( !v7 )
        return 0;
      v8 = *(_QWORD *)(v3 + 24 * (1LL - (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)));
      if ( !v8 )
        return 0;
    }
    if ( v7 != v2 && v8 != v2 )
      return 0;
    **a1 = v7;
    *a1[1] = v8;
    if ( *(_BYTE *)(v3 + 16) <= 0x17u )
      return 0;
    *a1[2] = v3;
    return 1;
  }
}
