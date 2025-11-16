// Function: sub_17319E0
// Address: 0x17319e0
//
__int64 __fastcall sub_17319E0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    v5 = *(_BYTE *)(v4 + 16);
    if ( v5 == 52 )
    {
      v13 = *(_QWORD *)(v4 - 48);
      if ( !v13 )
        return 0;
      **a1 = v13;
      v7 = *(_QWORD *)(v4 - 24);
      if ( !v7 )
        return 0;
    }
    else
    {
      if ( v5 != 5 )
        return 0;
      if ( *(_WORD *)(v4 + 18) != 28 )
        return 0;
      v6 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
      if ( !v6 )
        return 0;
      **a1 = v6;
      v7 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
      if ( !v7 )
        return 0;
    }
    *a1[1] = v7;
    v12 = *(_QWORD *)(a2 - 24);
    if ( !v12 )
      return 0;
  }
  else
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v8 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v9 = *(_BYTE *)(v8 + 16);
    if ( v9 == 52 )
    {
      v14 = *(_QWORD *)(v8 - 48);
      if ( !v14 )
        return 0;
      **a1 = v14;
      v11 = *(_QWORD *)(v8 - 24);
      if ( !v11 )
        return 0;
    }
    else
    {
      if ( v9 != 5 )
        return 0;
      if ( *(_WORD *)(v8 + 18) != 28 )
        return 0;
      v10 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
      if ( !v10 )
        return 0;
      **a1 = v10;
      v11 = *(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
      if ( !v11 )
        return 0;
    }
    *a1[1] = v11;
    v12 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v12 )
      return 0;
  }
  *a1[2] = v12;
  return 1;
}
