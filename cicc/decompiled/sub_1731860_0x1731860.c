// Function: sub_1731860
// Address: 0x1731860
//
__int64 __fastcall sub_1731860(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    v5 = *(_QWORD *)(a2 - 48);
    v3 = 0;
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 == 52 )
    {
      if ( *(_QWORD *)(v5 - 48) != *(_QWORD *)a1 )
        return v3;
      v7 = *(_QWORD *)(v5 - 24);
      if ( !v7 )
        return v3;
    }
    else
    {
      if ( v6 != 5 )
        return v3;
      if ( *(_WORD *)(v5 + 18) != 28 )
        return v3;
      if ( *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)) != *(_QWORD *)a1 )
        return v3;
      v7 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
      if ( !v7 )
        return v3;
    }
    **(_QWORD **)(a1 + 8) = v7;
    LOBYTE(v3) = *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 - 24);
    return v3;
  }
  v3 = 0;
  if ( v2 == 5 && *(_WORD *)(a2 + 18) == 28 )
  {
    v8 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v9 = *(_BYTE *)(v8 + 16);
    if ( v9 == 52 )
    {
      if ( *(_QWORD *)(v8 - 48) != *(_QWORD *)a1 )
        return v3;
      v10 = *(_QWORD *)(v8 - 24);
      if ( !v10 )
        return v3;
    }
    else
    {
      if ( v9 != 5 )
        return v3;
      if ( *(_WORD *)(v8 + 18) != 28 )
        return v3;
      if ( *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)) != *(_QWORD *)a1 )
        return v3;
      v10 = *(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
      if ( !v10 )
        return v3;
    }
    **(_QWORD **)(a1 + 8) = v10;
    LOBYTE(v3) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == *(_QWORD *)(a1 + 16);
  }
  return v3;
}
