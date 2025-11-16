// Function: sub_17348B0
// Address: 0x17348b0
//
__int64 __fastcall sub_17348B0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  char v4; // al
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx

  v2 = *(_QWORD *)(a2 + 8);
  v3 = 0;
  if ( v2 && !*(_QWORD *)(v2 + 8) )
  {
    v4 = *(_BYTE *)(a2 + 16);
    if ( v4 == 35 )
    {
      v8 = *(_QWORD *)(a2 - 48);
      v9 = *(_QWORD *)(a2 - 24);
      if ( v8 != *a1 || (v3 = 1, a1[1] != v9) )
      {
        v3 = 0;
        if ( *a1 == v9 )
          LOBYTE(v3) = a1[1] == v8;
      }
    }
    else if ( v4 == 5 && *(_WORD *)(a2 + 18) == 11 )
    {
      v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v6 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( v5 != *a1 || (v3 = 1, a1[1] != v6) )
      {
        v3 = 0;
        if ( *a1 == v6 )
          LOBYTE(v3) = a1[1] == v5;
      }
    }
  }
  return v3;
}
