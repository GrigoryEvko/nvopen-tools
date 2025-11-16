// Function: sub_1734730
// Address: 0x1734730
//
__int64 __fastcall sub_1734730(_QWORD *a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 v8; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    v5 = *(_QWORD *)(a2 - 48);
    v6 = *(_QWORD *)(a2 - 24);
    if ( v5 != *a1 || (v3 = 1, a1[1] != v6) )
    {
      v3 = 0;
      if ( *a1 == v6 )
        LOBYTE(v3) = a1[1] == v5;
    }
  }
  else
  {
    v3 = 0;
    if ( v2 == 5 && *(_WORD *)(a2 + 18) == 28 )
    {
      v7 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v8 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *a1 != v7 || (v3 = 1, a1[1] != v8) )
      {
        v3 = 0;
        if ( *a1 == v8 )
          LOBYTE(v3) = a1[1] == v7;
      }
    }
  }
  return v3;
}
