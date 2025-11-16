// Function: sub_171EF90
// Address: 0x171ef90
//
__int64 __fastcall sub_171EF90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v5; // al
  unsigned int v6; // r8d
  __int64 v8; // rax
  __int64 v9; // rax

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 37 )
  {
    v8 = *(_QWORD *)(a2 - 48);
    if ( v8 )
    {
      **(_QWORD **)a1 = v8;
      LOBYTE(a5) = *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 - 24);
      return a5;
    }
    return 0;
  }
  v6 = 0;
  if ( v5 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 13 )
    return 0;
  v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v9 )
    return 0;
  **(_QWORD **)a1 = v9;
  LOBYTE(v6) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == *(_QWORD *)(a1 + 8);
  return v6;
}
