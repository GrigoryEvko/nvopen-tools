// Function: sub_13D4800
// Address: 0x13d4800
//
__int64 __fastcall sub_13D4800(_QWORD *a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    v3 = 0;
    if ( v2 == 5 && *(_WORD *)(a2 + 18) == 26 )
    {
      v3 = 1;
      if ( *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) != *a1 )
        LOBYTE(v3) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == *a1;
    }
    return v3;
  }
  v3 = 1;
  if ( *(_QWORD *)(a2 - 48) == *a1 )
    return v3;
  LOBYTE(v3) = *(_QWORD *)(a2 - 24) == *a1;
  return v3;
}
