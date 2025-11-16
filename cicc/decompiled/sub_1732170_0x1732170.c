// Function: sub_1732170
// Address: 0x1732170
//
__int64 __fastcall sub_1732170(_QWORD *a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d

  v2 = *(_BYTE *)(a2 + 16);
  v3 = 0;
  if ( v2 != 50 )
  {
    if ( v2 == 5 && *(_WORD *)(a2 + 18) == 26 && *a1 == *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
      LOBYTE(v3) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == a1[1];
    return v3;
  }
  if ( *a1 != *(_QWORD *)(a2 - 48) )
    return v3;
  LOBYTE(v3) = *(_QWORD *)(a2 - 24) == a1[1];
  return v3;
}
