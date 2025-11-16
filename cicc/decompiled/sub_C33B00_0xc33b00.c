// Function: sub_C33B00
// Address: 0xc33b00
//
__int64 __fastcall sub_C33B00(__int64 a1)
{
  unsigned int v1; // r12d
  _DWORD *v3; // rbx

  LOBYTE(v1) = (*(_BYTE *)(a1 + 20) & 7) != 3 && (*(_BYTE *)(a1 + 20) & 6) != 0;
  if ( !(_BYTE)v1 )
    return v1;
  v3 = *(_DWORD **)a1;
  if ( *(_DWORD *)(a1 + 16) != **(_DWORD **)a1 )
    return 0;
  if ( v3[4] != 1 || v3[5] != 1 )
    return sub_C339A0(a1);
  if ( v3 == sub_C333E0() )
    return v1;
  return sub_C33A50(a1);
}
