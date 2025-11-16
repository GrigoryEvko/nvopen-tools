// Function: sub_104A900
// Address: 0x104a900
//
char __fastcall sub_104A900(__int64 *a1)
{
  int v1; // eax
  __int64 v2; // rdi
  __int64 v3; // rdi

  v2 = *a1;
  LOBYTE(v1) = *(_BYTE *)v2 <= 0x1Cu;
  if ( (unsigned __int8)(*(_BYTE *)v2 - 63) <= 0x15u )
    v1 |= (0x21FFF1uLL >> (*(_BYTE *)v2 - 63)) & 1;
  if ( !(_BYTE)v1 && *(_BYTE *)v2 == 42 )
  {
    if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
      v3 = *(_QWORD *)(v2 - 8);
    else
      v3 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    LOBYTE(v1) = **(_BYTE **)(v3 + 32) == 17;
  }
  return v1;
}
