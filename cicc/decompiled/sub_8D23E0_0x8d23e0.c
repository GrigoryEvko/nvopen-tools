// Function: sub_8D23E0
// Address: 0x8d23e0
//
__int64 __fastcall sub_8D23E0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v2 = 0;
  if ( v1 == 8 && (*(_BYTE *)(a1 + 169) & 0x20) == 0 && (*(_WORD *)(a1 + 168) & 0x180) == 0 )
    return *(_QWORD *)(a1 + 176) == 0;
  return v2;
}
