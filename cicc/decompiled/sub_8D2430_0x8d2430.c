// Function: sub_8D2430
// Address: 0x8d2430
//
_BOOL8 __fastcall sub_8D2430(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 8
      && !*(_QWORD *)(a1 + 176)
      && (dword_4F077BC || (*(_BYTE *)(a1 + 169) & 0x20) == 0)
      && (*(_WORD *)(a1 + 168) & 0x180) == 0;
}
