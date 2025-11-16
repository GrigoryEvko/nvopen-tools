// Function: sub_8E3180
// Address: 0x8e3180
//
_BOOL8 __fastcall sub_8E3180(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return !v1
      || (unsigned __int8)(v1 - 9) <= 2u
      || dword_4D047E4 && v1 == 2 && (*(_BYTE *)(a1 + 161) & 8) != 0
      || v1 == 14;
}
