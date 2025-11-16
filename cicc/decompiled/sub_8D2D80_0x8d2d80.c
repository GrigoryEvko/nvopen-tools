// Function: sub_8D2D80
// Address: 0x8d2d80
//
_BOOL8 __fastcall sub_8D2D80(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 2 && (*(_BYTE *)(a1 + 161) & 0x10) == 0 || (unsigned __int8)(v1 - 3) <= 2u;
}
