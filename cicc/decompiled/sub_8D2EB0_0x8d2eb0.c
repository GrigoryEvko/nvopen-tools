// Function: sub_8D2EB0
// Address: 0x8d2eb0
//
_BOOL8 __fastcall sub_8D2EB0(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 6 && (*(_BYTE *)(a1 + 168) & 1) == 0 && sub_8D2530(*(_QWORD *)(a1 + 160));
}
