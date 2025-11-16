// Function: sub_8D2B50
// Address: 0x8d2b50
//
_BOOL8 __fastcall sub_8D2B50(__int64 a1)
{
  char v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return v1 == 5;
}
