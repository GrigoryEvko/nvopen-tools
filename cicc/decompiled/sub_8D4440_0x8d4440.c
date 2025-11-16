// Function: sub_8D4440
// Address: 0x8d4440
//
__int64 __fastcall sub_8D4440(__int64 a1)
{
  char v1; // al

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  while ( (*(_BYTE *)(a1 + 169) & 0x20) == 0 )
  {
    do
    {
      a1 = *(_QWORD *)(a1 + 160);
      v1 = *(_BYTE *)(a1 + 140);
    }
    while ( v1 == 12 );
    if ( v1 != 8 )
      return 0;
  }
  return 1;
}
