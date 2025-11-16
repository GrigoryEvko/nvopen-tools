// Function: sub_8D43F0
// Address: 0x8d43f0
//
__int64 __fastcall sub_8D43F0(__int64 a1)
{
  char v1; // al

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  while ( (*(_WORD *)(a1 + 168) & 0x180) == 0 )
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
