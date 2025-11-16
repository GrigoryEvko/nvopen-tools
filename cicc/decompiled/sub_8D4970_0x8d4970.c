// Function: sub_8D4970
// Address: 0x8d4970
//
__int64 __fastcall sub_8D4970(__int64 a1)
{
  while ( 1 )
  {
    if ( *(_BYTE *)(a1 + 140) != 12 )
      return 0;
    if ( *(_QWORD *)(a1 + 8) )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return 1;
}
