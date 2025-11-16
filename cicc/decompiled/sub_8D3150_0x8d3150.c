// Function: sub_8D3150
// Address: 0x8d3150
//
__int64 __fastcall sub_8D3150(__int64 a1)
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
  if ( v1 == 7 )
    return *(_WORD *)(*(_QWORD *)(a1 + 168) + 18LL) != 0;
  return v2;
}
