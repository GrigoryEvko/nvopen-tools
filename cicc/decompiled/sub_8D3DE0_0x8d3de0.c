// Function: sub_8D3DE0
// Address: 0x8d3de0
//
__int64 __fastcall sub_8D3DE0(__int64 a1)
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
  if ( v1 == 14 )
    return *(_BYTE *)(a1 + 160) == 2;
  return v2;
}
