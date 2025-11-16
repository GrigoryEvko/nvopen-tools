// Function: sub_8D3BB0
// Address: 0x8d3bb0
//
__int64 __fastcall sub_8D3BB0(__int64 a1)
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
  v2 = 1;
  if ( v1 != 8 )
  {
    v2 = 0;
    if ( (unsigned __int8)(v1 - 9) <= 2u )
      return (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 178LL) & 0x40) != 0;
  }
  return v2;
}
