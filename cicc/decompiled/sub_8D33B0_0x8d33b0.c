// Function: sub_8D33B0
// Address: 0x8d33b0
//
__int64 __fastcall sub_8D33B0(__int64 a1)
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
  if ( (unsigned __int8)(v1 - 2) <= 3u )
    return v2;
  if ( v1 != 6 )
    return ((unsigned __int8)(v1 - 19) <= 1u || v1 == 13) && v1 != 13;
  return (*(_BYTE *)(a1 + 168) & 1) == 0;
}
