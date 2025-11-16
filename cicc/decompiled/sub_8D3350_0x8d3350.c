// Function: sub_8D3350
// Address: 0x8d3350
//
__int64 __fastcall sub_8D3350(__int64 a1)
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
  if ( (unsigned __int8)(v1 - 2) > 3u && (v1 != 6 || (*(_BYTE *)(a1 + 168) & 1) != 0) )
    return (v1 == 13) | (unsigned __int8)((unsigned __int8)(v1 - 19) <= 1u);
  return v2;
}
