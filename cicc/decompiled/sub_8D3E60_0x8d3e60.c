// Function: sub_8D3E60
// Address: 0x8d3e60
//
__int64 __fastcall sub_8D3E60(__int64 a1)
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
  if ( (unsigned __int8)(v1 - 9) <= 2u )
    return *(_BYTE *)(a1 + 177) & 1;
  return v2;
}
