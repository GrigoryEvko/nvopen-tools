// Function: sub_8D3D70
// Address: 0x8d3d70
//
_BOOL8 __fastcall sub_8D3D70(__int64 a1)
{
  char v1; // al
  char v2; // al
  __int64 v4; // rdx
  char v5; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( v1 != 6 || (*(_BYTE *)(a1 + 168) & 1) == 0 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    return v2 == 14;
  }
  v4 = *(_QWORD *)(a1 + 160);
  v2 = *(_BYTE *)(v4 + 140);
  if ( v2 != 12 )
    return v2 == 14;
  do
  {
    v4 = *(_QWORD *)(v4 + 160);
    v5 = *(_BYTE *)(v4 + 140);
  }
  while ( v5 == 12 );
  return v5 == 14;
}
