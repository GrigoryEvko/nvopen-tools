// Function: sub_8D2780
// Address: 0x8d2780
//
__int64 __fastcall sub_8D2780(__int64 a1)
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
  if ( v1 != 2 )
    return v2;
  v2 = 1;
  if ( unk_4D04000 )
    return v2;
  else
    return ((*(_BYTE *)(a1 + 161) >> 3) ^ 1) & 1;
}
