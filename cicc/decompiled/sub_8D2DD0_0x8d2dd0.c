// Function: sub_8D2DD0
// Address: 0x8d2dd0
//
__int64 __fastcall sub_8D2DD0(__int64 a1)
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
  if ( (unsigned __int8)(v1 - 2) <= 3u )
  {
    if ( dword_4F077C4 == 2 && v1 == 2 )
      return ((*(_BYTE *)(a1 + 161) >> 3) ^ 1) & 1;
    else
      return 1;
  }
  return v2;
}
