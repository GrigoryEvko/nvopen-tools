// Function: sub_8D9760
// Address: 0x8d9760
//
__int64 __fastcall sub_8D9760(__int64 a1)
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
  if ( v1 != 1 )
  {
    v2 = 0;
    if ( (unsigned __int8)(v1 - 9) <= 2u && (*(_BYTE *)(a1 + 176) & 0x20) != 0 )
      return dword_4F077BC == 0;
  }
  return v2;
}
