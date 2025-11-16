// Function: sub_808EC0
// Address: 0x808ec0
//
__int64 __fastcall sub_808EC0(__int64 a1, _DWORD *a2)
{
  char v2; // al

  *a2 = 0;
  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 != 12 )
      break;
LABEL_5:
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 184) - 6) <= 1u && (*(_BYTE *)(a1 + 186) & 8) != 0 )
      return a1;
    *a2 |= *(_BYTE *)(a1 + 185) & 0x7F;
    a1 = *(_QWORD *)(a1 + 160);
  }
  while ( v2 == 8 )
  {
    a1 = *(_QWORD *)(a1 + 160);
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 == 12 )
      goto LABEL_5;
  }
  return 0;
}
