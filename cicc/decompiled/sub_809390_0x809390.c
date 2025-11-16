// Function: sub_809390
// Address: 0x809390
//
__int64 __fastcall sub_809390(__int64 a1)
{
  char v2; // al

  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 != 9 )
      break;
LABEL_8:
    if ( (*(_BYTE *)(a1 + 178) & 0x20) == 0 || !*(_QWORD *)(*(_QWORD *)(a1 + 168) + 256LL) )
      return a1;
    a1 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 256LL);
  }
  while ( v2 == 12 )
  {
    if ( !dword_4D0425C || *(_BYTE *)(a1 + 184) != 1 || (*(_BYTE *)(a1 + 186) & 8) == 0 || (unsigned int)sub_809230(a1) )
      return sub_8D2250(a1);
    a1 = *(_QWORD *)(a1 + 160);
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 == 9 )
      goto LABEL_8;
  }
  return a1;
}
