// Function: sub_712570
// Address: 0x712570
//
_BOOL8 __fastcall sub_712570(__int64 a1)
{
  __int64 v2; // r12
  char i; // al

  if ( *(_BYTE *)(a1 + 173) != 1 )
    return 0;
  v2 = *(_QWORD *)(a1 + 128);
  for ( i = *(_BYTE *)(v2 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  if ( i != 19 )
  {
    if ( ((*(_BYTE *)(a1 + 169) & 4) != 0 || i == 6 && dword_4F077C0 && !(unsigned int)sub_8D4C80(v2))
      && (!HIDWORD(qword_4F077B4) || qword_4F077A8 > 0x9E33u || !(unsigned int)sub_8D2780(v2))
      || (unsigned int)sub_6210B0(a1, 0)
      || !unk_4D04000 && (unsigned int)sub_8D2870(v2) )
    {
      return 0;
    }
    if ( dword_4D04394 )
      return (unsigned int)sub_8D29A0(v2) == 0;
  }
  return 1;
}
