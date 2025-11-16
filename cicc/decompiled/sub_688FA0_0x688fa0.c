// Function: sub_688FA0
// Address: 0x688fa0
//
__int64 __fastcall sub_688FA0(_QWORD *a1)
{
  _DWORD v2[3]; // [rsp+Ch] [rbp-14h] BYREF

  v2[0] = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8D3A70(*a1) && (*(_BYTE *)(unk_4D03C50 + 16LL) > 3u || word_4D04898) )
      sub_845C60(a1, 0, unk_4D0439C == 0 ? 231 : 64, 2048, v2);
    if ( v2[0] )
      return sub_6FC8A0(a1);
  }
  sub_6F69D0(a1, 0);
  return sub_6FC8A0(a1);
}
