// Function: sub_624600
// Address: 0x624600
//
_BOOL8 __fastcall sub_624600(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // [rsp+Ch] [rbp-24h]

  v5 = 0;
  if ( (unsigned int)sub_8D2310(a1) )
  {
    sub_6851C0(90, a3);
    v5 = 1;
  }
  else if ( (unsigned int)sub_8D3410(a1) )
  {
    sub_6851C0(91, a3);
    v5 = 1;
  }
  else if ( dword_4F077C4 == 2 && unk_4D047EC && (unsigned int)sub_8DD010(a1) )
  {
    sub_6851C0(1403, a3);
    v5 = 1;
  }
  if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 && (unsigned int)sub_8D4C10(a1, dword_4F077C4 != 2) )
    sub_6243B0(a1, a2, a3);
  return v5 == 0;
}
