// Function: sub_760730
// Address: 0x760730
//
void __fastcall sub_760730(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 203);
  if ( (v1 & 4) != 0 )
  {
    *(_BYTE *)(a1 + 203) = v1 & 0xF3;
    sub_7605A0(a1);
  }
  else if ( (v1 & 8) != 0 )
  {
    *(_BYTE *)(a1 + 203) = v1 & 0xF7;
    sub_75BCD0(a1);
  }
}
