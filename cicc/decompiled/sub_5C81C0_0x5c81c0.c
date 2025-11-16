// Function: sub_5C81C0
// Address: 0x5c81c0
//
char *__fastcall sub_5C81C0(__int64 a1, char *a2)
{
  if ( a2[192] >= 0 )
  {
    sub_684B30(1854, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  else
  {
    a2[202] |= 2u;
    if ( unk_4F077BC )
      a2[203] |= 0x60u;
    return a2;
  }
}
