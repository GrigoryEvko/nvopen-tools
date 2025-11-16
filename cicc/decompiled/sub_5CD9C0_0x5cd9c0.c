// Function: sub_5CD9C0
// Address: 0x5cd9c0
//
__int64 __fastcall sub_5CD9C0(__int64 a1, __int64 a2, char a3)
{
  if ( a3 == 11 )
  {
    *(_BYTE *)(a2 + 196) |= 0x40u;
    if ( *(char *)(a2 + 192) < 0
      && (*(_BYTE *)(a1 + 9) == 2 || (*(_BYTE *)(a1 + 11) & 0x10) != 0)
      && (*(_BYTE *)(a2 + 89) & 4) != 0
      && (*(_BYTE *)(a2 + 203) & 1) == 0 )
    {
      sub_736C60(28, *(_QWORD *)(a2 + 104));
    }
  }
  else
  {
    sub_5CCAE0(unk_4F077B8 == 0 ? 8 : 5, a1);
  }
  return a2;
}
