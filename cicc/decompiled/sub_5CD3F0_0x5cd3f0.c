// Function: sub_5CD3F0
// Address: 0x5cd3f0
//
__int64 __fastcall sub_5CD3F0(__int64 a1, __int64 a2, char a3)
{
  char *v5; // rdi

  v5 = "r:+x!|v:+x!";
  if ( unk_4F077A8 >= 0x9D08u )
    v5 = "r:-x!|v:-x!";
  sub_5CCB50(v5, a1, a2, a3);
  if ( a3 == 11 )
  {
    *(_BYTE *)(a2 + 200) |= 0x60u;
  }
  else
  {
    if ( a3 != 7 )
      sub_721090(v5);
    *(_BYTE *)(a2 + 168) |= 0x18u;
  }
  if ( *(_QWORD *)(a1 + 32) )
    return sub_5C7D80(a1, a2, a3);
  else
    return a2;
}
