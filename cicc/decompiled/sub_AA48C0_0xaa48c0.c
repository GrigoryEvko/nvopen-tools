// Function: sub_AA48C0
// Address: 0xaa48c0
//
void __fastcall sub_AA48C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi

  *(_QWORD *)(a2 + 40) = a1 - 48;
  sub_AA48B0(a1 - 48);
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v2 = sub_AA4890(a1 - 48);
    if ( v2 )
      sub_BD8920(v2, a2);
  }
}
