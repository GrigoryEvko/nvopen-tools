// Function: sub_67C3A0
// Address: 0x67c3a0
//
__int64 __fastcall sub_67C3A0(__int64 a1)
{
  __int64 v2; // rdi

  if ( !unk_4F0697C )
    return sub_87D380(a1, &qword_4CFFDC0);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
    return sub_87D380(a1, &qword_4CFFDC0);
  v2 = sub_67C320(*(_QWORD *)(a1 + 64));
  if ( !v2 )
    return sub_87D380(a1, &qword_4CFFDC0);
  sub_87D380(v2, &qword_4CFFDC0);
  sub_8238B0(qword_4D039E8, "::", 2);
  return sub_87D250(a1, &qword_4CFFDC0, 1);
}
