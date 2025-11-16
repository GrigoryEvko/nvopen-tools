// Function: sub_73E6E0
// Address: 0x73e6e0
//
_BYTE *__fastcall sub_73E6E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rdi

  *(_QWORD *)(a1 + 16) = a2;
  v2 = sub_72CBE0();
  v3 = sub_73DBF0(0x56u, v2, a1);
  if ( unk_4D04810 )
    v3[60] |= 2u;
  return sub_732B10((__int64)v3);
}
