// Function: sub_1610D10
// Address: 0x1610d10
//
__int64 __fastcall sub_1610D10(__int64 a1)
{
  _QWORD *v2; // rdi

  v2 = (_QWORD *)(a1 + 568);
  *(v2 - 71) = &unk_49ED7E8;
  *(v2 - 51) = &unk_49ED8A0;
  *v2 = &unk_49ED8E0;
  sub_1610730((__int64)v2);
  sub_160F3F0(a1 + 160);
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 1312);
}
