// Function: sub_1610BC0
// Address: 0x1610bc0
//
__int64 __fastcall sub_1610BC0(__int64 a1)
{
  _QWORD *v2; // rdi

  v2 = (_QWORD *)(a1 + 568);
  *(v2 - 71) = &unk_49ED910;
  *(v2 - 51) = &unk_49ED9C8;
  *v2 = &unk_49EDA08;
  sub_1610730((__int64)v2);
  sub_160F3F0(a1 + 160);
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 1320);
}
