// Function: sub_1610C30
// Address: 0x1610c30
//
__int64 __fastcall sub_1610C30(__int64 a1)
{
  __int64 v1; // r13
  _QWORD *v3; // rdi

  v1 = a1 - 160;
  v3 = (_QWORD *)(a1 + 408);
  *(v3 - 71) = &unk_49ED910;
  *(v3 - 51) = &unk_49ED9C8;
  *v3 = &unk_49EDA08;
  sub_1610730((__int64)v3);
  sub_160F3F0(a1);
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 1320);
}
