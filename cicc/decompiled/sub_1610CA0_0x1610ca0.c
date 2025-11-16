// Function: sub_1610CA0
// Address: 0x1610ca0
//
__int64 __fastcall sub_1610CA0(_QWORD *a1)
{
  *(a1 - 71) = &unk_49ED910;
  *(a1 - 51) = &unk_49ED9C8;
  *a1 = &unk_49EDA08;
  sub_1610730((__int64)a1);
  sub_160F3F0((__int64)(a1 - 51));
  sub_16366C0(a1 - 71);
  return j_j___libc_free_0(a1 - 71, 1320);
}
