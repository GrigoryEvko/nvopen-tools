// Function: sub_1610D80
// Address: 0x1610d80
//
__int64 __fastcall sub_1610D80(_QWORD *a1)
{
  *(a1 - 71) = &unk_49ED7E8;
  *(a1 - 51) = &unk_49ED8A0;
  *a1 = &unk_49ED8E0;
  sub_1610730((__int64)a1);
  sub_160F3F0((__int64)(a1 - 51));
  sub_16366C0(a1 - 71);
  return j_j___libc_free_0(a1 - 71, 1312);
}
