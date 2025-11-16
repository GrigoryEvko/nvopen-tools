// Function: sub_1610DF0
// Address: 0x1610df0
//
__int64 __fastcall sub_1610DF0(__int64 a1)
{
  __int64 v1; // r13
  _QWORD *v3; // rdi

  v1 = a1 - 160;
  v3 = (_QWORD *)(a1 + 408);
  *(v3 - 71) = &unk_49ED7E8;
  *(v3 - 51) = &unk_49ED8A0;
  *v3 = &unk_49ED8E0;
  sub_1610730((__int64)v3);
  sub_160F3F0(a1);
  sub_16366C0(v1);
  return j_j___libc_free_0(v1, 1312);
}
