// Function: sub_160F6B0
// Address: 0x160f6b0
//
__int64 __fastcall sub_160F6B0(__int64 a1)
{
  _QWORD *v2; // rdi

  v2 = (_QWORD *)(a1 + 408);
  *(v2 - 51) = off_49EDAA8;
  *v2 = &unk_49EE078;
  sub_16366C0(v2);
  sub_160F3F0(a1);
  return j_j___libc_free_0(a1, 568);
}
