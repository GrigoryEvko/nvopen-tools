// Function: sub_13EB5B0
// Address: 0x13eb5b0
//
__int64 __fastcall sub_13EB5B0(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 20;
  *(v2 - 20) = (__int64)&unk_49EA918;
  sub_13EB550(v2);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 200);
}
