// Function: sub_13E65D0
// Address: 0x13e65d0
//
__int64 __fastcall sub_13E65D0(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 20;
  *(v2 - 20) = (__int64)&unk_49EA7A0;
  sub_1368A00(v2);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 200);
}
