// Function: sub_13EB560
// Address: 0x13eb560
//
__int64 __fastcall sub_13EB560(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 20;
  *(v2 - 20) = (__int64)&unk_49EA918;
  sub_13EB550(v2);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
