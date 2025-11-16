// Function: sub_B820B0
// Address: 0xb820b0
//
__int64 __fastcall sub_B820B0(_QWORD *a1, __int64 a2)
{
  *(a1 - 22) = &unk_49DAA78;
  *a1 = &unk_49DAB30;
  sub_B81E70((__int64)a1, a2);
  sub_BB9260(a1 - 22);
  return j_j___libc_free_0(a1 - 22, 568);
}
