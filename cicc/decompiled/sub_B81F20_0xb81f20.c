// Function: sub_B81F20
// Address: 0xb81f20
//
__int64 __fastcall sub_B81F20(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 176);
  *(v3 - 22) = &unk_49DAA78;
  *v3 = &unk_49DAB30;
  sub_B81E70((__int64)v3, a2);
  return sub_BB9260(a1);
}
