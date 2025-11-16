// Function: sub_30A3C30
// Address: 0x30a3c30
//
__int64 __fastcall sub_30A3C30(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 176);
  *(v3 - 22) = off_4A31E10;
  *v3 = &unk_4A31EC8;
  sub_B81E70((__int64)v3, a2);
  return sub_BB9260(a1);
}
