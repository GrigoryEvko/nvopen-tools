// Function: sub_30A4030
// Address: 0x30a4030
//
void __fastcall sub_30A4030(unsigned __int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 176);
  *(v3 - 22) = off_4A31E10;
  *v3 = &unk_4A31EC8;
  sub_B81E70((__int64)v3, a2);
  sub_BB9260(a1);
  j_j___libc_free_0(a1);
}
