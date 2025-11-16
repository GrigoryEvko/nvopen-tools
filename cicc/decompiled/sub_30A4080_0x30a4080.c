// Function: sub_30A4080
// Address: 0x30a4080
//
void __fastcall sub_30A4080(_QWORD *a1, __int64 a2)
{
  *(a1 - 22) = off_4A31E10;
  *a1 = &unk_4A31EC8;
  sub_B81E70((__int64)a1, a2);
  sub_BB9260((__int64)(a1 - 22));
  j_j___libc_free_0((unsigned __int64)(a1 - 22));
}
