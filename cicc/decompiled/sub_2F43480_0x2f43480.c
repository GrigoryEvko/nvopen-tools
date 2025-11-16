// Function: sub_2F43480
// Address: 0x2f43480
//
void __fastcall sub_2F43480(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = off_4A2AF00;
  sub_2F43140(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
