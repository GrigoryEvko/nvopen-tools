// Function: sub_22C3210
// Address: 0x22c3210
//
void __fastcall sub_22C3210(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 22);
  *(_QWORD *)(v2 - 176) = &unk_4A09DB8;
  sub_22C31B0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
