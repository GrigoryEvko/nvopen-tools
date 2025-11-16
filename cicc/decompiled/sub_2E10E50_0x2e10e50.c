// Function: sub_2E10E50
// Address: 0x2e10e50
//
void __fastcall sub_2E10E50(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = &unk_4A28410;
  sub_2E10BB0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
