// Function: sub_2E10E00
// Address: 0x2e10e00
//
__int64 __fastcall sub_2E10E00(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = &unk_4A28410;
  sub_2E10BB0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
