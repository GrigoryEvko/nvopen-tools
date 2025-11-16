// Function: sub_2E22EC0
// Address: 0x2e22ec0
//
__int64 __fastcall sub_2E22EC0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = &unk_4A285A8;
  sub_2E22D50(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
