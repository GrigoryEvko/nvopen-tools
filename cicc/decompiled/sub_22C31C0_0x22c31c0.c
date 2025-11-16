// Function: sub_22C31C0
// Address: 0x22c31c0
//
__int64 __fastcall sub_22C31C0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 22);
  *(_QWORD *)(v2 - 176) = &unk_4A09DB8;
  sub_22C31B0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
