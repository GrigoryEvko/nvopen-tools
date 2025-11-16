// Function: sub_2FACEA0
// Address: 0x2facea0
//
__int64 __fastcall sub_2FACEA0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = &unk_4A2BFF8;
  sub_2FACD60(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
