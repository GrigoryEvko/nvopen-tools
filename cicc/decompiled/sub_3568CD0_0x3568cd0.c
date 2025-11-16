// Function: sub_3568CD0
// Address: 0x3568cd0
//
__int64 __fastcall sub_3568CD0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = &unk_4A39380;
  sub_3568C80(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
