// Function: sub_2F43430
// Address: 0x2f43430
//
__int64 __fastcall sub_2F43430(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 25);
  *(_QWORD *)(v2 - 200) = off_4A2AF00;
  sub_2F43140(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
