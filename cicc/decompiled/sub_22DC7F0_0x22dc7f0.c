// Function: sub_22DC7F0
// Address: 0x22dc7f0
//
__int64 __fastcall sub_22DC7F0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 22);
  *(_QWORD *)(v2 - 176) = &unk_4A0A0E8;
  sub_22DC7A0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
