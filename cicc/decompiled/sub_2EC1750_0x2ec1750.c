// Function: sub_2EC1750
// Address: 0x2ec1750
//
__int64 __fastcall sub_2EC1750(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = off_4A29BB8;
  *v2 = &unk_4A29A90;
  sub_2EC14D0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
