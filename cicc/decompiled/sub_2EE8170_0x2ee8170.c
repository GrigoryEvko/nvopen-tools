// Function: sub_2EE8170
// Address: 0x2ee8170
//
__int64 __fastcall sub_2EE8170(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = &unk_4A2A2B8;
  sub_2EE80E0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
