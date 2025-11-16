// Function: sub_2EC1E10
// Address: 0x2ec1e10
//
void __fastcall sub_2EC1E10(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = off_4A29BB8;
  *v2 = &unk_4A29A90;
  sub_2EC14D0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
