// Function: sub_2EC1E70
// Address: 0x2ec1e70
//
void __fastcall sub_2EC1E70(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = off_4A29AF0;
  *v2 = &unk_4A29A90;
  sub_2EC14D0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
