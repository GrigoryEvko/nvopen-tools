// Function: sub_2EE81C0
// Address: 0x2ee81c0
//
void __fastcall sub_2EE81C0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = &unk_4A2A2B8;
  sub_2EE80E0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
