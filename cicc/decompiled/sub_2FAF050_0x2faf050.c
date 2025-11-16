// Function: sub_2FAF050
// Address: 0x2faf050
//
void __fastcall sub_2FAF050(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = &unk_4A2C0C0;
  sub_2FAEF10(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
