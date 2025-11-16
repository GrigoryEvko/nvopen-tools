// Function: ctor_219
// Address: 0x4e7d60
//
int ctor_219()
{
  int v0; // eax
  int v1; // eax

  qword_4FB3A00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB3A0C &= 0xF000u;
  qword_4FB3A10 = 0;
  qword_4FB3A18 = 0;
  qword_4FB3A20 = 0;
  qword_4FB3A28 = 0;
  qword_4FB3A30 = 0;
  dword_4FB3A08 = v0;
  qword_4FB3A38 = 0;
  qword_4FB3A48 = (__int64)qword_4FA01C0;
  qword_4FB3A58 = (__int64)&unk_4FB3A78;
  qword_4FB3A60 = (__int64)&unk_4FB3A78;
  qword_4FB3A40 = 0;
  qword_4FB3A50 = 0;
  qword_4FB3AA8 = (__int64)&unk_49E74A8;
  qword_4FB3A68 = 4;
  qword_4FB3A00 = (__int64)&unk_49EEAF0;
  dword_4FB3A70 = 0;
  qword_4FB3AB8 = (__int64)&unk_49EEE10;
  byte_4FB3A98 = 0;
  dword_4FB3AA0 = 0;
  byte_4FB3AB4 = 1;
  dword_4FB3AB0 = 0;
  sub_16B8280(&qword_4FB3A00, "likely-branch-weight", 20);
  dword_4FB3AA0 = 2000;
  byte_4FB3AB4 = 1;
  dword_4FB3AB0 = 2000;
  qword_4FB3A30 = 56;
  LOBYTE(word_4FB3A0C) = word_4FB3A0C & 0x9F | 0x20;
  qword_4FB3A28 = (__int64)"Weight of the branch likely to be taken (default = 2000)";
  sub_16B88A0(&qword_4FB3A00);
  __cxa_atexit(sub_12EDE60, &qword_4FB3A00, &qword_4A427C0);
  qword_4FB3920 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB392C &= 0xF000u;
  qword_4FB3930 = 0;
  qword_4FB3938 = 0;
  qword_4FB3940 = 0;
  qword_4FB3948 = 0;
  qword_4FB3950 = 0;
  dword_4FB3928 = v1;
  qword_4FB39C8 = (__int64)&unk_49E74A8;
  qword_4FB3968 = (__int64)qword_4FA01C0;
  qword_4FB3978 = (__int64)&unk_4FB3998;
  qword_4FB3980 = (__int64)&unk_4FB3998;
  qword_4FB3920 = (__int64)&unk_49EEAF0;
  qword_4FB39D8 = (__int64)&unk_49EEE10;
  qword_4FB3958 = 0;
  qword_4FB3960 = 0;
  qword_4FB3970 = 0;
  qword_4FB3988 = 4;
  dword_4FB3990 = 0;
  byte_4FB39B8 = 0;
  dword_4FB39C0 = 0;
  byte_4FB39D4 = 1;
  dword_4FB39D0 = 0;
  sub_16B8280(&qword_4FB3920, "unlikely-branch-weight", 22);
  dword_4FB39C0 = 1;
  byte_4FB39D4 = 1;
  dword_4FB39D0 = 1;
  qword_4FB3950 = 55;
  LOBYTE(word_4FB392C) = word_4FB392C & 0x9F | 0x20;
  qword_4FB3948 = (__int64)"Weight of the branch unlikely to be taken (default = 1)";
  sub_16B88A0(&qword_4FB3920);
  return __cxa_atexit(sub_12EDE60, &qword_4FB3920, &qword_4A427C0);
}
