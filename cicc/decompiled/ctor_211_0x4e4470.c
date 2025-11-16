// Function: ctor_211
// Address: 0x4e4470
//
int ctor_211()
{
  int v0; // edx

  qword_4FB0D60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB0D6C &= 0xF000u;
  qword_4FB0D70 = 0;
  qword_4FB0DA8 = (__int64)qword_4FA01C0;
  qword_4FB0D78 = 0;
  qword_4FB0D80 = 0;
  qword_4FB0D88 = 0;
  dword_4FB0D68 = v0;
  qword_4FB0DB8 = (__int64)&unk_4FB0DD8;
  qword_4FB0DC0 = (__int64)&unk_4FB0DD8;
  qword_4FB0D90 = 0;
  qword_4FB0D98 = 0;
  qword_4FB0E08 = (__int64)&unk_49E74A8;
  qword_4FB0DA0 = 0;
  qword_4FB0DB0 = 0;
  qword_4FB0D60 = (__int64)&unk_49EEAF0;
  qword_4FB0DC8 = 4;
  dword_4FB0DD0 = 0;
  qword_4FB0E18 = (__int64)&unk_49EEE10;
  byte_4FB0DF8 = 0;
  dword_4FB0E00 = 0;
  byte_4FB0E14 = 1;
  dword_4FB0E10 = 0;
  sub_16B8280(&qword_4FB0D60, "reroll-num-tolerated-failed-matches", 35);
  dword_4FB0E00 = 400;
  byte_4FB0E14 = 1;
  dword_4FB0E10 = 400;
  qword_4FB0D90 = 80;
  LOBYTE(word_4FB0D6C) = word_4FB0D6C & 0x9F | 0x20;
  qword_4FB0D88 = (__int64)"The maximum number of failures to tolerate during fuzzy matching. (default: 400)";
  sub_16B88A0(&qword_4FB0D60);
  return __cxa_atexit(sub_12EDE60, &qword_4FB0D60, &qword_4A427C0);
}
