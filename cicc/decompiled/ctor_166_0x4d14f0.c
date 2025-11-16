// Function: ctor_166
// Address: 0x4d14f0
//
int ctor_166()
{
  int v0; // edx

  qword_4FA2560 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA256C &= 0xF000u;
  qword_4FA2570 = 0;
  qword_4FA25A8 = (__int64)qword_4FA01C0;
  qword_4FA2578 = 0;
  qword_4FA2580 = 0;
  qword_4FA2588 = 0;
  dword_4FA2568 = v0;
  qword_4FA25B8 = (__int64)&unk_4FA25D8;
  qword_4FA25C0 = (__int64)&unk_4FA25D8;
  qword_4FA2590 = 0;
  qword_4FA2598 = 0;
  qword_4FA2608 = (__int64)&unk_49E74A8;
  qword_4FA25A0 = 0;
  qword_4FA25B0 = 0;
  qword_4FA2560 = (__int64)&unk_49EEAF0;
  qword_4FA25C8 = 4;
  dword_4FA25D0 = 0;
  qword_4FA2618 = (__int64)&unk_49EEE10;
  byte_4FA25F8 = 0;
  dword_4FA2600 = 0;
  byte_4FA2614 = 1;
  dword_4FA2610 = 0;
  sub_16B8280(&qword_4FA2560, "instcombine-guard-widening-window", 33);
  dword_4FA2600 = 3;
  byte_4FA2614 = 1;
  dword_4FA2610 = 3;
  qword_4FA2588 = (__int64)"How wide an instruction window to bypass looking for another guard";
  qword_4FA2590 = 66;
  sub_16B88A0(&qword_4FA2560);
  return __cxa_atexit(sub_12EDE60, &qword_4FA2560, &qword_4A427C0);
}
