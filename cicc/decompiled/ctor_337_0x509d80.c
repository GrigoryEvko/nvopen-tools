// Function: ctor_337
// Address: 0x509d80
//
int ctor_337()
{
  int v0; // eax
  int v1; // eax

  qword_4FCE340 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCE34C &= 0xF000u;
  qword_4FCE388 = (__int64)qword_4FA01C0;
  qword_4FCE350 = 0;
  qword_4FCE358 = 0;
  qword_4FCE360 = 0;
  dword_4FCE348 = v0;
  qword_4FCE398 = (__int64)&unk_4FCE3B8;
  qword_4FCE3A0 = (__int64)&unk_4FCE3B8;
  qword_4FCE368 = 0;
  qword_4FCE370 = 0;
  qword_4FCE3E8 = (__int64)&unk_49E74E8;
  word_4FCE3F0 = 256;
  qword_4FCE378 = 0;
  qword_4FCE380 = 0;
  qword_4FCE340 = (__int64)&unk_49EEC70;
  qword_4FCE390 = 0;
  byte_4FCE3D8 = 0;
  qword_4FCE3F8 = (__int64)&unk_49EEDB0;
  qword_4FCE3A8 = 4;
  dword_4FCE3B0 = 0;
  byte_4FCE3E0 = 0;
  sub_16B8280(&qword_4FCE340, "twoaddr-reschedule", 18);
  qword_4FCE368 = (__int64)"Coalesce copies by rescheduling (default=true)";
  word_4FCE3F0 = 257;
  byte_4FCE3E0 = 1;
  qword_4FCE370 = 46;
  LOBYTE(word_4FCE34C) = word_4FCE34C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCE340);
  __cxa_atexit(sub_12EDEC0, &qword_4FCE340, &qword_4A427C0);
  qword_4FCE260 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCE26C &= 0xF000u;
  qword_4FCE270 = 0;
  qword_4FCE278 = 0;
  qword_4FCE280 = 0;
  qword_4FCE288 = 0;
  qword_4FCE290 = 0;
  dword_4FCE268 = v1;
  qword_4FCE2B8 = (__int64)&unk_4FCE2D8;
  qword_4FCE2C0 = (__int64)&unk_4FCE2D8;
  qword_4FCE2A8 = (__int64)qword_4FA01C0;
  qword_4FCE298 = 0;
  qword_4FCE308 = (__int64)&unk_49E74A8;
  qword_4FCE2A0 = 0;
  qword_4FCE2B0 = 0;
  qword_4FCE260 = (__int64)&unk_49EEAF0;
  qword_4FCE2C8 = 4;
  dword_4FCE2D0 = 0;
  qword_4FCE318 = (__int64)&unk_49EEE10;
  byte_4FCE2F8 = 0;
  dword_4FCE300 = 0;
  byte_4FCE314 = 1;
  dword_4FCE310 = 0;
  sub_16B8280(&qword_4FCE260, "dataflow-edge-limit", 19);
  dword_4FCE300 = 3;
  byte_4FCE314 = 1;
  dword_4FCE310 = 3;
  qword_4FCE290 = 94;
  LOBYTE(word_4FCE26C) = word_4FCE26C & 0x9F | 0x20;
  qword_4FCE288 = (__int64)"Maximum number of dataflow edges to traverse when evaluating the benefit of commuting operands";
  sub_16B88A0(&qword_4FCE260);
  return __cxa_atexit(sub_12EDE60, &qword_4FCE260, &qword_4A427C0);
}
