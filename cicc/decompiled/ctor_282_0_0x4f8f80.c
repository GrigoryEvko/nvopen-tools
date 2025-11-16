// Function: ctor_282_0
// Address: 0x4f8f80
//
int ctor_282_0()
{
  int v0; // eax
  int v1; // esi
  int v2; // edi
  int v3; // edi
  int v4; // eax
  int v5; // eax
  char v7; // [rsp+23h] [rbp-4Dh] BYREF
  int v8; // [rsp+24h] [rbp-4Ch] BYREF
  char *v9; // [rsp+28h] [rbp-48h] BYREF
  const char *v10; // [rsp+30h] [rbp-40h]
  __int64 v11; // [rsp+38h] [rbp-38h]

  sub_1D03E50(&unk_4FC14A0, "list-burr", "Bottom-up register reduction list scheduling", sub_1D05200);
  __cxa_atexit(sub_1CFC0C0, &unk_4FC14A0, &qword_4A427C0);
  sub_1D03E50(&unk_4FC1460, "source", "Similar to list-burr but schedules in source order when possible", sub_1D05510);
  __cxa_atexit(sub_1CFC0C0, &unk_4FC1460, &qword_4A427C0);
  sub_1D03E50(
    &unk_4FC1420,
    "list-hybrid",
    "Bottom-up register pressure aware list scheduling which tries to balance latency and register pressure",
    sub_1D05820);
  __cxa_atexit(sub_1CFC0C0, &unk_4FC1420, &qword_4A427C0);
  sub_1D03E50(
    &unk_4FC13E0,
    "list-ilp",
    "Bottom-up register pressure aware list scheduling which tries to balance ILP and register pressure",
    sub_1D04DC0);
  __cxa_atexit(sub_1CFC0C0, &unk_4FC13E0, &qword_4A427C0);
  v10 = "Disable cycle-level precision during preRA scheduling";
  v11 = 53;
  v7 = 0;
  v9 = &v7;
  v8 = 1;
  sub_1D03EC0(&unk_4FC1300, "disable-sched-cycles", &v8, &v9);
  __cxa_atexit(sub_12EDEC0, &unk_4FC1300, &qword_4A427C0);
  v7 = 0;
  v10 = "Disable regpressure priority in sched=list-ilp";
  v11 = 46;
  v9 = &v7;
  v8 = 1;
  sub_1D04040(&unk_4FC1220, "disable-sched-reg-pressure", &v8, &v9);
  __cxa_atexit(sub_12EDEC0, &unk_4FC1220, &qword_4A427C0);
  qword_4FC1140 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FC11D8 = 0;
  word_4FC114C &= 0xF000u;
  qword_4FC1150 = 0;
  qword_4FC1158 = 0;
  qword_4FC1160 = 0;
  qword_4FC1168 = 0;
  dword_4FC1148 = v0;
  qword_4FC1170 = 0;
  qword_4FC1188 = (__int64)qword_4FA01C0;
  qword_4FC1198 = (__int64)&unk_4FC11B8;
  qword_4FC11A0 = (__int64)&unk_4FC11B8;
  qword_4FC1178 = 0;
  qword_4FC1180 = 0;
  qword_4FC11E8 = (__int64)&unk_49E74E8;
  word_4FC11F0 = 256;
  qword_4FC1190 = 0;
  qword_4FC11A8 = 4;
  qword_4FC1140 = (__int64)&unk_49EEC70;
  qword_4FC11F8 = (__int64)&unk_49EEDB0;
  dword_4FC11B0 = 0;
  byte_4FC11E0 = 0;
  sub_16B8280(&qword_4FC1140, "disable-sched-live-uses", 23);
  word_4FC11F0 = 257;
  byte_4FC11E0 = 1;
  qword_4FC1170 = 43;
  qword_4FC1168 = (__int64)"Disable live use priority in sched=list-ilp";
  LOBYTE(word_4FC114C) = word_4FC114C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC1140);
  __cxa_atexit(sub_12EDEC0, &qword_4FC1140, &qword_4A427C0);
  qword_4FC1060 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FC10F8 = 0;
  word_4FC1110 = 256;
  qword_4FC1070 = 0;
  word_4FC106C &= 0xF000u;
  qword_4FC1118 = (__int64)&unk_49EEDB0;
  qword_4FC1060 = (__int64)&unk_49EEC70;
  dword_4FC1068 = v1;
  qword_4FC10B8 = (__int64)&unk_4FC10D8;
  qword_4FC10C0 = (__int64)&unk_4FC10D8;
  qword_4FC10A8 = (__int64)qword_4FA01C0;
  qword_4FC1108 = (__int64)&unk_49E74E8;
  qword_4FC1078 = 0;
  qword_4FC1080 = 0;
  qword_4FC1088 = 0;
  qword_4FC1090 = 0;
  qword_4FC1098 = 0;
  qword_4FC10A0 = 0;
  qword_4FC10B0 = 0;
  qword_4FC10C8 = 4;
  dword_4FC10D0 = 0;
  byte_4FC1100 = 0;
  sub_16B8280(&qword_4FC1060, "disable-sched-vrcycle", 21);
  qword_4FC1088 = (__int64)"Disable virtual register cycle interference checks";
  byte_4FC1100 = 0;
  qword_4FC1090 = 50;
  word_4FC1110 = 256;
  LOBYTE(word_4FC106C) = word_4FC106C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC1060);
  __cxa_atexit(sub_12EDEC0, &qword_4FC1060, &qword_4A427C0);
  v11 = 32;
  v10 = "Disable physreg def-use affinity";
  v7 = 0;
  v9 = &v7;
  v8 = 1;
  sub_1D04040(&unk_4FC0F80, "disable-sched-physreg-join", &v8, &v9);
  __cxa_atexit(sub_12EDEC0, &unk_4FC0F80, &qword_4A427C0);
  v11 = 43;
  v10 = "Disable no-stall priority in sched=list-ilp";
  v7 = 1;
  v9 = &v7;
  v8 = 1;
  sub_1D03EC0(&unk_4FC0EA0, "disable-sched-stalls", &v8, &v9);
  __cxa_atexit(sub_12EDEC0, &unk_4FC0EA0, &qword_4A427C0);
  qword_4FC0DC0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FC0E58 = 0;
  word_4FC0DCC &= 0xF000u;
  qword_4FC0DD0 = 0;
  qword_4FC0DD8 = 0;
  qword_4FC0E78 = (__int64)&unk_49EEDB0;
  qword_4FC0DC0 = (__int64)&unk_49EEC70;
  dword_4FC0DC8 = v2;
  qword_4FC0E18 = (__int64)&unk_4FC0E38;
  qword_4FC0E08 = (__int64)qword_4FA01C0;
  qword_4FC0E20 = (__int64)&unk_4FC0E38;
  word_4FC0E70 = 256;
  qword_4FC0E68 = (__int64)&unk_49E74E8;
  qword_4FC0DE0 = 0;
  qword_4FC0DE8 = 0;
  qword_4FC0DF0 = 0;
  qword_4FC0DF8 = 0;
  qword_4FC0E00 = 0;
  qword_4FC0E10 = 0;
  qword_4FC0E28 = 4;
  dword_4FC0E30 = 0;
  byte_4FC0E60 = 0;
  sub_16B8280(&qword_4FC0DC0, "disable-sched-critical-path", 27);
  word_4FC0E70 = 256;
  byte_4FC0E60 = 0;
  qword_4FC0DF0 = 48;
  qword_4FC0DE8 = (__int64)"Disable critical path priority in sched=list-ilp";
  LOBYTE(word_4FC0DCC) = word_4FC0DCC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC0DC0);
  __cxa_atexit(sub_12EDEC0, &qword_4FC0DC0, &qword_4A427C0);
  v9 = &v7;
  v10 = "Disable scheduled-height priority in sched=list-ilp";
  v11 = 51;
  v7 = 0;
  v8 = 1;
  sub_1D03EC0(&unk_4FC0CE0, "disable-sched-height", &v8, &v9);
  __cxa_atexit(sub_12EDEC0, &unk_4FC0CE0, &qword_4A427C0);
  qword_4FC0C00 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC0C0C &= 0xF000u;
  word_4FC0CB0 = 256;
  qword_4FC0C10 = 0;
  qword_4FC0C18 = 0;
  qword_4FC0CB8 = (__int64)&unk_49EEDB0;
  qword_4FC0C00 = (__int64)&unk_49EEC70;
  dword_4FC0C08 = v3;
  qword_4FC0C58 = (__int64)&unk_4FC0C78;
  qword_4FC0C60 = (__int64)&unk_4FC0C78;
  qword_4FC0C48 = (__int64)qword_4FA01C0;
  qword_4FC0CA8 = (__int64)&unk_49E74E8;
  qword_4FC0C20 = 0;
  qword_4FC0C28 = 0;
  qword_4FC0C30 = 0;
  qword_4FC0C38 = 0;
  qword_4FC0C40 = 0;
  qword_4FC0C50 = 0;
  qword_4FC0C68 = 4;
  dword_4FC0C70 = 0;
  byte_4FC0C98 = 0;
  byte_4FC0CA0 = 0;
  sub_16B8280(&qword_4FC0C00, "disable-2addr-hack", 18);
  word_4FC0CB0 = 257;
  byte_4FC0CA0 = 1;
  qword_4FC0C30 = 36;
  LOBYTE(word_4FC0C0C) = word_4FC0C0C & 0x9F | 0x20;
  qword_4FC0C28 = (__int64)"Disable scheduler's two-address hack";
  sub_16B88A0(&qword_4FC0C00);
  __cxa_atexit(sub_12EDEC0, &qword_4FC0C00, &qword_4A427C0);
  qword_4FC0B20 = (__int64)&unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC0B2C &= 0xF000u;
  qword_4FC0B30 = 0;
  qword_4FC0B38 = 0;
  qword_4FC0B40 = 0;
  qword_4FC0B48 = 0;
  qword_4FC0B50 = 0;
  dword_4FC0B28 = v4;
  qword_4FC0B58 = 0;
  qword_4FC0B68 = (__int64)qword_4FA01C0;
  qword_4FC0B78 = (__int64)&unk_4FC0B98;
  qword_4FC0B80 = (__int64)&unk_4FC0B98;
  qword_4FC0B60 = 0;
  qword_4FC0B70 = 0;
  qword_4FC0BC8 = (__int64)&unk_49E74C8;
  qword_4FC0B88 = 4;
  dword_4FC0B90 = 0;
  qword_4FC0B20 = (__int64)&unk_49EEB70;
  byte_4FC0BB8 = 0;
  dword_4FC0BC0 = 0;
  qword_4FC0BD8 = (__int64)&unk_49EEDF0;
  byte_4FC0BD4 = 1;
  dword_4FC0BD0 = 0;
  sub_16B8280(&qword_4FC0B20, "max-sched-reorder", 17);
  dword_4FC0BC0 = 6;
  byte_4FC0BD4 = 1;
  dword_4FC0BD0 = 6;
  qword_4FC0B50 = 76;
  LOBYTE(word_4FC0B2C) = word_4FC0B2C & 0x9F | 0x20;
  qword_4FC0B48 = (__int64)"Number of instructions to allow ahead of the critical path in sched=list-ilp";
  sub_16B88A0(&qword_4FC0B20);
  __cxa_atexit(sub_12EDEA0, &qword_4FC0B20, &qword_4A427C0);
  qword_4FC0A40 = (__int64)&unk_49EED30;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC0A4C &= 0xF000u;
  qword_4FC0A50 = 0;
  qword_4FC0A58 = 0;
  qword_4FC0A60 = 0;
  qword_4FC0A68 = 0;
  qword_4FC0A70 = 0;
  dword_4FC0A48 = v5;
  qword_4FC0A78 = 0;
  qword_4FC0A88 = (__int64)qword_4FA01C0;
  qword_4FC0A98 = (__int64)&unk_4FC0AB8;
  qword_4FC0AA0 = (__int64)&unk_4FC0AB8;
  qword_4FC0A80 = 0;
  qword_4FC0A90 = 0;
  qword_4FC0AE8 = (__int64)&unk_49E74A8;
  qword_4FC0AA8 = 4;
  dword_4FC0AB0 = 0;
  qword_4FC0A40 = (__int64)&unk_49EEAF0;
  byte_4FC0AD8 = 0;
  dword_4FC0AE0 = 0;
  qword_4FC0AF8 = (__int64)&unk_49EEE10;
  byte_4FC0AF4 = 1;
  dword_4FC0AF0 = 0;
  sub_16B8280(&qword_4FC0A40, "sched-avg-ipc", 13);
  dword_4FC0AE0 = 1;
  byte_4FC0AF4 = 1;
  dword_4FC0AF0 = 1;
  qword_4FC0A70 = 51;
  LOBYTE(word_4FC0A4C) = word_4FC0A4C & 0x9F | 0x20;
  qword_4FC0A68 = (__int64)"Average inst/cycle whan no target itinerary exists.";
  sub_16B88A0(&qword_4FC0A40);
  return __cxa_atexit(sub_12EDE60, &qword_4FC0A40, &qword_4A427C0);
}
