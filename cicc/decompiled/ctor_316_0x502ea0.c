// Function: ctor_316
// Address: 0x502ea0
//
int ctor_316()
{
  int v0; // eax
  int v1; // eax
  int v3; // [rsp+0h] [rbp-50h] BYREF
  int v4; // [rsp+4h] [rbp-4Ch] BYREF
  int *v5; // [rsp+8h] [rbp-48h] BYREF
  const char *v6; // [rsp+10h] [rbp-40h] BYREF
  __int64 v7; // [rsp+18h] [rbp-38h]

  qword_4FC8FA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC8FAC &= 0xF000u;
  qword_4FC8FE8 = (__int64)qword_4FA01C0;
  qword_4FC8FB0 = 0;
  qword_4FC8FB8 = 0;
  qword_4FC8FC0 = 0;
  dword_4FC8FA8 = v0;
  qword_4FC8FF8 = (__int64)&unk_4FC9018;
  qword_4FC9000 = (__int64)&unk_4FC9018;
  qword_4FC8FC8 = 0;
  qword_4FC8FD0 = 0;
  qword_4FC9048 = (__int64)&unk_49E74E8;
  word_4FC9050 = 256;
  qword_4FC8FD8 = 0;
  qword_4FC8FE0 = 0;
  qword_4FC8FA0 = (__int64)&unk_49EEC70;
  qword_4FC8FF0 = 0;
  byte_4FC9038 = 0;
  qword_4FC9058 = (__int64)&unk_49EEDB0;
  qword_4FC9008 = 4;
  dword_4FC9010 = 0;
  byte_4FC9040 = 0;
  sub_16B8280(&qword_4FC8FA0, "post-RA-scheduler", 17);
  qword_4FC8FC8 = (__int64)"Enable scheduling after register allocation";
  word_4FC9050 = 256;
  byte_4FC9040 = 0;
  qword_4FC8FD0 = 43;
  LOBYTE(word_4FC8FAC) = word_4FC8FAC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC8FA0);
  __cxa_atexit(sub_12EDEC0, &qword_4FC8FA0, &qword_4A427C0);
  v4 = 1;
  v5 = (int *)"none";
  v6 = "Break post-RA scheduling anti-dependencies: \"critical\", \"all\", or \"none\"";
  v7 = 72;
  qword_4FC8E80 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FC8F18 = 0;
  word_4FC8E8C &= 0xF000u;
  qword_4FC8E90 = 0;
  qword_4FC8E98 = 0;
  qword_4FC8EA0 = 0;
  qword_4FC8EA8 = 0;
  dword_4FC8E88 = v1;
  qword_4FC8ED8 = (__int64)&unk_4FC8EF8;
  qword_4FC8EE0 = (__int64)&unk_4FC8EF8;
  qword_4FC8F20 = (__int64)&byte_4FC8F30;
  qword_4FC8F48 = (__int64)&byte_4FC8F58;
  qword_4FC8EC8 = (__int64)qword_4FA01C0;
  qword_4FC8EB0 = 0;
  qword_4FC8F40 = (__int64)&unk_49EED10;
  qword_4FC8EB8 = 0;
  qword_4FC8EC0 = 0;
  qword_4FC8E80 = (__int64)&unk_49EEBF0;
  qword_4FC8ED0 = 0;
  qword_4FC8EE8 = 4;
  qword_4FC8F70 = (__int64)&unk_49EEE90;
  qword_4FC8F78 = (__int64)&byte_4FC8F88;
  dword_4FC8EF0 = 0;
  qword_4FC8F28 = 0;
  byte_4FC8F30 = 0;
  qword_4FC8F50 = 0;
  byte_4FC8F58 = 0;
  byte_4FC8F68 = 0;
  qword_4FC8F80 = 0;
  byte_4FC8F88 = 0;
  sub_1EAAA50(&qword_4FC8E80, "break-anti-dependencies", &v6, &v5, &v4);
  sub_16B88A0(&qword_4FC8E80);
  __cxa_atexit(sub_12F0C20, &qword_4FC8E80, &qword_4A427C0);
  v5 = &v4;
  v6 = "Debug control MBBs that are scheduled";
  v3 = 1;
  v4 = 0;
  v7 = 37;
  ((void (__fastcall *)(void *, const char *, const char **, int **, int *))sub_1EAA8D0)(
    &unk_4FC8DA0,
    "postra-sched-debugdiv",
    &v6,
    &v5,
    &v3);
  __cxa_atexit(sub_12EDEA0, &unk_4FC8DA0, &qword_4A427C0);
  v5 = &v4;
  v3 = 1;
  v4 = 0;
  v6 = "Debug control MBBs that are scheduled";
  v7 = 37;
  ((void (__fastcall *)(void *, const char *, const char **, int **, int *, const char *))sub_1EAA8D0)(
    &unk_4FC8CC0,
    "postra-sched-debugmod",
    &v6,
    &v5,
    &v3,
    "Debug control MBBs that are scheduled");
  return __cxa_atexit(sub_12EDEA0, &unk_4FC8CC0, &qword_4A427C0);
}
