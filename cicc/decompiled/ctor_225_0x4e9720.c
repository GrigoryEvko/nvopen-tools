// Function: ctor_225
// Address: 0x4e9720
//
int ctor_225()
{
  int v0; // edx
  char v2; // [rsp+3h] [rbp-4Dh] BYREF
  int v3; // [rsp+4h] [rbp-4Ch] BYREF
  char *v4; // [rsp+8h] [rbp-48h] BYREF
  const char *v5; // [rsp+10h] [rbp-40h] BYREF
  __int64 v6; // [rsp+18h] [rbp-38h]

  v5 = "Checking sinking scheduling effect";
  v6 = 34;
  v3 = 1;
  v2 = 1;
  v4 = &v2;
  sub_1A63790(&unk_4FB4A00, "sink-check-sched", &v4, &v3, &v5);
  __cxa_atexit(sub_12EDEC0, &unk_4FB4A00, &qword_4A427C0);
  v6 = 36;
  v5 = "Sinking single-use only instructions";
  v3 = 1;
  v2 = 1;
  v4 = &v2;
  sub_1A63790(&unk_4FB4920, "sink-single-only", &v4, &v3, &v5);
  __cxa_atexit(sub_12EDEC0, &unk_4FB4920, &qword_4A427C0);
  qword_4FB4840 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB484C &= 0xF000u;
  qword_4FB4850 = 0;
  qword_4FB4888 = (__int64)qword_4FA01C0;
  qword_4FB4858 = 0;
  qword_4FB4860 = 0;
  qword_4FB4868 = 0;
  dword_4FB4848 = v0;
  qword_4FB4898 = (__int64)&unk_4FB48B8;
  qword_4FB48A0 = (__int64)&unk_4FB48B8;
  qword_4FB4870 = 0;
  qword_4FB4878 = 0;
  qword_4FB48E8 = (__int64)&unk_49E74C8;
  qword_4FB4880 = 0;
  qword_4FB4890 = 0;
  qword_4FB4840 = (__int64)&unk_49EEB70;
  qword_4FB48A8 = 4;
  dword_4FB48B0 = 0;
  qword_4FB48F8 = (__int64)&unk_49EEDF0;
  byte_4FB48D8 = 0;
  dword_4FB48E0 = 0;
  byte_4FB48F4 = 1;
  dword_4FB48F0 = 0;
  sub_16B8280(&qword_4FB4840, "sink-level", 10);
  dword_4FB48E0 = 10;
  byte_4FB48F4 = 1;
  dword_4FB48F0 = 10;
  qword_4FB4870 = 21;
  LOBYTE(word_4FB484C) = word_4FB484C & 0x9F | 0x20;
  qword_4FB4868 = (__int64)"Control sinking level";
  sub_16B88A0(&qword_4FB4840);
  return __cxa_atexit(sub_12EDEA0, &qword_4FB4840, &qword_4A427C0);
}
