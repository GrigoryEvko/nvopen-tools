// Function: ctor_720
// Address: 0x5c0d60
//
int ctor_720()
{
  char *v0; // rbx
  int v1; // edx

  v0 = getenv("AS_SECURE_LOG_FILE");
  qword_5052900 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505290C &= 0xF000u;
  qword_5052910 = 0;
  qword_5052948 = (__int64)qword_4FA01C0;
  qword_5052918 = 0;
  qword_5052920 = 0;
  qword_5052928 = 0;
  dword_5052908 = v1;
  qword_5052958 = (__int64)&unk_5052978;
  qword_5052960 = (__int64)&unk_5052978;
  qword_50529C8 = (__int64)&qword_5052900;
  qword_5052930 = 0;
  qword_50529A8 = (__int64)&unk_4A3E008;
  qword_5052938 = 0;
  qword_5052940 = 0;
  qword_5052900 = (__int64)&unk_4A3E078;
  qword_5052950 = 0;
  byte_5052998 = 0;
  qword_50529C0 = (__int64)&unk_4A3E028;
  qword_50529D0 = (__int64)&unk_50529E0;
  qword_50529D8 = 0x800000000LL;
  qword_5052968 = 4;
  dword_5052970 = 0;
  qword_50529A0 = 0;
  byte_50529B8 = 1;
  qword_50529B0 = 0;
  sub_16B8280(&qword_5052900, "as-secure-log-file-name", 23);
  qword_50529A0 = (__int64)v0;
  qword_5052928 = (__int64)"As secure log file name (initialized from AS_SECURE_LOG_FILE env variable)";
  qword_50529B0 = (__int64)v0;
  byte_50529B8 = 1;
  qword_5052930 = 74;
  LOBYTE(word_505290C) = word_505290C & 0x9F | 0x20;
  sub_16B88A0(&qword_5052900);
  return __cxa_atexit(sub_38BB880, &qword_5052900, &qword_4A427C0);
}
