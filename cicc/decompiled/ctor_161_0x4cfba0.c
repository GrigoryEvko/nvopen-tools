// Function: ctor_161
// Address: 0x4cfba0
//
int ctor_161()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4FA1160 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA116C &= 0xF000u;
  qword_4FA11A8 = (__int64)qword_4FA01C0;
  qword_4FA11B8 = (__int64)&unk_4FA11D8;
  qword_4FA11C0 = (__int64)&unk_4FA11D8;
  qword_4FA1170 = 0;
  dword_4FA1168 = v0;
  qword_4FA1208 = (__int64)&unk_49E74E8;
  qword_4FA1178 = 0;
  qword_4FA1180 = 0;
  qword_4FA1160 = (__int64)&unk_49EAB58;
  qword_4FA1188 = 0;
  qword_4FA1190 = 0;
  qword_4FA1218 = (__int64)&unk_49EEDB0;
  qword_4FA1198 = 0;
  qword_4FA11A0 = 0;
  qword_4FA11B0 = 0;
  qword_4FA11C8 = 4;
  dword_4FA11D0 = 0;
  byte_4FA11F8 = 0;
  qword_4FA1200 = 0;
  byte_4FA1211 = 0;
  sub_16B8280(&qword_4FA1160, "disable-symbolication", 21);
  qword_4FA1190 = 37;
  qword_4FA1188 = (__int64)"Disable symbolizing crash backtraces.";
  if ( qword_4FA1200 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4FA1160, &v3, 0, 0, v1);
  }
  else
  {
    byte_4FA1211 = 1;
    qword_4FA1200 = (__int64)&byte_4FA1228;
    byte_4FA1210 = byte_4FA1228;
  }
  LOBYTE(word_4FA116C) = word_4FA116C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA1160);
  return __cxa_atexit(sub_13F9A70, &qword_4FA1160, &qword_4A427C0);
}
