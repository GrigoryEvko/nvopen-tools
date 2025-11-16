// Function: ctor_251
// Address: 0x4f0230
//
int ctor_251()
{
  int v0; // edx
  char v2; // [rsp+13h] [rbp-4Dh] BYREF
  int v3; // [rsp+14h] [rbp-4Ch] BYREF
  char *v4; // [rsp+18h] [rbp-48h] BYREF
  char *v5; // [rsp+20h] [rbp-40h] BYREF
  __int64 v6; // [rsp+28h] [rbp-38h]

  qword_4FB9D60 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB9D6C &= 0xF000u;
  qword_4FB9D70 = 0;
  qword_4FB9DA8 = (__int64)qword_4FA01C0;
  qword_4FB9D78 = 0;
  qword_4FB9D80 = 0;
  qword_4FB9D88 = 0;
  dword_4FB9D68 = v0;
  qword_4FB9DB8 = (__int64)&unk_4FB9DD8;
  qword_4FB9DC0 = (__int64)&unk_4FB9DD8;
  qword_4FB9D90 = 0;
  qword_4FB9D98 = 0;
  qword_4FB9E08 = (__int64)&unk_49E74E8;
  word_4FB9E10 = 256;
  qword_4FB9DA0 = 0;
  qword_4FB9DB0 = 0;
  qword_4FB9D60 = (__int64)&unk_49EEC70;
  qword_4FB9DC8 = 4;
  byte_4FB9DF8 = 0;
  qword_4FB9E18 = (__int64)&unk_49EEDB0;
  dword_4FB9DD0 = 0;
  byte_4FB9E00 = 0;
  sub_16B8280(&qword_4FB9D60, "nv-ocl", 6);
  qword_4FB9D88 = (__int64)"deprecated";
  word_4FB9E10 = 256;
  byte_4FB9E00 = 0;
  LOBYTE(word_4FB9D6C) = word_4FB9D6C & 0x9F | 0x20;
  qword_4FB9D90 = 10;
  sub_16B88A0(&qword_4FB9D60);
  __cxa_atexit(sub_12EDEC0, &qword_4FB9D60, &qword_4A427C0);
  v4 = &v2;
  v5 = "deprecated";
  v2 = 0;
  v3 = 1;
  v6 = 10;
  sub_1BF9970(&unk_4FB9C80, "nv-cuda", &v5, &v3, &v4);
  __cxa_atexit(sub_12EDEC0, &unk_4FB9C80, &qword_4A427C0);
  v5 = "deprecated";
  v4 = &v2;
  v2 = 0;
  v3 = 1;
  v6 = 10;
  sub_1BF9970(&unk_4FB9BA0, "drvcuda", &v5, &v3, &v4);
  __cxa_atexit(sub_12EDEC0, &unk_4FB9BA0, &qword_4A427C0);
  v5 = "deprecated";
  v2 = 0;
  v4 = &v2;
  v3 = 1;
  v6 = 10;
  sub_1BF9970(&unk_4FB9AC0, "drvnvcl", &v5, &v3, &v4);
  return __cxa_atexit(sub_12EDEC0, &unk_4FB9AC0, &qword_4A427C0);
}
