// Function: ctor_366
// Address: 0x512000
//
int ctor_366()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4FD4220 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD422C &= 0xF000u;
  qword_4FD4268 = (__int64)qword_4FA01C0;
  qword_4FD4278 = (__int64)&unk_4FD4298;
  qword_4FD4280 = (__int64)&unk_4FD4298;
  qword_4FD4230 = 0;
  dword_4FD4228 = v0;
  qword_4FD42C8 = (__int64)&unk_49E74E8;
  qword_4FD4238 = 0;
  qword_4FD4240 = 0;
  qword_4FD4220 = (__int64)&unk_49EAB58;
  qword_4FD4248 = 0;
  qword_4FD4250 = 0;
  qword_4FD42D8 = (__int64)&unk_49EEDB0;
  qword_4FD4258 = 0;
  qword_4FD4260 = 0;
  qword_4FD4270 = 0;
  qword_4FD4288 = 4;
  dword_4FD4290 = 0;
  byte_4FD42B8 = 0;
  qword_4FD42C0 = 0;
  byte_4FD42D1 = 0;
  sub_16B8280(&qword_4FD4220, "nvptx-kernel-params-restrict", 28);
  qword_4FD4250 = 76;
  LOBYTE(word_4FD422C) = word_4FD422C & 0xF8 | 1;
  qword_4FD4248 = (__int64)"NVPTX Specific: Programmer asserts that any kernel ptr parameter is restrict";
  if ( qword_4FD42C0 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4FD4220, &v3, 0, 0, v1);
  }
  else
  {
    byte_4FD42D1 = 1;
    qword_4FD42C0 = (__int64)byte_4FD42E8;
    byte_4FD42D0 = byte_4FD42E8[0];
  }
  sub_16B88A0(&qword_4FD4220);
  return __cxa_atexit(sub_13F9A70, &qword_4FD4220, &qword_4A427C0);
}
