// Function: ctor_123
// Address: 0x4ad1f0
//
int ctor_123()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4F99220 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9922C &= 0xF000u;
  qword_4F99268 = (__int64)&unk_4FA01C0;
  qword_4F99278 = (__int64)&unk_4F99298;
  qword_4F99280 = (__int64)&unk_4F99298;
  qword_4F99230 = 0;
  dword_4F99228 = v0;
  qword_4F992C8 = (__int64)&unk_49E74E8;
  qword_4F99238 = 0;
  qword_4F99240 = 0;
  qword_4F99220 = (__int64)&unk_49EAB58;
  qword_4F99248 = 0;
  qword_4F99250 = 0;
  qword_4F99258 = 0;
  qword_4F99260 = 0;
  qword_4F99270 = 0;
  qword_4F99288 = 4;
  dword_4F99290 = 0;
  byte_4F992B8 = 0;
  qword_4F992C0 = 0;
  byte_4F992D1 = 0;
  qword_4F992D8 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4F99220, "verify-loop-info", 16);
  if ( qword_4F992C0 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4F99220, &v3, 0, 0, v1);
  }
  else
  {
    byte_4F992D1 = 1;
    qword_4F992C0 = (__int64)byte_4F992E8;
    byte_4F992D0 = byte_4F992E8[0];
  }
  qword_4F99250 = 33;
  LOBYTE(word_4F9922C) = word_4F9922C & 0x9F | 0x20;
  qword_4F99248 = (__int64)"Verify loop info (time consuming)";
  sub_16B88A0(&qword_4F99220);
  return __cxa_atexit(sub_13F9A70, &qword_4F99220, &qword_4A427C0);
}
