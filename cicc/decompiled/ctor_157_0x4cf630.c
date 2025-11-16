// Function: ctor_157
// Address: 0x4cf630
//
int ctor_157()
{
  __int64 v0; // rbx
  int v1; // edx
  __int64 v2; // rax
  const char *v4; // [rsp+0h] [rbp-30h] BYREF
  char v5; // [rsp+10h] [rbp-20h]
  char v6; // [rsp+11h] [rbp-1Fh]

  v0 = sub_16BAF20();
  qword_4FA0260 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FA0270 = 0;
  qword_4FA0278 = 0;
  qword_4FA0280 = 0;
  word_4FA026C = word_4FA026C & 0xF000 | 1;
  qword_4FA0288 = 0;
  qword_4FA02A8 = (__int64)qword_4FA01C0;
  qword_4FA02B8 = (__int64)&unk_4FA02D8;
  qword_4FA02C0 = (__int64)&unk_4FA02D8;
  dword_4FA0268 = v1;
  qword_4FA0290 = 0;
  qword_4FA0260 = (__int64)&unk_49EF0D0;
  qword_4FA0298 = 0;
  qword_4FA02A0 = 0;
  qword_4FA0320 = (__int64)&unk_49EEE90;
  qword_4FA02B0 = 0;
  qword_4FA02C8 = 4;
  dword_4FA02D0 = 0;
  byte_4FA02F8 = 0;
  qword_4FA0300 = 0;
  qword_4FA0308 = 0;
  qword_4FA0310 = 0;
  qword_4FA0318 = 0;
  sub_16B8280(&qword_4FA0260, "debug-counter", 13);
  HIBYTE(word_4FA026C) |= 2u;
  qword_4FA0288 = (__int64)"Comma separated list of debug counter skip and count";
  qword_4FA0290 = 52;
  LOBYTE(word_4FA026C) = word_4FA026C & 0x98 | 0x21;
  if ( qword_4FA0300 )
  {
    v2 = sub_16E8CB0();
    v6 = 1;
    v4 = "cl::location(x) specified more than once!";
    v5 = 3;
    sub_16B1F90(&qword_4FA0260, &v4, 0, 0, v2);
  }
  else
  {
    qword_4FA0300 = v0;
  }
  sub_16B88A0(&qword_4FA0260);
  qword_4FA0260 = (__int64)off_49EF150;
  return __cxa_atexit(sub_16BA800, &qword_4FA0260, &qword_4A427C0);
}
