// Function: ctor_303
// Address: 0x4ff610
//
int ctor_303()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4FC6300 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC630C &= 0xF000u;
  qword_4FC6348 = (__int64)qword_4FA01C0;
  qword_4FC6358 = (__int64)&unk_4FC6378;
  qword_4FC6360 = (__int64)&unk_4FC6378;
  qword_4FC6310 = 0;
  dword_4FC6308 = v0;
  qword_4FC63A8 = (__int64)&unk_49E74E8;
  qword_4FC6318 = 0;
  qword_4FC6320 = 0;
  qword_4FC6300 = (__int64)&unk_49EAB58;
  qword_4FC6328 = 0;
  qword_4FC6330 = 0;
  qword_4FC6338 = 0;
  qword_4FC6340 = 0;
  qword_4FC6350 = 0;
  qword_4FC6368 = 4;
  dword_4FC6370 = 0;
  byte_4FC6398 = 0;
  qword_4FC63A0 = 0;
  byte_4FC63B1 = 0;
  qword_4FC63B8 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FC6300, "verify-machine-dom-info", 23);
  if ( qword_4FC63A0 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4FC6300, &v3, 0, 0, v1);
  }
  else
  {
    byte_4FC63B1 = 1;
    qword_4FC63A0 = (__int64)&byte_4FC63C8;
    byte_4FC63B0 = byte_4FC63C8;
  }
  qword_4FC6330 = 46;
  LOBYTE(word_4FC630C) = word_4FC630C & 0x9F | 0x20;
  qword_4FC6328 = (__int64)"Verify machine dominator info (time consuming)";
  sub_16B88A0(&qword_4FC6300);
  return __cxa_atexit(sub_13F9A70, &qword_4FC6300, &qword_4A427C0);
}
