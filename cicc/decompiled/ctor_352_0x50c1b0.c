// Function: ctor_352
// Address: 0x50c1b0
//
int ctor_352()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4FCF980 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCF98C &= 0xF000u;
  qword_4FCF9C8 = (__int64)qword_4FA01C0;
  qword_4FCF9D8 = (__int64)&unk_4FCF9F8;
  qword_4FCF9E0 = (__int64)&unk_4FCF9F8;
  qword_4FCF990 = 0;
  dword_4FCF988 = v0;
  qword_4FCFA28 = (__int64)&unk_49E74E8;
  qword_4FCF998 = 0;
  qword_4FCF9A0 = 0;
  qword_4FCF980 = (__int64)&unk_49EAB58;
  qword_4FCF9A8 = 0;
  qword_4FCF9B0 = 0;
  qword_4FCF9B8 = 0;
  qword_4FCF9C0 = 0;
  qword_4FCF9D0 = 0;
  qword_4FCF9E8 = 4;
  dword_4FCF9F0 = 0;
  byte_4FCFA18 = 0;
  qword_4FCFA20 = 0;
  byte_4FCFA31 = 0;
  qword_4FCFA38 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FCF980, "verify-regalloc", 15);
  if ( qword_4FCFA20 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4FCF980, &v3, 0, 0, v1);
  }
  else
  {
    byte_4FCFA31 = 1;
    qword_4FCFA20 = (__int64)byte_4FCF965;
    byte_4FCFA30 = byte_4FCF965[0];
  }
  qword_4FCF9B0 = 33;
  LOBYTE(word_4FCF98C) = word_4FCF98C & 0x9F | 0x20;
  qword_4FCF9A8 = (__int64)"Verify during register allocation";
  sub_16B88A0(&qword_4FCF980);
  return __cxa_atexit(sub_13F9A70, &qword_4FCF980, &qword_4A427C0);
}
