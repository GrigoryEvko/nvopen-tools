// Function: ctor_279
// Address: 0x4f87e0
//
int ctor_279()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4FC0640 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC064C &= 0xF000u;
  qword_4FC0688 = (__int64)qword_4FA01C0;
  qword_4FC0698 = (__int64)&unk_4FC06B8;
  qword_4FC06A0 = (__int64)&unk_4FC06B8;
  qword_4FC0650 = 0;
  dword_4FC0648 = v0;
  qword_4FC06E8 = (__int64)&unk_49E74E8;
  qword_4FC0658 = 0;
  qword_4FC0660 = 0;
  qword_4FC0640 = (__int64)&unk_49EAB58;
  qword_4FC0668 = 0;
  qword_4FC0670 = 0;
  qword_4FC06F8 = (__int64)&unk_49EEDB0;
  qword_4FC0678 = 0;
  qword_4FC0680 = 0;
  qword_4FC0690 = 0;
  qword_4FC06A8 = 4;
  dword_4FC06B0 = 0;
  byte_4FC06D8 = 0;
  qword_4FC06E0 = 0;
  byte_4FC06F1 = 0;
  sub_16B8280(&qword_4FC0640, "opt-unsafe-algebra", 18);
  qword_4FC0670 = 39;
  LOBYTE(word_4FC064C) = word_4FC064C & 0x98 | 0x21;
  qword_4FC0668 = (__int64)"Aggresive floating point simplification";
  if ( qword_4FC06E0 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4FC0640, &v3, 0, 0, v1);
  }
  else
  {
    byte_4FC06F1 = 1;
    qword_4FC06E0 = (__int64)byte_4FC0708;
    byte_4FC06F0 = byte_4FC0708[0];
  }
  sub_16B88A0(&qword_4FC0640);
  return __cxa_atexit(sub_13F9A70, &qword_4FC0640, &qword_4A427C0);
}
