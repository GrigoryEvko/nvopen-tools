// Function: ctor_271
// Address: 0x4f6d60
//
int ctor_271()
{
  int v0; // eax
  int v1; // eax

  qword_4FBEC80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBEC8C &= 0xF000u;
  qword_4FBECC8 = (__int64)qword_4FA01C0;
  qword_4FBEC90 = 0;
  qword_4FBEC98 = 0;
  qword_4FBECA0 = 0;
  dword_4FBEC88 = v0;
  qword_4FBECD8 = (__int64)&unk_4FBECF8;
  qword_4FBECE0 = (__int64)&unk_4FBECF8;
  qword_4FBECA8 = 0;
  qword_4FBECB0 = 0;
  qword_4FBED28 = (__int64)&unk_49E74E8;
  word_4FBED30 = 256;
  qword_4FBECB8 = 0;
  qword_4FBECC0 = 0;
  qword_4FBEC80 = (__int64)&unk_49EEC70;
  qword_4FBECD0 = 0;
  byte_4FBED18 = 0;
  qword_4FBED38 = (__int64)&unk_49EEDB0;
  qword_4FBECE8 = 4;
  dword_4FBECF0 = 0;
  byte_4FBED20 = 0;
  sub_16B8280(&qword_4FBEC80, "nvvm-reflect-enable", 19);
  word_4FBED30 = 257;
  byte_4FBED20 = 1;
  qword_4FBECB0 = 35;
  LOBYTE(word_4FBEC8C) = word_4FBEC8C & 0x9F | 0x20;
  qword_4FBECA8 = (__int64)"NVVM reflection, enabled by default";
  sub_16B88A0(&qword_4FBEC80);
  __cxa_atexit(sub_12EDEC0, &qword_4FBEC80, &qword_4A427C0);
  qword_4FBEBA0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FBEBB0 = 0;
  qword_4FBEBB8 = 0;
  qword_4FBEBC0 = 0;
  qword_4FBEBC8 = 0;
  qword_4FBEBD0 = 0;
  qword_4FBEBD8 = 0;
  dword_4FBEBA8 = v1;
  qword_4FBEBE8 = (__int64)qword_4FA01C0;
  qword_4FBEBE0 = 0;
  qword_4FBEBF0 = 0;
  word_4FBEBAC = word_4FBEBAC & 0xF000 | 1;
  qword_4FBEBF8 = (__int64)&unk_4FBEC18;
  qword_4FBEC00 = (__int64)&unk_4FBEC18;
  qword_4FBEC08 = 4;
  dword_4FBEC10 = 0;
  qword_4FBEBA0 = (__int64)&unk_49E75F8;
  byte_4FBEC38 = 0;
  qword_4FBEC40 = 0;
  qword_4FBEC70 = (__int64)&unk_49EEE90;
  qword_4FBEC48 = 0;
  qword_4FBEC50 = 0;
  qword_4FBEC58 = 0;
  qword_4FBEC60 = 0;
  qword_4FBEC68 = 0;
  sub_16B8280(&qword_4FBEBA0, "R", 1);
  qword_4FBEBE0 = 10;
  qword_4FBEBD8 = (__int64)"name=<int>";
  qword_4FBEBC8 = (__int64)"A list of string=num assignments";
  qword_4FBEBD0 = 32;
  LOBYTE(word_4FBEBAC) = word_4FBEBAC & 0x87 | 0x30;
  sub_16B88A0(&qword_4FBEBA0);
  return __cxa_atexit(sub_12F08D0, &qword_4FBEBA0, &qword_4A427C0);
}
