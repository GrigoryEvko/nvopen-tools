// Function: ctor_732
// Address: 0x5c60a0
//
int ctor_732()
{
  int v0; // edx

  qword_5057620 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505762C &= 0xF000u;
  qword_5057630 = 0;
  qword_5057668 = (__int64)qword_4FA01C0;
  qword_5057638 = 0;
  qword_5057640 = 0;
  qword_5057648 = 0;
  dword_5057628 = v0;
  qword_5057678 = (__int64)&unk_5057698;
  qword_5057680 = (__int64)&unk_5057698;
  qword_5057650 = 0;
  qword_5057658 = 0;
  qword_50576C8 = (__int64)&unk_49E74E8;
  word_50576D0 = 256;
  qword_5057660 = 0;
  qword_5057670 = 0;
  qword_5057620 = (__int64)&unk_49EEC70;
  qword_5057688 = 4;
  byte_50576B8 = 0;
  qword_50576D8 = (__int64)&unk_49EEDB0;
  dword_5057690 = 0;
  byte_50576C0 = 0;
  sub_16B8280(&qword_5057620, "simplify-mir", 12);
  qword_5057650 = 51;
  LOBYTE(word_505762C) = word_505762C & 0x9F | 0x20;
  qword_5057648 = (__int64)"Leave out unnecessary information when printing MIR";
  sub_16B88A0(&qword_5057620);
  return __cxa_atexit(sub_12EDEC0, &qword_5057620, &qword_4A427C0);
}
