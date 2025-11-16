// Function: ctor_183
// Address: 0x4da630
//
int ctor_183()
{
  int v0; // eax
  int v1; // eax

  qword_4FAA6A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAA6AC &= 0xF000u;
  qword_4FAA6B0 = 0;
  qword_4FAA6B8 = 0;
  qword_4FAA6C0 = 0;
  qword_4FAA6C8 = 0;
  qword_4FAA6D0 = 0;
  dword_4FAA6A8 = v0;
  qword_4FAA6D8 = 0;
  qword_4FAA6E8 = (__int64)qword_4FA01C0;
  qword_4FAA6F8 = (__int64)&unk_4FAA718;
  qword_4FAA700 = (__int64)&unk_4FAA718;
  qword_4FAA6E0 = 0;
  qword_4FAA6F0 = 0;
  word_4FAA750 = 256;
  qword_4FAA748 = (__int64)&unk_49E74E8;
  qword_4FAA708 = 4;
  qword_4FAA6A0 = (__int64)&unk_49EEC70;
  byte_4FAA738 = 0;
  qword_4FAA758 = (__int64)&unk_49EEDB0;
  dword_4FAA710 = 0;
  byte_4FAA740 = 0;
  sub_16B8280(&qword_4FAA6A0, "enable-nonnull-arg-prop", 23);
  qword_4FAA6D0 = 80;
  LOBYTE(word_4FAA6AC) = word_4FAA6AC & 0x9F | 0x20;
  qword_4FAA6C8 = (__int64)"Try to propagate nonnull argument attributes from callsites to caller functions.";
  sub_16B88A0(&qword_4FAA6A0);
  __cxa_atexit(sub_12EDEC0, &qword_4FAA6A0, &qword_4A427C0);
  qword_4FAA5C0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAA5CC &= 0xF000u;
  word_4FAA670 = 256;
  qword_4FAA5D0 = 0;
  qword_4FAA5D8 = 0;
  qword_4FAA5E0 = 0;
  qword_4FAA5E8 = 0;
  dword_4FAA5C8 = v1;
  qword_4FAA668 = (__int64)&unk_49E74E8;
  qword_4FAA608 = (__int64)qword_4FA01C0;
  qword_4FAA618 = (__int64)&unk_4FAA638;
  qword_4FAA620 = (__int64)&unk_4FAA638;
  qword_4FAA5C0 = (__int64)&unk_49EEC70;
  qword_4FAA678 = (__int64)&unk_49EEDB0;
  qword_4FAA5F0 = 0;
  qword_4FAA5F8 = 0;
  qword_4FAA600 = 0;
  qword_4FAA610 = 0;
  qword_4FAA628 = 4;
  dword_4FAA630 = 0;
  byte_4FAA658 = 0;
  byte_4FAA660 = 0;
  sub_16B8280(&qword_4FAA5C0, "disable-nounwind-inference", 26);
  qword_4FAA5F0 = 60;
  LOBYTE(word_4FAA5CC) = word_4FAA5CC & 0x9F | 0x20;
  qword_4FAA5E8 = (__int64)"Stop inferring nounwind attribute during function-attrs pass";
  sub_16B88A0(&qword_4FAA5C0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FAA5C0, &qword_4A427C0);
}
