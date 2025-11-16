// Function: ctor_347
// Address: 0x50bb10
//
int ctor_347()
{
  int v0; // edx

  qword_4FCF5C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCF5CC &= 0xF000u;
  qword_4FCF5D0 = 0;
  qword_4FCF608 = (__int64)qword_4FA01C0;
  qword_4FCF5D8 = 0;
  qword_4FCF5E0 = 0;
  qword_4FCF5E8 = 0;
  dword_4FCF5C8 = v0;
  qword_4FCF618 = (__int64)&unk_4FCF638;
  qword_4FCF620 = (__int64)&unk_4FCF638;
  qword_4FCF5F0 = 0;
  qword_4FCF5F8 = 0;
  qword_4FCF668 = (__int64)&unk_49E74E8;
  word_4FCF670 = 256;
  qword_4FCF600 = 0;
  qword_4FCF610 = 0;
  qword_4FCF5C0 = (__int64)&unk_49EEC70;
  qword_4FCF628 = 4;
  byte_4FCF658 = 0;
  qword_4FCF678 = (__int64)&unk_49EEDB0;
  dword_4FCF630 = 0;
  byte_4FCF660 = 0;
  sub_16B8280(&qword_4FCF5C0, "verify-cfiinstrs", 16);
  qword_4FCF5E8 = (__int64)"Verify Call Frame Information instructions";
  word_4FCF670 = 256;
  byte_4FCF660 = 0;
  qword_4FCF5F0 = 42;
  LOBYTE(word_4FCF5CC) = word_4FCF5CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FCF5C0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCF5C0, &qword_4A427C0);
}
