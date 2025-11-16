// Function: ctor_189
// Address: 0x4dd940
//
int ctor_189()
{
  int v0; // eax
  int v1; // eax

  qword_4FAC6C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAC6CC &= 0xF000u;
  qword_4FAC708 = (__int64)qword_4FA01C0;
  qword_4FAC6D0 = 0;
  qword_4FAC6D8 = 0;
  qword_4FAC6E0 = 0;
  dword_4FAC6C8 = v0;
  qword_4FAC718 = (__int64)&unk_4FAC738;
  qword_4FAC720 = (__int64)&unk_4FAC738;
  qword_4FAC6E8 = 0;
  qword_4FAC6F0 = 0;
  qword_4FAC768 = (__int64)&unk_49E74A8;
  qword_4FAC6F8 = 0;
  qword_4FAC700 = 0;
  qword_4FAC6C0 = (__int64)&unk_49EEAF0;
  qword_4FAC710 = 0;
  byte_4FAC758 = 0;
  qword_4FAC778 = (__int64)&unk_49EEE10;
  qword_4FAC728 = 4;
  dword_4FAC730 = 0;
  dword_4FAC760 = 0;
  byte_4FAC774 = 1;
  dword_4FAC770 = 0;
  sub_16B8280(&qword_4FAC6C0, "mergefunc-sanity", 16);
  qword_4FAC6F0 = 135;
  qword_4FAC6E8 = (__int64)"How many functions in module could be used for MergeFunctions pass sanity check. '0' disables"
                           " this check. Works only with '-debug' key.";
  dword_4FAC760 = 0;
  byte_4FAC774 = 1;
  dword_4FAC770 = 0;
  LOBYTE(word_4FAC6CC) = word_4FAC6CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAC6C0);
  __cxa_atexit(sub_12EDE60, &qword_4FAC6C0, &qword_4A427C0);
  qword_4FAC5E0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAC5EC &= 0xF000u;
  qword_4FAC5F0 = 0;
  qword_4FAC5F8 = 0;
  qword_4FAC600 = 0;
  qword_4FAC608 = 0;
  qword_4FAC610 = 0;
  dword_4FAC5E8 = v1;
  qword_4FAC638 = (__int64)&unk_4FAC658;
  qword_4FAC640 = (__int64)&unk_4FAC658;
  qword_4FAC628 = (__int64)qword_4FA01C0;
  qword_4FAC618 = 0;
  qword_4FAC688 = (__int64)&unk_49E74E8;
  word_4FAC690 = 256;
  qword_4FAC620 = 0;
  qword_4FAC630 = 0;
  qword_4FAC5E0 = (__int64)&unk_49EEC70;
  qword_4FAC648 = 4;
  byte_4FAC678 = 0;
  qword_4FAC698 = (__int64)&unk_49EEDB0;
  dword_4FAC650 = 0;
  byte_4FAC680 = 0;
  sub_16B8280(&qword_4FAC5E0, "mergefunc-preserve-debug-info", 29);
  word_4FAC690 = 256;
  byte_4FAC680 = 0;
  qword_4FAC610 = 69;
  LOBYTE(word_4FAC5EC) = word_4FAC5EC & 0x9F | 0x20;
  qword_4FAC608 = (__int64)"Preserve debug info in thunk when mergefunc transformations are made.";
  sub_16B88A0(&qword_4FAC5E0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FAC5E0, &qword_4A427C0);
}
