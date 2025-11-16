// Function: ctor_327
// Address: 0x5057b0
//
int ctor_327()
{
  int v0; // edx

  qword_4FCA760 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCA76C &= 0xF000u;
  qword_4FCA770 = 0;
  qword_4FCA7A8 = (__int64)qword_4FA01C0;
  qword_4FCA778 = 0;
  qword_4FCA780 = 0;
  qword_4FCA788 = 0;
  dword_4FCA768 = v0;
  qword_4FCA7B8 = (__int64)&unk_4FCA7D8;
  qword_4FCA7C0 = (__int64)&unk_4FCA7D8;
  qword_4FCA790 = 0;
  qword_4FCA798 = 0;
  qword_4FCA808 = (__int64)&unk_49EECF0;
  qword_4FCA7A0 = 0;
  qword_4FCA7B0 = 0;
  qword_4FCA760 = (__int64)&unk_49FDF40;
  qword_4FCA7C8 = 4;
  dword_4FCA7D0 = 0;
  qword_4FCA818 = (__int64)&unk_49EEDD0;
  byte_4FCA7F8 = 0;
  dword_4FCA800 = 0;
  byte_4FCA814 = 1;
  dword_4FCA810 = 0;
  sub_16B8280(&qword_4FCA760, "enable-shrink-wrap", 18);
  qword_4FCA790 = 31;
  LOBYTE(word_4FCA76C) = word_4FCA76C & 0x9F | 0x20;
  qword_4FCA788 = (__int64)"enable the shrink-wrapping pass";
  sub_16B88A0(&qword_4FCA760);
  return __cxa_atexit(sub_1ED7DF0, &qword_4FCA760, &qword_4A427C0);
}
