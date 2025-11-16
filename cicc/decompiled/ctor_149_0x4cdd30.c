// Function: ctor_149
// Address: 0x4cdd30
//
int ctor_149()
{
  int v0; // edx

  qword_4F9ED80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9ED8C &= 0xF000u;
  qword_4F9ED90 = 0;
  qword_4F9EDC8 = (__int64)&unk_4FA01C0;
  qword_4F9ED98 = 0;
  qword_4F9EDA0 = 0;
  qword_4F9EDA8 = 0;
  dword_4F9ED88 = v0;
  qword_4F9EDD8 = (__int64)&unk_4F9EDF8;
  qword_4F9EDE0 = (__int64)&unk_4F9EDF8;
  qword_4F9EDB0 = 0;
  qword_4F9EDB8 = 0;
  qword_4F9EE28 = (__int64)&unk_49E74C8;
  qword_4F9EDC0 = 0;
  qword_4F9EDD0 = 0;
  qword_4F9ED80 = (__int64)&unk_49EEB70;
  qword_4F9EDE8 = 4;
  dword_4F9EDF0 = 0;
  qword_4F9EE38 = (__int64)&unk_49EEDF0;
  byte_4F9EE18 = 0;
  dword_4F9EE20 = 0;
  byte_4F9EE34 = 1;
  dword_4F9EE30 = 0;
  sub_16B8280(&qword_4F9ED80, "opt-bisect-limit", 16);
  dword_4F9EE20 = 0x7FFFFFFF;
  byte_4F9EE34 = 1;
  dword_4F9EE30 = 0x7FFFFFFF;
  qword_4F9EDB0 = 31;
  LOBYTE(word_4F9ED8C) = word_4F9ED8C & 0x98 | 0x20;
  qword_4F9EDA8 = (__int64)"Maximum optimization to perform";
  sub_16B88A0(&qword_4F9ED80);
  return __cxa_atexit(sub_12EDEA0, &qword_4F9ED80, &qword_4A427C0);
}
