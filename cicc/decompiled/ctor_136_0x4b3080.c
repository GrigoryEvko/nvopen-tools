// Function: ctor_136
// Address: 0x4b3080
//
int ctor_136()
{
  int v0; // eax
  int v1; // eax

  qword_4F9D680 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D68C &= 0xF000u;
  qword_4F9D6C8 = (__int64)&unk_4FA01C0;
  qword_4F9D690 = 0;
  qword_4F9D698 = 0;
  qword_4F9D6A0 = 0;
  dword_4F9D688 = v0;
  qword_4F9D6D8 = (__int64)&unk_4F9D6F8;
  qword_4F9D6E0 = (__int64)&unk_4F9D6F8;
  qword_4F9D6A8 = 0;
  qword_4F9D6B0 = 0;
  qword_4F9D728 = (__int64)&unk_49E74E8;
  word_4F9D730 = 256;
  qword_4F9D6B8 = 0;
  qword_4F9D6C0 = 0;
  qword_4F9D680 = (__int64)&unk_49EEC70;
  qword_4F9D6D0 = 0;
  byte_4F9D718 = 0;
  qword_4F9D738 = (__int64)&unk_49EEDB0;
  qword_4F9D6E8 = 4;
  dword_4F9D6F0 = 0;
  byte_4F9D720 = 0;
  sub_16B8280(&qword_4F9D680, "imply-fp-condition", 18);
  word_4F9D730 = 257;
  byte_4F9D720 = 1;
  LOBYTE(word_4F9D68C) = word_4F9D68C & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9D680);
  __cxa_atexit(sub_12EDEC0, &qword_4F9D680, &qword_4A427C0);
  qword_4F9D5A0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D5AC &= 0xF000u;
  qword_4F9D5B0 = 0;
  qword_4F9D5B8 = 0;
  qword_4F9D5C0 = 0;
  qword_4F9D5C8 = 0;
  qword_4F9D5D0 = 0;
  dword_4F9D5A8 = v1;
  qword_4F9D5F8 = (__int64)&unk_4F9D618;
  qword_4F9D600 = (__int64)&unk_4F9D618;
  qword_4F9D5E8 = (__int64)&unk_4FA01C0;
  qword_4F9D5D8 = 0;
  qword_4F9D648 = (__int64)&unk_49E74A8;
  qword_4F9D5E0 = 0;
  qword_4F9D5F0 = 0;
  qword_4F9D5A0 = (__int64)&unk_49EEAF0;
  qword_4F9D608 = 4;
  dword_4F9D610 = 0;
  qword_4F9D658 = (__int64)&unk_49EEE10;
  byte_4F9D638 = 0;
  dword_4F9D640 = 0;
  byte_4F9D654 = 1;
  dword_4F9D650 = 0;
  sub_16B8280(&qword_4F9D5A0, "dom-conditions-max-uses", 23);
  dword_4F9D640 = 20;
  byte_4F9D654 = 1;
  dword_4F9D650 = 20;
  LOBYTE(word_4F9D5AC) = word_4F9D5AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9D5A0);
  return __cxa_atexit(sub_12EDE60, &qword_4F9D5A0, &qword_4A427C0);
}
