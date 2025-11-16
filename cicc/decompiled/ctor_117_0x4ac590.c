// Function: ctor_117
// Address: 0x4ac590
//
int ctor_117()
{
  int v0; // edx

  qword_4F98960 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9896C &= 0xF000u;
  qword_4F98970 = 0;
  qword_4F989A8 = (__int64)&unk_4FA01C0;
  qword_4F98978 = 0;
  qword_4F98980 = 0;
  qword_4F98988 = 0;
  dword_4F98968 = v0;
  qword_4F989B8 = (__int64)&unk_4F989D8;
  qword_4F989C0 = (__int64)&unk_4F989D8;
  qword_4F98A00 = (__int64)&byte_4F98A10;
  qword_4F98A28 = (__int64)&byte_4F98A38;
  qword_4F98990 = 0;
  qword_4F98998 = 0;
  qword_4F98A20 = (__int64)&unk_49EED10;
  qword_4F989A0 = 0;
  qword_4F989B0 = 0;
  qword_4F98960 = (__int64)&unk_49EEBF0;
  qword_4F989C8 = 4;
  dword_4F989D0 = 0;
  qword_4F98A50 = (__int64)&unk_49EEE90;
  qword_4F98A58 = (__int64)&byte_4F98A68;
  byte_4F989F8 = 0;
  qword_4F98A08 = 0;
  byte_4F98A10 = 0;
  qword_4F98A30 = 0;
  byte_4F98A38 = 0;
  byte_4F98A48 = 0;
  qword_4F98A60 = 0;
  byte_4F98A68 = 0;
  sub_16B8280(&qword_4F98960, "cfg-func-name", 13);
  qword_4F98990 = 70;
  LOBYTE(word_4F9896C) = word_4F9896C & 0x9F | 0x20;
  qword_4F98988 = (__int64)"The name of a function (or its substring) whose CFG is viewed/printed.";
  sub_16B88A0(&qword_4F98960);
  return __cxa_atexit(sub_12F0C20, &qword_4F98960, &qword_4A427C0);
}
