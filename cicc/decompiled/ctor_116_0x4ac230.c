// Function: ctor_116
// Address: 0x4ac230
//
int ctor_116()
{
  int v0; // eax
  int v1; // eax
  int result; // eax

  qword_4F98860 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9886C &= 0xF000u;
  qword_4F988A8 = (__int64)&unk_4FA01C0;
  qword_4F98870 = 0;
  qword_4F98878 = 0;
  qword_4F98880 = 0;
  dword_4F98868 = v0;
  qword_4F988B8 = (__int64)&unk_4F988D8;
  qword_4F988C0 = (__int64)&unk_4F988D8;
  qword_4F98888 = 0;
  qword_4F98890 = 0;
  qword_4F98908 = (__int64)&unk_49E74E8;
  word_4F98910 = 256;
  qword_4F98898 = 0;
  qword_4F988A0 = 0;
  qword_4F98860 = (__int64)&unk_49EEC70;
  qword_4F988B0 = 0;
  byte_4F988F8 = 0;
  qword_4F98918 = (__int64)&unk_49EEDB0;
  qword_4F988C8 = 4;
  dword_4F988D0 = 0;
  byte_4F98900 = 0;
  sub_16B8280(&qword_4F98860, "print-bpi", 9);
  word_4F98910 = 256;
  byte_4F98900 = 0;
  qword_4F98890 = 34;
  LOBYTE(word_4F9886C) = word_4F9886C & 0x9F | 0x20;
  qword_4F98888 = (__int64)"Print the branch probability info.";
  sub_16B88A0(&qword_4F98860);
  __cxa_atexit(sub_12EDEC0, &qword_4F98860, &qword_4A427C0);
  qword_4F98740 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9874C &= 0xF000u;
  qword_4F98750 = 0;
  qword_4F98758 = 0;
  qword_4F98760 = 0;
  qword_4F98768 = 0;
  qword_4F98770 = 0;
  dword_4F98748 = v1;
  qword_4F98798 = (__int64)&unk_4F987B8;
  qword_4F987A0 = (__int64)&unk_4F987B8;
  qword_4F987E0 = &byte_4F987F0;
  qword_4F98808 = (__int64)&byte_4F98818;
  qword_4F98788 = (__int64)&unk_4FA01C0;
  qword_4F98778 = 0;
  qword_4F98800 = (__int64)&unk_49EED10;
  qword_4F98780 = 0;
  qword_4F98790 = 0;
  qword_4F98740 = (__int64)&unk_49EEBF0;
  qword_4F987A8 = 4;
  dword_4F987B0 = 0;
  qword_4F98830 = (__int64)&unk_49EEE90;
  qword_4F98838 = (__int64)&byte_4F98848;
  byte_4F987D8 = 0;
  qword_4F987E8 = 0;
  byte_4F987F0 = 0;
  qword_4F98810 = 0;
  byte_4F98818 = 0;
  byte_4F98828 = 0;
  qword_4F98840 = 0;
  byte_4F98848 = 0;
  sub_16B8280(&qword_4F98740, "print-bpi-func-name", 19);
  qword_4F98770 = 88;
  LOBYTE(word_4F9874C) = word_4F9874C & 0x9F | 0x20;
  qword_4F98768 = (__int64)"The option to specify the name of the function whose branch probability info is printed.";
  sub_16B88A0(&qword_4F98740);
  result = __cxa_atexit(sub_12F0C20, &qword_4F98740, &qword_4A427C0);
  dword_4F98720 = 1;
  return result;
}
