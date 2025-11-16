// Function: ctor_139
// Address: 0x4b35d0
//
int ctor_139()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4F9DA20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9DA2C &= 0xF000u;
  qword_4F9DA30 = 0;
  qword_4F9DA38 = 0;
  qword_4F9DA40 = 0;
  qword_4F9DA48 = 0;
  qword_4F9DA50 = 0;
  dword_4F9DA28 = v0;
  qword_4F9DA58 = 0;
  qword_4F9DA68 = (__int64)&unk_4FA01C0;
  qword_4F9DA78 = (__int64)&unk_4F9DA98;
  qword_4F9DA80 = (__int64)&unk_4F9DA98;
  qword_4F9DA60 = 0;
  qword_4F9DA70 = 0;
  qword_4F9DAC8 = (__int64)&unk_49E74A8;
  qword_4F9DA88 = 4;
  qword_4F9DA20 = (__int64)&unk_49EEAF0;
  dword_4F9DA90 = 0;
  qword_4F9DAD8 = (__int64)&unk_49EEE10;
  byte_4F9DAB8 = 0;
  dword_4F9DAC0 = 0;
  byte_4F9DAD4 = 1;
  dword_4F9DAD0 = 0;
  sub_16B8280(&qword_4F9DA20, "icp-remaining-percent-threshold", 31);
  dword_4F9DAC0 = 30;
  byte_4F9DAD4 = 1;
  dword_4F9DAD0 = 30;
  qword_4F9DA50 = 91;
  LOBYTE(word_4F9DA2C) = word_4F9DA2C & 0x98 | 0x21;
  qword_4F9DA48 = (__int64)"The percentage threshold against remaining unpromoted indirect call count for the promotion";
  sub_16B88A0(&qword_4F9DA20);
  __cxa_atexit(sub_12EDE60, &qword_4F9DA20, &qword_4A427C0);
  qword_4F9D940 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D94C &= 0xF000u;
  qword_4F9D950 = 0;
  qword_4F9D958 = 0;
  qword_4F9D960 = 0;
  qword_4F9D968 = 0;
  qword_4F9D970 = 0;
  dword_4F9D948 = v1;
  qword_4F9D9E8 = (__int64)&unk_49E74A8;
  qword_4F9D988 = (__int64)&unk_4FA01C0;
  qword_4F9D998 = (__int64)&unk_4F9D9B8;
  qword_4F9D9A0 = (__int64)&unk_4F9D9B8;
  qword_4F9D940 = (__int64)&unk_49EEAF0;
  qword_4F9D9F8 = (__int64)&unk_49EEE10;
  qword_4F9D978 = 0;
  qword_4F9D980 = 0;
  qword_4F9D990 = 0;
  qword_4F9D9A8 = 4;
  dword_4F9D9B0 = 0;
  byte_4F9D9D8 = 0;
  dword_4F9D9E0 = 0;
  byte_4F9D9F4 = 1;
  dword_4F9D9F0 = 0;
  sub_16B8280(&qword_4F9D940, "icp-total-percent-threshold", 27);
  dword_4F9D9E0 = 5;
  byte_4F9D9F4 = 1;
  dword_4F9D9F0 = 5;
  qword_4F9D970 = 62;
  LOBYTE(word_4F9D94C) = word_4F9D94C & 0x98 | 0x21;
  qword_4F9D968 = (__int64)"The percentage threshold against total count for the promotion";
  sub_16B88A0(&qword_4F9D940);
  __cxa_atexit(sub_12EDE60, &qword_4F9D940, &qword_4A427C0);
  qword_4F9D860 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9D86C &= 0xF000u;
  qword_4F9D870 = 0;
  qword_4F9D878 = 0;
  qword_4F9D880 = 0;
  qword_4F9D888 = 0;
  qword_4F9D890 = 0;
  dword_4F9D868 = v2;
  qword_4F9D908 = (__int64)&unk_49E74A8;
  qword_4F9D8A8 = (__int64)&unk_4FA01C0;
  qword_4F9D8B8 = (__int64)&unk_4F9D8D8;
  qword_4F9D8C0 = (__int64)&unk_4F9D8D8;
  qword_4F9D860 = (__int64)&unk_49EEAF0;
  qword_4F9D918 = (__int64)&unk_49EEE10;
  qword_4F9D898 = 0;
  qword_4F9D8A0 = 0;
  qword_4F9D8B0 = 0;
  qword_4F9D8C8 = 4;
  dword_4F9D8D0 = 0;
  byte_4F9D8F8 = 0;
  dword_4F9D900 = 0;
  byte_4F9D914 = 1;
  dword_4F9D910 = 0;
  sub_16B8280(&qword_4F9D860, "icp-max-prom", 12);
  dword_4F9D900 = 3;
  byte_4F9D914 = 1;
  dword_4F9D910 = 3;
  qword_4F9D890 = 60;
  LOBYTE(word_4F9D86C) = word_4F9D86C & 0x98 | 0x21;
  qword_4F9D888 = (__int64)"Max number of promotions for a single indirect call callsite";
  sub_16B88A0(&qword_4F9D860);
  return __cxa_atexit(sub_12EDE60, &qword_4F9D860, &qword_4A427C0);
}
