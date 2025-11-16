// Function: ctor_106
// Address: 0x4a61d0
//
int ctor_106()
{
  int v0; // edx

  getenv("bar");
  if ( getenv("bar") == (char *)-1LL )
  {
    sub_1EB6E00();
    sub_1EB4F60();
    sub_1EBDCD0();
    sub_1ECC880();
    nullsub_693();
    nullsub_691();
    nullsub_692();
    sub_1D05200(0, 2);
    sub_1D05510(0, 2);
    sub_1D05820(0, 2);
    sub_1CFBF70(0, 2);
    sub_1D469E0(0, 2);
    sub_1D122D0(0, 2);
  }
  if ( getenv("bar") == (char *)-1LL )
    sub_4A5950();
  qword_4F92DC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F92DCC &= 0xF000u;
  qword_4F92E08 = (__int64)&unk_4FA01C0;
  qword_4F92E18 = (__int64)&unk_4F92E38;
  qword_4F92E20 = (__int64)&unk_4F92E38;
  qword_4F92DD0 = 0;
  dword_4F92DC8 = v0;
  qword_4F92E60 = (__int64)&unk_49E6A80;
  qword_4F92DD8 = 0;
  qword_4F92DE0 = 0;
  qword_4F92DC0 = (__int64)&unk_49E6AA0;
  qword_4F92DE8 = 0;
  qword_4F92DF0 = 0;
  qword_4F92E68 = (__int64)&unk_49EEE90;
  qword_4F92DF8 = 0;
  qword_4F92E00 = 0;
  qword_4F92E10 = 0;
  qword_4F92E28 = 4;
  dword_4F92E30 = 0;
  byte_4F92E58 = 0;
  sub_16B8280(&qword_4F92DC0, "load", 4);
  qword_4F92E00 = 14;
  qword_4F92DF0 = 25;
  LOBYTE(word_4F92DCC) = word_4F92DCC & 0xF8 | 1;
  qword_4F92DF8 = (__int64)"pluginfilename";
  qword_4F92DE8 = (__int64)"Load the specified plugin";
  sub_16B88A0(&qword_4F92DC0);
  __cxa_atexit(sub_12D3DC0, &qword_4F92DC0, &qword_4A427C0);
  sub_2208040(&unk_4F92DA0);
  return __cxa_atexit(sub_2208810, &unk_4F92DA0, &qword_4A427C0);
}
