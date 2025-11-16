// Function: ctor_266
// Address: 0x4f5060
//
int ctor_266()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FBDB00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBDB0C &= 0xF000u;
  qword_4FBDB10 = 0;
  qword_4FBDB18 = 0;
  qword_4FBDB20 = 0;
  qword_4FBDB28 = 0;
  qword_4FBDB30 = 0;
  dword_4FBDB08 = v0;
  qword_4FBDB38 = 0;
  qword_4FBDB48 = (__int64)qword_4FA01C0;
  qword_4FBDB58 = (__int64)&unk_4FBDB78;
  qword_4FBDB60 = (__int64)&unk_4FBDB78;
  qword_4FBDB40 = 0;
  qword_4FBDB50 = 0;
  word_4FBDBB0 = 256;
  qword_4FBDBA8 = (__int64)&unk_49E74E8;
  qword_4FBDB68 = 4;
  qword_4FBDB00 = (__int64)&unk_49EEC70;
  byte_4FBDB98 = 0;
  qword_4FBDBB8 = (__int64)&unk_49EEDB0;
  dword_4FBDB70 = 0;
  byte_4FBDBA0 = 0;
  sub_16B8280(&qword_4FBDB00, "lsa-opt", 7);
  word_4FBDBB0 = 257;
  byte_4FBDBA0 = 1;
  qword_4FBDB30 = 47;
  LOBYTE(word_4FBDB0C) = word_4FBDB0C & 0x9F | 0x20;
  qword_4FBDB28 = (__int64)"Optimize copying of struct args to local memory";
  sub_16B88A0(&qword_4FBDB00);
  __cxa_atexit(sub_12EDEC0, &qword_4FBDB00, &qword_4A427C0);
  qword_4FBDA20 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBDAD0 = 256;
  word_4FBDA2C &= 0xF000u;
  qword_4FBDA30 = 0;
  qword_4FBDA38 = 0;
  qword_4FBDA40 = 0;
  dword_4FBDA28 = v1;
  qword_4FBDAC8 = (__int64)&unk_49E74E8;
  qword_4FBDA68 = (__int64)qword_4FA01C0;
  qword_4FBDA78 = (__int64)&unk_4FBDA98;
  qword_4FBDA80 = (__int64)&unk_4FBDA98;
  qword_4FBDA20 = (__int64)&unk_49EEC70;
  qword_4FBDAD8 = (__int64)&unk_49EEDB0;
  qword_4FBDA48 = 0;
  qword_4FBDA50 = 0;
  qword_4FBDA58 = 0;
  qword_4FBDA60 = 0;
  qword_4FBDA70 = 0;
  qword_4FBDA88 = 4;
  dword_4FBDA90 = 0;
  byte_4FBDAB8 = 0;
  byte_4FBDAC0 = 0;
  sub_16B8280(&qword_4FBDA20, "lower-read-only-devicefn-byval", 30);
  word_4FBDAD0 = 256;
  byte_4FBDAC0 = 0;
  qword_4FBDA50 = 60;
  LOBYTE(word_4FBDA2C) = word_4FBDA2C & 0x9F | 0x20;
  qword_4FBDA48 = (__int64)"Handling byval attribute of args to device functions as well";
  sub_16B88A0(&qword_4FBDA20);
  __cxa_atexit(sub_12EDEC0, &qword_4FBDA20, &qword_4A427C0);
  qword_4FBD940 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBD9F0 = 256;
  word_4FBD94C &= 0xF000u;
  qword_4FBD950 = 0;
  qword_4FBD958 = 0;
  qword_4FBD960 = 0;
  dword_4FBD948 = v2;
  qword_4FBD9E8 = (__int64)&unk_49E74E8;
  qword_4FBD988 = (__int64)qword_4FA01C0;
  qword_4FBD998 = (__int64)&unk_4FBD9B8;
  qword_4FBD9A0 = (__int64)&unk_4FBD9B8;
  qword_4FBD940 = (__int64)&unk_49EEC70;
  qword_4FBD9F8 = (__int64)&unk_49EEDB0;
  qword_4FBD968 = 0;
  qword_4FBD970 = 0;
  qword_4FBD978 = 0;
  qword_4FBD980 = 0;
  qword_4FBD990 = 0;
  qword_4FBD9A8 = 4;
  dword_4FBD9B0 = 0;
  byte_4FBD9D8 = 0;
  byte_4FBD9E0 = 0;
  sub_16B8280(&qword_4FBD940, "hoist-load-param", 16);
  byte_4FBD9E0 = 0;
  word_4FBD9F0 = 256;
  qword_4FBD970 = 40;
  LOBYTE(word_4FBD94C) = word_4FBD94C & 0x9F | 0x20;
  qword_4FBD968 = (__int64)"Generate all ld.param in the entry block";
  sub_16B88A0(&qword_4FBD940);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBD940, &qword_4A427C0);
}
