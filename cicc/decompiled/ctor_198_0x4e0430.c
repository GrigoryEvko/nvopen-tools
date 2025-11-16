// Function: ctor_198
// Address: 0x4e0430
//
int ctor_198()
{
  int v0; // eax
  int v1; // eax

  qword_4FAE440 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE44C &= 0xF000u;
  qword_4FAE450 = 0;
  qword_4FAE458 = 0;
  qword_4FAE460 = 0;
  qword_4FAE468 = 0;
  qword_4FAE470 = 0;
  dword_4FAE448 = v0;
  qword_4FAE478 = 0;
  qword_4FAE488 = (__int64)qword_4FA01C0;
  qword_4FAE498 = (__int64)&unk_4FAE4B8;
  qword_4FAE4A0 = (__int64)&unk_4FAE4B8;
  qword_4FAE480 = 0;
  qword_4FAE490 = 0;
  word_4FAE4F0 = 256;
  qword_4FAE4E8 = (__int64)&unk_49E74E8;
  qword_4FAE4A8 = 4;
  qword_4FAE440 = (__int64)&unk_49EEC70;
  byte_4FAE4D8 = 0;
  qword_4FAE4F8 = (__int64)&unk_49EEDB0;
  dword_4FAE4B0 = 0;
  byte_4FAE4E0 = 0;
  sub_16B8280(&qword_4FAE440, "enable-dse-partial-overwrite-tracking", 37);
  word_4FAE4F0 = 257;
  byte_4FAE4E0 = 1;
  qword_4FAE470 = 40;
  LOBYTE(word_4FAE44C) = word_4FAE44C & 0x9F | 0x20;
  qword_4FAE468 = (__int64)"Enable partial-overwrite tracking in DSE";
  sub_16B88A0(&qword_4FAE440);
  __cxa_atexit(sub_12EDEC0, &qword_4FAE440, &qword_4A427C0);
  qword_4FAE360 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE410 = 256;
  word_4FAE36C &= 0xF000u;
  qword_4FAE370 = 0;
  qword_4FAE378 = 0;
  qword_4FAE380 = 0;
  dword_4FAE368 = v1;
  qword_4FAE408 = (__int64)&unk_49E74E8;
  qword_4FAE3A8 = (__int64)qword_4FA01C0;
  qword_4FAE3B8 = (__int64)&unk_4FAE3D8;
  qword_4FAE3C0 = (__int64)&unk_4FAE3D8;
  qword_4FAE360 = (__int64)&unk_49EEC70;
  qword_4FAE418 = (__int64)&unk_49EEDB0;
  qword_4FAE388 = 0;
  qword_4FAE390 = 0;
  qword_4FAE398 = 0;
  qword_4FAE3A0 = 0;
  qword_4FAE3B0 = 0;
  qword_4FAE3C8 = 4;
  dword_4FAE3D0 = 0;
  byte_4FAE3F8 = 0;
  byte_4FAE400 = 0;
  sub_16B8280(&qword_4FAE360, "enable-dse-partial-store-merging", 32);
  word_4FAE410 = 257;
  byte_4FAE400 = 1;
  qword_4FAE390 = 35;
  LOBYTE(word_4FAE36C) = word_4FAE36C & 0x9F | 0x20;
  qword_4FAE388 = (__int64)"Enable partial store merging in DSE";
  sub_16B88A0(&qword_4FAE360);
  return __cxa_atexit(sub_12EDEC0, &qword_4FAE360, &qword_4A427C0);
}
