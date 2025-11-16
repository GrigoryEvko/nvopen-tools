// Function: ctor_191
// Address: 0x4decd0
//
int ctor_191()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v4; // [rsp+24h] [rbp-6Ch] BYREF
  int *v5; // [rsp+28h] [rbp-68h] BYREF
  const char *v6; // [rsp+30h] [rbp-60h]
  __int64 v7; // [rsp+38h] [rbp-58h]
  char *v8; // [rsp+40h] [rbp-50h] BYREF
  __int64 v9; // [rsp+48h] [rbp-48h]
  char v10; // [rsp+50h] [rbp-40h] BYREF

  qword_4FAD5A0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAD5AC &= 0xF000u;
  qword_4FAD5E8 = (__int64)qword_4FA01C0;
  qword_4FAD5B0 = 0;
  qword_4FAD5B8 = 0;
  qword_4FAD5C0 = 0;
  dword_4FAD5A8 = v0;
  qword_4FAD5F8 = (__int64)&unk_4FAD618;
  qword_4FAD600 = (__int64)&unk_4FAD618;
  qword_4FAD640 = (__int64)&byte_4FAD650;
  qword_4FAD668 = (__int64)&byte_4FAD678;
  qword_4FAD5C8 = 0;
  qword_4FAD5D0 = 0;
  qword_4FAD660 = (__int64)&unk_49EED10;
  qword_4FAD5D8 = 0;
  qword_4FAD5E0 = 0;
  qword_4FAD5A0 = (__int64)&unk_49EEBF0;
  qword_4FAD5F0 = 0;
  byte_4FAD638 = 0;
  qword_4FAD690 = (__int64)&unk_49EEE90;
  qword_4FAD698 = (__int64)&byte_4FAD6A8;
  qword_4FAD608 = 4;
  dword_4FAD610 = 0;
  qword_4FAD648 = 0;
  byte_4FAD650 = 0;
  qword_4FAD670 = 0;
  byte_4FAD678 = 0;
  byte_4FAD688 = 0;
  qword_4FAD6A0 = 0;
  byte_4FAD6A8 = 0;
  sub_16B8280(&qword_4FAD5A0, "sample-profile-file", 19);
  v8 = &v10;
  sub_18A3750(&v8, byte_3F871B3);
  sub_2240AE0(&qword_4FAD640, &v8);
  byte_4FAD688 = 1;
  sub_2240AE0(&qword_4FAD668, &v8);
  sub_2240A30(&v8);
  qword_4FAD5E0 = 8;
  qword_4FAD5D8 = (__int64)"filename";
  qword_4FAD5C8 = (__int64)"Profile file loaded by -sample-profile";
  qword_4FAD5D0 = 38;
  LOBYTE(word_4FAD5AC) = word_4FAD5AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAD5A0);
  __cxa_atexit(sub_12F0C20, &qword_4FAD5A0, &qword_4A427C0);
  qword_4FAD4C0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAD4CC &= 0xF000u;
  qword_4FAD4D0 = 0;
  qword_4FAD4D8 = 0;
  qword_4FAD4E0 = 0;
  qword_4FAD4E8 = 0;
  qword_4FAD4F0 = 0;
  dword_4FAD4C8 = v1;
  qword_4FAD518 = (__int64)&unk_4FAD538;
  qword_4FAD520 = (__int64)&unk_4FAD538;
  qword_4FAD508 = (__int64)qword_4FA01C0;
  qword_4FAD4F8 = 0;
  qword_4FAD568 = (__int64)&unk_49E74A8;
  qword_4FAD500 = 0;
  qword_4FAD510 = 0;
  qword_4FAD4C0 = (__int64)&unk_49EEAF0;
  qword_4FAD528 = 4;
  dword_4FAD530 = 0;
  qword_4FAD578 = (__int64)&unk_49EEE10;
  byte_4FAD558 = 0;
  dword_4FAD560 = 0;
  byte_4FAD574 = 1;
  dword_4FAD570 = 0;
  sub_16B8280(&qword_4FAD4C0, "sample-profile-max-propagate-iterations", 39);
  dword_4FAD560 = 100;
  qword_4FAD4E8 = (__int64)"Maximum number of iterations to go through when propagating sample block/edge weights through the CFG.";
  byte_4FAD574 = 1;
  dword_4FAD570 = 100;
  qword_4FAD4F0 = 102;
  sub_16B88A0(&qword_4FAD4C0);
  __cxa_atexit(sub_12EDE60, &qword_4FAD4C0, &qword_4A427C0);
  v5 = &v4;
  v8 = "N";
  v6 = "Emit a warning if less than N% of records in the input profile are matched to the IR.";
  v7 = 85;
  v9 = 1;
  v4 = 0;
  sub_18A7A50(&unk_4FAD3E0, "sample-profile-check-record-coverage", &v5, &v8);
  __cxa_atexit(sub_12EDE60, &unk_4FAD3E0, &qword_4A427C0);
  v7 = 85;
  v6 = "Emit a warning if less than N% of samples in the input profile are matched to the IR.";
  v8 = "N";
  v5 = &v4;
  v9 = 1;
  v4 = 0;
  sub_18A7A50(&unk_4FAD300, "sample-profile-check-sample-coverage", &v5, &v8);
  __cxa_atexit(sub_12EDE60, &unk_4FAD300, &qword_4A427C0);
  qword_4FAD220 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAD22C &= 0xF000u;
  qword_4FAD230 = 0;
  qword_4FAD238 = 0;
  qword_4FAD240 = 0;
  qword_4FAD248 = 0;
  qword_4FAD250 = 0;
  dword_4FAD228 = v2;
  qword_4FAD278 = (__int64)&unk_4FAD298;
  qword_4FAD280 = (__int64)&unk_4FAD298;
  qword_4FAD268 = (__int64)qword_4FA01C0;
  qword_4FAD258 = 0;
  qword_4FAD2C8 = (__int64)&unk_49E74E8;
  word_4FAD2D0 = 256;
  qword_4FAD260 = 0;
  qword_4FAD270 = 0;
  qword_4FAD220 = (__int64)&unk_49EEC70;
  qword_4FAD288 = 4;
  byte_4FAD2B8 = 0;
  qword_4FAD2D8 = (__int64)&unk_49EEDB0;
  dword_4FAD290 = 0;
  byte_4FAD2C0 = 0;
  sub_16B8280(&qword_4FAD220, "no-warn-sample-unused", 21);
  word_4FAD2D0 = 256;
  byte_4FAD2C0 = 0;
  qword_4FAD250 = 120;
  LOBYTE(word_4FAD22C) = word_4FAD22C & 0x9F | 0x20;
  qword_4FAD248 = (__int64)"Use this option to turn off/on warnings about function with samples but without debug informa"
                           "tion to use those samples. ";
  sub_16B88A0(&qword_4FAD220);
  return __cxa_atexit(sub_12EDEC0, &qword_4FAD220, &qword_4A427C0);
}
