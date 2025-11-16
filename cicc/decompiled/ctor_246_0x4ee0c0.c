// Function: ctor_246
// Address: 0x4ee0c0
//
int __fastcall ctor_246(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // edx
  __int64 v7; // r9
  __int64 v8; // r9
  __int64 v9; // r9
  __int64 v10; // r9
  __int64 v11; // r9
  __int64 v13; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v14; // [rsp+8h] [rbp-48h] BYREF
  const char *v15; // [rsp+10h] [rbp-40h] BYREF
  __int64 v16; // [rsp+18h] [rbp-38h]

  v15 = "Aggregates containing large number of elements will not be split";
  v14 = &v13;
  v16 = 64;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_1B7FEE0)(
    &unk_4FB7DE0,
    "max-aggr-elems",
    (char *)&v13 + 4,
    &v15,
    &v14,
    a6,
    0x100000032LL);
  __cxa_atexit(sub_12EDE60, &unk_4FB7DE0, &qword_4A427C0);
  qword_4FB7D00 = (__int64)&unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB7D0C &= 0xF000u;
  qword_4FB7D10 = 0;
  qword_4FB7D48 = (__int64)qword_4FA01C0;
  qword_4FB7D18 = 0;
  qword_4FB7D20 = 0;
  qword_4FB7D28 = 0;
  dword_4FB7D08 = v6;
  qword_4FB7D58 = (__int64)&unk_4FB7D78;
  qword_4FB7D60 = (__int64)&unk_4FB7D78;
  qword_4FB7D30 = 0;
  qword_4FB7D38 = 0;
  qword_4FB7DA8 = (__int64)&unk_49E74E8;
  word_4FB7DB0 = 256;
  qword_4FB7D40 = 0;
  qword_4FB7D50 = 0;
  qword_4FB7D00 = (__int64)&unk_49EEC70;
  qword_4FB7D68 = 4;
  byte_4FB7D98 = 0;
  qword_4FB7DB8 = (__int64)&unk_49EEDB0;
  dword_4FB7D70 = 0;
  byte_4FB7DA0 = 0;
  sub_16B8280(&qword_4FB7D00, "vect-split-aggr", 15);
  word_4FB7DB0 = 257;
  byte_4FB7DA0 = 1;
  qword_4FB7D30 = 48;
  LOBYTE(word_4FB7D0C) = word_4FB7D0C & 0x9F | 0x20;
  qword_4FB7D28 = (__int64)"Should aggregates be split before vectorization.";
  sub_16B88A0(&qword_4FB7D00);
  __cxa_atexit(sub_12EDEC0, &qword_4FB7D00, &qword_4A427C0);
  v14 = &v13;
  v15 = "Should longer sequences of small datatypes be considered for upsizing during vectorization.";
  LOBYTE(v13) = 0;
  v16 = 91;
  HIDWORD(v13) = 1;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_1B80060)(
    &unk_4FB7C20,
    "disable-ldst-upsizing",
    (char *)&v13 + 4,
    &v15,
    &v14,
    v7,
    v13);
  __cxa_atexit(sub_12EDEC0, &unk_4FB7C20, &qword_4A427C0);
  v14 = &v13;
  v15 = "Should Loads be introduced in gaps to enable vectorization.";
  LOBYTE(v13) = 1;
  v16 = 59;
  HIDWORD(v13) = 1;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_1B801E0)(
    &unk_4FB7B40,
    "vect-fill-gaps",
    (char *)&v13 + 4,
    &v15,
    &v14,
    v8,
    v13);
  __cxa_atexit(sub_12EDEC0, &unk_4FB7B40, &qword_4A427C0);
  v14 = &v13;
  v15 = "Chains containing large number of load/stores will not be vectorized, for compile time";
  v16 = 86;
  ((void (__fastcall *)(void *, char *, char *, const char **, __int64 **, __int64, __int64))sub_1B7FEE0)(
    &unk_4FB7A60,
    "max-chain-size",
    (char *)&v13 + 4,
    &v15,
    &v14,
    v9,
    0x1000007D0LL);
  __cxa_atexit(sub_12EDE60, &unk_4FB7A60, &qword_4A427C0);
  v14 = &v13;
  v15 = "Allow expensive analysis for aggressive load-store vectorization";
  LOBYTE(v13) = 0;
  v16 = 64;
  HIDWORD(v13) = 1;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_1B801E0)(
    &unk_4FB7980,
    "aggressive-lsv",
    (char *)&v13 + 4,
    &v15,
    &v14,
    v10,
    v13);
  __cxa_atexit(sub_12EDEC0, &unk_4FB7980, &qword_4A427C0);
  v14 = &v13;
  LOBYTE(v13) = 1;
  v15 = "Should sequences of smaller datatypes in an aggregate be merged to a wider datatype before vectorization.";
  v16 = 105;
  HIDWORD(v13) = 1;
  ((void (__fastcall *)(void *, const char *, char *, const char **, __int64 **, __int64, __int64))sub_1B80060)(
    &unk_4FB78A0,
    "vect-split-aggr-merge",
    (char *)&v13 + 4,
    &v15,
    &v14,
    v11,
    v13);
  return __cxa_atexit(sub_12EDEC0, &unk_4FB78A0, &qword_4A427C0);
}
