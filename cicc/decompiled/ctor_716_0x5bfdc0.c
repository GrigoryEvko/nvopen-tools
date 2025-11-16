// Function: ctor_716
// Address: 0x5bfdc0
//
int ctor_716()
{
  int v0; // eax
  __int64 v1; // rax
  int v2; // eax
  __int64 v3; // rax
  int v4; // eax
  __int64 v5; // rax
  int v6; // eax
  int v7; // eax
  int v8; // eax
  int v9; // eax
  _QWORD v11[2]; // [rsp+10h] [rbp-50h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  qword_50524E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_50524EC &= 0xF000u;
  qword_50524F0 = 0;
  qword_50524F8 = 0;
  qword_5052500 = 0;
  qword_5052508 = 0;
  dword_50524E8 = v0;
  qword_5052538 = (__int64)&unk_5052558;
  qword_5052540 = (__int64)&unk_5052558;
  qword_5052588 = (__int64)&unk_49E74A8;
  qword_5052510 = 0;
  qword_5052528 = (__int64)qword_4FA01C0;
  qword_50524E0 = (__int64)&unk_49FFFF8;
  qword_5052518 = 0;
  qword_5052520 = 0;
  qword_5052598 = (__int64)&unk_49EEE10;
  qword_5052530 = 0;
  qword_5052548 = 4;
  dword_5052550 = 0;
  byte_5052578 = 0;
  qword_5052580 = 0;
  byte_5052594 = 0;
  sub_16B8280(&qword_50524E0, "force-vector-width", 18);
  qword_5052510 = 40;
  LOBYTE(word_50524EC) = word_50524EC & 0x9F | 0x20;
  qword_5052508 = (__int64)"Sets the SIMD width. Zero is autoselect.";
  if ( qword_5052580 )
  {
    v1 = sub_16E8CB0();
    v13 = 1;
    v11[0] = "cl::location(x) specified more than once!";
    v12 = 3;
    sub_16B1F90(&qword_50524E0, v11, 0, 0, v1);
  }
  else
  {
    byte_5052594 = 1;
    qword_5052580 = (__int64)dword_50524C8;
    dword_5052590 = dword_50524C8[0];
  }
  sub_16B88A0(&qword_50524E0);
  __cxa_atexit(sub_2044000, &qword_50524E0, &qword_4A427C0);
  qword_5052400 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505240C &= 0xF000u;
  qword_5052410 = 0;
  qword_5052418 = 0;
  qword_5052420 = 0;
  qword_5052428 = 0;
  dword_5052408 = v2;
  qword_5052458 = (__int64)&unk_5052478;
  qword_5052460 = (__int64)&unk_5052478;
  qword_50524A8 = (__int64)&unk_49E74A8;
  qword_5052430 = 0;
  qword_5052448 = (__int64)qword_4FA01C0;
  qword_5052400 = (__int64)&unk_49FFFF8;
  qword_5052438 = 0;
  qword_5052440 = 0;
  qword_50524B8 = (__int64)&unk_49EEE10;
  qword_5052450 = 0;
  qword_5052468 = 4;
  dword_5052470 = 0;
  byte_5052498 = 0;
  qword_50524A0 = 0;
  byte_50524B4 = 0;
  sub_16B8280(&qword_5052400, "force-vector-interleave", 23);
  qword_5052430 = 60;
  LOBYTE(word_505240C) = word_505240C & 0x9F | 0x20;
  qword_5052428 = (__int64)"Sets the vectorization interleave count. Zero is autoselect.";
  if ( qword_50524A0 )
  {
    v3 = sub_16E8CB0();
    v13 = 1;
    v11[0] = "cl::location(x) specified more than once!";
    v12 = 3;
    sub_16B1F90(&qword_5052400, v11, 0, 0, v3);
  }
  else
  {
    byte_50524B4 = 1;
    qword_50524A0 = (__int64)dword_50523E8;
    dword_50524B0 = dword_50523E8[0];
  }
  sub_16B88A0(&qword_5052400);
  __cxa_atexit(sub_2044000, &qword_5052400, &qword_4A427C0);
  qword_5052320 = (__int64)&unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505232C &= 0xF000u;
  qword_5052330 = 0;
  qword_5052338 = 0;
  qword_5052340 = 0;
  qword_5052348 = 0;
  dword_5052328 = v4;
  qword_5052378 = (__int64)&unk_5052398;
  qword_5052380 = (__int64)&unk_5052398;
  qword_50523C8 = (__int64)&unk_49E74A8;
  qword_5052350 = 0;
  qword_5052368 = (__int64)qword_4FA01C0;
  qword_5052320 = (__int64)&unk_49FFFF8;
  qword_5052358 = 0;
  qword_5052360 = 0;
  qword_50523D8 = (__int64)&unk_49EEE10;
  qword_5052370 = 0;
  qword_5052388 = 4;
  dword_5052390 = 0;
  byte_50523B8 = 0;
  qword_50523C0 = 0;
  byte_50523D4 = 0;
  sub_16B8280(&qword_5052320, "runtime-memory-check-threshold", 30);
  qword_5052350 = 123;
  LOBYTE(word_505232C) = word_505232C & 0x9F | 0x20;
  qword_5052348 = (__int64)"When performing memory disambiguation checks at runtime do not generate more than this number"
                           " of comparisons (default = 8).";
  if ( qword_50523C0 )
  {
    v5 = sub_16E8CB0();
    v13 = 1;
    v11[0] = "cl::location(x) specified more than once!";
    v12 = 3;
    sub_16B1F90(&qword_5052320, v11, 0, 0, v5);
  }
  else
  {
    qword_50523C0 = (__int64)&unk_5052308;
  }
  *(_DWORD *)qword_50523C0 = 8;
  byte_50523D4 = 1;
  dword_50523D0 = 8;
  sub_16B88A0(&qword_5052320);
  __cxa_atexit(sub_2044000, &qword_5052320, &qword_4A427C0);
  qword_5052240 = (__int64)&unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505224C &= 0xF000u;
  qword_5052250 = 0;
  qword_5052258 = 0;
  qword_5052260 = 0;
  qword_5052268 = 0;
  dword_5052248 = v6;
  qword_5052298 = (__int64)&unk_50522B8;
  qword_50522A0 = (__int64)&unk_50522B8;
  qword_50522E8 = (__int64)&unk_49E74A8;
  qword_5052240 = (__int64)&unk_49EEAF0;
  qword_5052288 = (__int64)qword_4FA01C0;
  qword_50522F8 = (__int64)&unk_49EEE10;
  qword_5052270 = 0;
  qword_5052278 = 0;
  qword_5052280 = 0;
  qword_5052290 = 0;
  qword_50522A8 = 4;
  dword_50522B0 = 0;
  byte_50522D8 = 0;
  dword_50522E0 = 0;
  byte_50522F4 = 1;
  dword_50522F0 = 0;
  sub_16B8280(&qword_5052240, "memory-check-merge-threshold", 28);
  qword_5052270 = 94;
  dword_50522E0 = 100;
  byte_50522F4 = 1;
  dword_50522F0 = 100;
  LOBYTE(word_505224C) = word_505224C & 0x9F | 0x20;
  qword_5052268 = (__int64)"Maximum number of comparisons done when trying to merge runtime memory checks. (default = 100)";
  sub_16B88A0(&qword_5052240);
  __cxa_atexit(sub_12EDE60, &qword_5052240, &qword_4A427C0);
  qword_5052160 = (__int64)&unk_49EED30;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505216C &= 0xF000u;
  qword_5052170 = 0;
  qword_5052178 = 0;
  qword_5052180 = 0;
  qword_5052208 = (__int64)&unk_49E74A8;
  qword_5052160 = (__int64)&unk_49EEAF0;
  dword_5052168 = v7;
  qword_50521A8 = (__int64)qword_4FA01C0;
  qword_50521B8 = (__int64)&unk_50521D8;
  qword_50521C0 = (__int64)&unk_50521D8;
  qword_5052218 = (__int64)&unk_49EEE10;
  qword_5052188 = 0;
  qword_5052190 = 0;
  qword_5052198 = 0;
  qword_50521A0 = 0;
  qword_50521B0 = 0;
  qword_50521C8 = 4;
  dword_50521D0 = 0;
  byte_50521F8 = 0;
  dword_5052200 = 0;
  byte_5052214 = 1;
  dword_5052210 = 0;
  sub_16B8280((char *)&unk_50521D8 - 120, "max-dependences", 15);
  qword_5052190 = 79;
  dword_5052200 = 100;
  byte_5052214 = 1;
  dword_5052210 = 100;
  LOBYTE(word_505216C) = word_505216C & 0x9F | 0x20;
  qword_5052188 = (__int64)"Maximum number of dependences collected by loop-access analysis (default = 100)";
  sub_16B88A0(&qword_5052160);
  __cxa_atexit(sub_12EDE60, &qword_5052160, &qword_4A427C0);
  qword_5052080 = (__int64)&unk_49EED30;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_505208C &= 0xF000u;
  qword_5052090 = 0;
  qword_5052098 = 0;
  qword_50520A0 = 0;
  qword_50520A8 = 0;
  dword_5052088 = v8;
  qword_50520D8 = (__int64)&unk_50520F8;
  qword_50520E0 = (__int64)&unk_50520F8;
  qword_50520C8 = (__int64)qword_4FA01C0;
  qword_50520B0 = 0;
  word_5052130 = 256;
  qword_5052128 = (__int64)&unk_49E74E8;
  qword_5052080 = (__int64)&unk_49EEC70;
  qword_5052138 = (__int64)&unk_49EEDB0;
  qword_50520B8 = 0;
  qword_50520C0 = 0;
  qword_50520D0 = 0;
  qword_50520E8 = 4;
  dword_50520F0 = 0;
  byte_5052118 = 0;
  byte_5052120 = 0;
  sub_16B8280(&qword_5052080, "enable-mem-access-versioning", 28);
  word_5052130 = 257;
  byte_5052120 = 1;
  qword_50520B0 = 47;
  LOBYTE(word_505208C) = word_505208C & 0x9F | 0x20;
  qword_50520A8 = (__int64)"Enable symbolic stride memory access versioning";
  sub_16B88A0(&qword_5052080);
  __cxa_atexit(sub_12EDEC0, &qword_5052080, &qword_4A427C0);
  qword_5051FA0 = (__int64)&unk_49EED30;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_5052050 = 256;
  qword_5051FB0 = 0;
  word_5051FAC &= 0xF000u;
  qword_5052048 = (__int64)&unk_49E74E8;
  qword_5051FA0 = (__int64)&unk_49EEC70;
  dword_5051FA8 = v9;
  qword_5051FE8 = (__int64)qword_4FA01C0;
  qword_5051FF8 = (__int64)&unk_5052018;
  qword_5052000 = (__int64)&unk_5052018;
  qword_5052058 = (__int64)&unk_49EEDB0;
  qword_5051FB8 = 0;
  qword_5051FC0 = 0;
  qword_5051FC8 = 0;
  qword_5051FD0 = 0;
  qword_5051FD8 = 0;
  qword_5051FE0 = 0;
  qword_5051FF0 = 0;
  qword_5052008 = 4;
  dword_5052010 = 0;
  byte_5052038 = 0;
  byte_5052040 = 0;
  sub_16B8280((char *)&unk_5052018 - 120, "store-to-load-forwarding-conflict-detection", 43);
  word_5052050 = 257;
  byte_5052040 = 1;
  qword_5051FD0 = 49;
  LOBYTE(word_5051FAC) = word_5051FAC & 0x9F | 0x20;
  qword_5051FC8 = (__int64)"Enable conflict detection in loop-access analysis";
  sub_16B88A0(&qword_5051FA0);
  return __cxa_atexit(sub_12EDEC0, &qword_5051FA0, &qword_4A427C0);
}
