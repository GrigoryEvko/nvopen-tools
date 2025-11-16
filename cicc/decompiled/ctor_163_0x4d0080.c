// Function: ctor_163
// Address: 0x4d0080
//
int ctor_163()
{
  int v0; // eax
  __int64 v1; // r13
  int v2; // eax
  __int64 v3; // rax
  const char *v5; // [rsp+0h] [rbp-40h] BYREF
  char v6; // [rsp+10h] [rbp-30h]
  char v7; // [rsp+11h] [rbp-2Fh]

  qword_4FA1540 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA154C &= 0xF000u;
  qword_4FA1550 = 0;
  qword_4FA1558 = 0;
  qword_4FA1560 = 0;
  qword_4FA1568 = 0;
  dword_4FA1548 = v0;
  qword_4FA1598 = (__int64)&unk_4FA15B8;
  qword_4FA15A0 = (__int64)&unk_4FA15B8;
  qword_4FA1570 = 0;
  qword_4FA1588 = (__int64)qword_4FA01C0;
  qword_4FA15E8 = (__int64)&unk_49E74E8;
  word_4FA15F0 = 256;
  qword_4FA1578 = 0;
  qword_4FA1580 = 0;
  qword_4FA1540 = (__int64)&unk_49EEC70;
  qword_4FA1590 = 0;
  byte_4FA15D8 = 0;
  qword_4FA15F8 = (__int64)&unk_49EEDB0;
  qword_4FA15A8 = 4;
  dword_4FA15B0 = 0;
  byte_4FA15E0 = 0;
  sub_16B8280(&qword_4FA1540, "track-memory", 12);
  qword_4FA1570 = 54;
  qword_4FA1568 = (__int64)"Enable -time-passes memory tracking (this may be slow)";
  LOBYTE(word_4FA154C) = word_4FA154C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA1540);
  __cxa_atexit(sub_12EDEC0, &qword_4FA1540, &qword_4A427C0);
  if ( !qword_4FA1630 )
    sub_16C1EA0(&qword_4FA1630, sub_16D5DD0, sub_16D6250);
  v1 = qword_4FA1630;
  qword_4FA1440 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA144C &= 0xF000u;
  qword_4FA1450 = 0;
  qword_4FA1458 = 0;
  qword_4FA1460 = 0;
  qword_4FA1468 = 0;
  qword_4FA1470 = 0;
  dword_4FA1448 = v2;
  qword_4FA1498 = (__int64)&unk_4FA14B8;
  qword_4FA14A0 = (__int64)&unk_4FA14B8;
  qword_4FA14F0 = (__int64)&byte_4FA1500;
  qword_4FA1478 = 0;
  qword_4FA1480 = 0;
  qword_4FA14E8 = (__int64)&unk_49EED10;
  qword_4FA1488 = (__int64)qword_4FA01C0;
  qword_4FA1490 = 0;
  qword_4FA1440 = (__int64)&unk_49EF608;
  qword_4FA14A8 = 4;
  dword_4FA14B0 = 0;
  qword_4FA1518 = (__int64)&unk_49EEE90;
  qword_4FA1520 = (__int64)&byte_4FA1530;
  byte_4FA14D8 = 0;
  qword_4FA14E0 = 0;
  qword_4FA14F8 = 0;
  byte_4FA1500 = 0;
  byte_4FA1510 = 0;
  qword_4FA1528 = 0;
  byte_4FA1530 = 0;
  sub_16B8280(&byte_4FA1530 - 240, "info-output-file", 16);
  qword_4FA1480 = 8;
  qword_4FA1478 = (__int64)"filename";
  qword_4FA1468 = (__int64)"File to append -stats and -timer output to";
  qword_4FA1470 = 42;
  LOBYTE(word_4FA144C) = word_4FA144C & 0x9F | 0x20;
  if ( qword_4FA14E0 )
  {
    v3 = sub_16E8CB0();
    v7 = 1;
    v5 = "cl::location(x) specified more than once!";
    v6 = 3;
    sub_16B1F90(&qword_4FA1440, &v5, 0, 0, v3);
  }
  else
  {
    qword_4FA14E0 = v1;
    byte_4FA1510 = 1;
    sub_2240AE0(&qword_4FA14F0, v1);
  }
  sub_16B88A0(&qword_4FA1440);
  return __cxa_atexit(sub_16D6110, &qword_4FA1440, &qword_4A427C0);
}
