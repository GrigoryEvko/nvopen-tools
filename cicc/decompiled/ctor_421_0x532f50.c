// Function: ctor_421
// Address: 0x532f50
//
int ctor_421()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FF1500 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF157C = 1;
  qword_4FF1550 = 0x100000000LL;
  dword_4FF150C &= 0x8000u;
  qword_4FF1518 = 0;
  qword_4FF1520 = 0;
  qword_4FF1528 = 0;
  dword_4FF1508 = v0;
  word_4FF1510 = 0;
  qword_4FF1530 = 0;
  qword_4FF1538 = 0;
  qword_4FF1540 = 0;
  qword_4FF1548 = (__int64)&unk_4FF1558;
  qword_4FF1560 = 0;
  qword_4FF1568 = (__int64)&unk_4FF1580;
  qword_4FF1570 = 1;
  dword_4FF1578 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF1550;
  v3 = (unsigned int)qword_4FF1550 + 1LL;
  if ( v3 > HIDWORD(qword_4FF1550) )
  {
    sub_C8D5F0((char *)&unk_4FF1558 - 16, &unk_4FF1558, v3, 8);
    v2 = (unsigned int)qword_4FF1550;
  }
  *(_QWORD *)(qword_4FF1548 + 8 * v2) = v1;
  LODWORD(qword_4FF1550) = qword_4FF1550 + 1;
  qword_4FF1588 = 0;
  qword_4FF1590 = (__int64)&unk_49D9748;
  qword_4FF1598 = 0;
  qword_4FF1500 = (__int64)&unk_49DC090;
  qword_4FF15A0 = (__int64)&unk_49DC1D0;
  qword_4FF15C0 = (__int64)nullsub_23;
  qword_4FF15B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF1500, "enable-vfe", 10);
  LOBYTE(qword_4FF1588) = 1;
  qword_4FF1530 = 35;
  LOBYTE(dword_4FF150C) = dword_4FF150C & 0x9F | 0x20;
  LOWORD(qword_4FF1598) = 257;
  qword_4FF1528 = (__int64)"Enable virtual function elimination";
  sub_C53130(&qword_4FF1500);
  return __cxa_atexit(sub_984900, &qword_4FF1500, &qword_4A427C0);
}
