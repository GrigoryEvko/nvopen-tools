// Function: ctor_357
// Address: 0x50e730
//
int ctor_357()
{
  int v0; // edx

  qword_4FD1560 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD156C &= 0xF000u;
  qword_4FD1570 = 0;
  qword_4FD15A8 = (__int64)qword_4FA01C0;
  qword_4FD1578 = 0;
  qword_4FD1580 = 0;
  qword_4FD1588 = 0;
  dword_4FD1568 = v0;
  qword_4FD15B8 = (__int64)&unk_4FD15D8;
  qword_4FD15C0 = (__int64)&unk_4FD15D8;
  qword_4FD1590 = 0;
  qword_4FD1598 = 0;
  qword_4FD1608 = (__int64)&unk_49E74E8;
  word_4FD1610 = 256;
  qword_4FD15A0 = 0;
  qword_4FD15B0 = 0;
  qword_4FD1560 = (__int64)&unk_49EEC70;
  qword_4FD15C8 = 4;
  byte_4FD15F8 = 0;
  qword_4FD1618 = (__int64)&unk_49EEDB0;
  dword_4FD15D0 = 0;
  byte_4FD1600 = 0;
  sub_16B8280(&qword_4FD1560, "nvptx-no-f16-math", 17);
  word_4FD1610 = 256;
  byte_4FD1600 = 0;
  qword_4FD1590 = 51;
  LOBYTE(word_4FD156C) = word_4FD156C & 0x98 | 0x21;
  qword_4FD1588 = (__int64)"NVPTX Specific: Disable generation of f16 math ops.";
  sub_16B88A0(&qword_4FD1560);
  return __cxa_atexit(sub_12EDEC0, &qword_4FD1560, &qword_4A427C0);
}
