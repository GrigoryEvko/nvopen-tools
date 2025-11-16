// Function: ctor_200
// Address: 0x4e07e0
//
int ctor_200()
{
  int v0; // edx

  qword_4FAE540 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAE54C &= 0xF000u;
  qword_4FAE550 = 0;
  qword_4FAE588 = (__int64)qword_4FA01C0;
  qword_4FAE558 = 0;
  qword_4FAE560 = 0;
  qword_4FAE568 = 0;
  dword_4FAE548 = v0;
  qword_4FAE598 = (__int64)&unk_4FAE5B8;
  qword_4FAE5A0 = (__int64)&unk_4FAE5B8;
  qword_4FAE570 = 0;
  qword_4FAE578 = 0;
  qword_4FAE5E8 = (__int64)&unk_49E74A8;
  qword_4FAE580 = 0;
  qword_4FAE590 = 0;
  qword_4FAE540 = (__int64)&unk_49EEAF0;
  qword_4FAE5A8 = 4;
  dword_4FAE5B0 = 0;
  qword_4FAE5F8 = (__int64)&unk_49EEE10;
  byte_4FAE5D8 = 0;
  dword_4FAE5E0 = 0;
  byte_4FAE5F4 = 1;
  dword_4FAE5F0 = 0;
  sub_16B8280(&qword_4FAE540, "float2int-max-integer-bw", 24);
  dword_4FAE5E0 = 64;
  byte_4FAE5F4 = 1;
  dword_4FAE5F0 = 64;
  qword_4FAE570 = 57;
  LOBYTE(word_4FAE54C) = word_4FAE54C & 0x9F | 0x20;
  qword_4FAE568 = (__int64)"Max integer bitwidth to consider in float2int(default=64)";
  sub_16B88A0(&qword_4FAE540);
  return __cxa_atexit(sub_12EDE60, &qword_4FAE540, &qword_4A427C0);
}
