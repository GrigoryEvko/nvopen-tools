// Function: ctor_717
// Address: 0x5c0910
//
int ctor_717()
{
  int v0; // edx

  qword_50525C0[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_50525C0[1]) &= 0xF000u;
  qword_50525C0[9] = qword_4FA01C0;
  qword_50525C0[11] = &qword_50525C0[15];
  qword_50525C0[12] = &qword_50525C0[15];
  LODWORD(qword_50525C0[1]) = v0;
  qword_50525C0[2] = 0;
  qword_50525C0[21] = &unk_49E74E8;
  LOWORD(qword_50525C0[22]) = 256;
  qword_50525C0[3] = 0;
  qword_50525C0[4] = 0;
  qword_50525C0[0] = &unk_49EEC70;
  qword_50525C0[5] = 0;
  qword_50525C0[6] = 0;
  qword_50525C0[23] = &unk_49EEDB0;
  qword_50525C0[7] = 0;
  qword_50525C0[8] = 0;
  qword_50525C0[10] = 0;
  qword_50525C0[13] = 4;
  LODWORD(qword_50525C0[14]) = 0;
  LOBYTE(qword_50525C0[19]) = 0;
  LOBYTE(qword_50525C0[20]) = 0;
  sub_16B8280(qword_50525C0, "enable-mssa-loop-dependency", 27);
  LOWORD(qword_50525C0[22]) = 256;
  LOBYTE(qword_50525C0[20]) = 0;
  qword_50525C0[6] = 49;
  BYTE4(qword_50525C0[1]) = BYTE4(qword_50525C0[1]) & 0x9F | 0x20;
  qword_50525C0[5] = "Enable MemorySSA dependency for loop pass manager";
  sub_16B88A0(qword_50525C0);
  return __cxa_atexit(sub_12EDEC0, qword_50525C0, &qword_4A427C0);
}
