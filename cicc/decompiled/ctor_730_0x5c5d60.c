// Function: ctor_730
// Address: 0x5c5d60
//
int ctor_730()
{
  int v0; // edx

  qword_5057460[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_5057460[1]) &= 0xF000u;
  qword_5057460[9] = qword_4FA01C0;
  qword_5057460[11] = &qword_5057460[15];
  qword_5057460[12] = &qword_5057460[15];
  LODWORD(qword_5057460[1]) = v0;
  qword_5057460[2] = 0;
  qword_5057460[21] = &unk_49E74A8;
  qword_5057460[3] = 0;
  qword_5057460[4] = 0;
  qword_5057460[0] = &unk_49EEAF0;
  qword_5057460[5] = 0;
  qword_5057460[6] = 0;
  qword_5057460[23] = &unk_49EEE10;
  qword_5057460[7] = 0;
  qword_5057460[8] = 0;
  qword_5057460[10] = 0;
  qword_5057460[13] = 4;
  LODWORD(qword_5057460[14]) = 0;
  LOBYTE(qword_5057460[19]) = 0;
  LODWORD(qword_5057460[20]) = 0;
  BYTE4(qword_5057460[22]) = 1;
  LODWORD(qword_5057460[22]) = 0;
  sub_16B8280(qword_5057460, "partial-unrolling-threshold", 27);
  LODWORD(qword_5057460[20]) = 0;
  qword_5057460[5] = "Threshold for partial unrolling";
  BYTE4(qword_5057460[22]) = 1;
  LODWORD(qword_5057460[22]) = 0;
  qword_5057460[6] = 31;
  BYTE4(qword_5057460[1]) = BYTE4(qword_5057460[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_5057460);
  return __cxa_atexit(sub_12EDE60, qword_5057460, &qword_4A427C0);
}
