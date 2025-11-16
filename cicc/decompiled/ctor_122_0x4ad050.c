// Function: ctor_122
// Address: 0x4ad050
//
int ctor_122()
{
  int v0; // edx

  qword_4F99140[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F99140[1]) &= 0xF000u;
  qword_4F99140[9] = &unk_4FA01C0;
  qword_4F99140[11] = &qword_4F99140[15];
  qword_4F99140[12] = &qword_4F99140[15];
  LODWORD(qword_4F99140[1]) = v0;
  qword_4F99140[2] = 0;
  qword_4F99140[21] = &unk_49E74A8;
  qword_4F99140[3] = 0;
  qword_4F99140[4] = 0;
  qword_4F99140[0] = &unk_49EEAF0;
  qword_4F99140[5] = 0;
  qword_4F99140[6] = 0;
  qword_4F99140[23] = &unk_49EEE10;
  qword_4F99140[7] = 0;
  qword_4F99140[8] = 0;
  qword_4F99140[10] = 0;
  qword_4F99140[13] = 4;
  LODWORD(qword_4F99140[14]) = 0;
  LOBYTE(qword_4F99140[19]) = 0;
  LODWORD(qword_4F99140[20]) = 0;
  BYTE4(qword_4F99140[22]) = 1;
  LODWORD(qword_4F99140[22]) = 0;
  sub_16B8280(qword_4F99140, "available-load-scan-limit", 25);
  LODWORD(qword_4F99140[20]) = 6;
  BYTE4(qword_4F99140[22]) = 1;
  LODWORD(qword_4F99140[22]) = 6;
  qword_4F99140[6] = 147;
  BYTE4(qword_4F99140[1]) = BYTE4(qword_4F99140[1]) & 0x9F | 0x20;
  qword_4F99140[5] = "Use this to specify the default maximum number of instructions to scan backward from a given instru"
                     "ction, when searching for available loaded value";
  sub_16B88A0(qword_4F99140);
  return __cxa_atexit(sub_12EDE60, qword_4F99140, &qword_4A427C0);
}
