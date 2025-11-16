// Function: ctor_143
// Address: 0x4b3ff0
//
int ctor_143()
{
  int v0; // edx

  qword_4F9DFA0[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F9DFA0[1]) &= 0xF000u;
  qword_4F9DFA0[9] = &unk_4FA01C0;
  qword_4F9DFA0[11] = &qword_4F9DFA0[15];
  qword_4F9DFA0[12] = &qword_4F9DFA0[15];
  LODWORD(qword_4F9DFA0[1]) = v0;
  qword_4F9DFA0[2] = 0;
  qword_4F9DFA0[21] = &unk_49E74E8;
  LOWORD(qword_4F9DFA0[22]) = 256;
  qword_4F9DFA0[3] = 0;
  qword_4F9DFA0[4] = 0;
  qword_4F9DFA0[0] = &unk_49EEC70;
  qword_4F9DFA0[5] = 0;
  qword_4F9DFA0[6] = 0;
  qword_4F9DFA0[23] = &unk_49EEDB0;
  qword_4F9DFA0[7] = 0;
  qword_4F9DFA0[8] = 0;
  qword_4F9DFA0[10] = 0;
  qword_4F9DFA0[13] = 4;
  LODWORD(qword_4F9DFA0[14]) = 0;
  LOBYTE(qword_4F9DFA0[19]) = 0;
  LOBYTE(qword_4F9DFA0[20]) = 0;
  sub_16B8280(qword_4F9DFA0, "use-dbg-addr", 12);
  qword_4F9DFA0[5] = "Use llvm.dbg.addr for all local variables";
  LOWORD(qword_4F9DFA0[22]) = 256;
  qword_4F9DFA0[6] = 41;
  LOBYTE(qword_4F9DFA0[20]) = 0;
  BYTE4(qword_4F9DFA0[1]) = BYTE4(qword_4F9DFA0[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4F9DFA0);
  return __cxa_atexit(sub_12EDEC0, qword_4F9DFA0, &qword_4A427C0);
}
