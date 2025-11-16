// Function: ctor_295
// Address: 0x4fcce0
//
int ctor_295()
{
  int v0; // edx

  qword_4FC4440[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FC4440[1]) &= 0xF000u;
  qword_4FC4440[9] = qword_4FA01C0;
  qword_4FC4440[11] = &qword_4FC4440[15];
  qword_4FC4440[12] = &qword_4FC4440[15];
  LODWORD(qword_4FC4440[1]) = v0;
  qword_4FC4440[2] = 0;
  qword_4FC4440[21] = &unk_49E74E8;
  LOWORD(qword_4FC4440[22]) = 256;
  qword_4FC4440[3] = 0;
  qword_4FC4440[4] = 0;
  qword_4FC4440[0] = &unk_49EEC70;
  qword_4FC4440[5] = 0;
  qword_4FC4440[6] = 0;
  qword_4FC4440[23] = &unk_49EEDB0;
  qword_4FC4440[7] = 0;
  qword_4FC4440[8] = 0;
  qword_4FC4440[10] = 0;
  qword_4FC4440[13] = 4;
  LODWORD(qword_4FC4440[14]) = 0;
  LOBYTE(qword_4FC4440[19]) = 0;
  LOBYTE(qword_4FC4440[20]) = 0;
  sub_16B8280(qword_4FC4440, "use-segment-set-for-physregs", 28);
  LOWORD(qword_4FC4440[22]) = 257;
  LOBYTE(qword_4FC4440[20]) = 1;
  qword_4FC4440[6] = 67;
  BYTE4(qword_4FC4440[1]) = BYTE4(qword_4FC4440[1]) & 0x9F | 0x20;
  qword_4FC4440[5] = "Use segment set for the computation of the live ranges of physregs.";
  sub_16B88A0(qword_4FC4440);
  return __cxa_atexit(sub_12EDEC0, qword_4FC4440, &qword_4A427C0);
}
