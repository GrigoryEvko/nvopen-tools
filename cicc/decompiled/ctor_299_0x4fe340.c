// Function: ctor_299
// Address: 0x4fe340
//
int ctor_299()
{
  int v0; // eax
  int v1; // eax

  qword_4FC5920[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FC5920[1]) &= 0xF000u;
  LODWORD(qword_4FC5920[1]) = v0;
  qword_4FC5920[2] = 0;
  qword_4FC5920[9] = qword_4FA01C0;
  qword_4FC5920[11] = &qword_4FC5920[15];
  qword_4FC5920[12] = &qword_4FC5920[15];
  qword_4FC5920[3] = 0;
  qword_4FC5920[4] = 0;
  qword_4FC5920[21] = &unk_49E74A8;
  qword_4FC5920[5] = 0;
  qword_4FC5920[0] = &unk_49EEAF0;
  qword_4FC5920[6] = 0;
  qword_4FC5920[23] = &unk_49EEE10;
  qword_4FC5920[7] = 0;
  qword_4FC5920[8] = 0;
  qword_4FC5920[10] = 0;
  qword_4FC5920[13] = 4;
  LODWORD(qword_4FC5920[14]) = 0;
  LOBYTE(qword_4FC5920[19]) = 0;
  LODWORD(qword_4FC5920[20]) = 0;
  BYTE4(qword_4FC5920[22]) = 1;
  LODWORD(qword_4FC5920[22]) = 0;
  sub_16B8280(qword_4FC5920, "static-likely-prob", 18);
  qword_4FC5920[6] = 70;
  qword_4FC5920[5] = "branch probability threshold in percentageto be considered very likely";
  LODWORD(qword_4FC5920[20]) = 80;
  BYTE4(qword_4FC5920[22]) = 1;
  LODWORD(qword_4FC5920[22]) = 80;
  BYTE4(qword_4FC5920[1]) = BYTE4(qword_4FC5920[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4FC5920);
  __cxa_atexit(sub_12EDE60, qword_4FC5920, &qword_4A427C0);
  qword_4FC5840[0] = &unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FC5840[1]) &= 0xF000u;
  LODWORD(qword_4FC5840[1]) = v1;
  qword_4FC5840[21] = &unk_49E74A8;
  qword_4FC5840[9] = qword_4FA01C0;
  qword_4FC5840[11] = &qword_4FC5840[15];
  qword_4FC5840[12] = &qword_4FC5840[15];
  qword_4FC5840[0] = &unk_49EEAF0;
  qword_4FC5840[23] = &unk_49EEE10;
  qword_4FC5840[2] = 0;
  qword_4FC5840[3] = 0;
  qword_4FC5840[4] = 0;
  qword_4FC5840[5] = 0;
  qword_4FC5840[6] = 0;
  qword_4FC5840[7] = 0;
  qword_4FC5840[8] = 0;
  qword_4FC5840[10] = 0;
  qword_4FC5840[13] = 4;
  LODWORD(qword_4FC5840[14]) = 0;
  LOBYTE(qword_4FC5840[19]) = 0;
  LODWORD(qword_4FC5840[20]) = 0;
  BYTE4(qword_4FC5840[22]) = 1;
  LODWORD(qword_4FC5840[22]) = 0;
  sub_16B8280(qword_4FC5840, "profile-likely-prob", 19);
  qword_4FC5840[6] = 97;
  qword_4FC5840[5] = "branch probability threshold in percentage to be considered very likely when profile is available";
  LODWORD(qword_4FC5840[20]) = 51;
  BYTE4(qword_4FC5840[22]) = 1;
  LODWORD(qword_4FC5840[22]) = 51;
  BYTE4(qword_4FC5840[1]) = BYTE4(qword_4FC5840[1]) & 0x9F | 0x20;
  sub_16B88A0(qword_4FC5840);
  return __cxa_atexit(sub_12EDE60, qword_4FC5840, &qword_4A427C0);
}
