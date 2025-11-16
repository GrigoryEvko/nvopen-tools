// Function: ctor_275
// Address: 0x4f7750
//
int ctor_275()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FBF2C0[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FBF2C0[1]) &= 0xF000u;
  LODWORD(qword_4FBF2C0[1]) = v0;
  qword_4FBF2C0[11] = &qword_4FBF2C0[15];
  qword_4FBF2C0[12] = &qword_4FBF2C0[15];
  qword_4FBF2C0[9] = qword_4FA01C0;
  qword_4FBF2C0[2] = 0;
  qword_4FBF2C0[21] = &unk_49E74C8;
  qword_4FBF2C0[3] = 0;
  qword_4FBF2C0[0] = &unk_49EEB70;
  qword_4FBF2C0[23] = &unk_49EEDF0;
  qword_4FBF2C0[4] = 0;
  qword_4FBF2C0[5] = 0;
  qword_4FBF2C0[6] = 0;
  qword_4FBF2C0[7] = 0;
  qword_4FBF2C0[8] = 0;
  qword_4FBF2C0[10] = 0;
  qword_4FBF2C0[13] = 4;
  LODWORD(qword_4FBF2C0[14]) = 0;
  LOBYTE(qword_4FBF2C0[19]) = 0;
  LODWORD(qword_4FBF2C0[20]) = 0;
  BYTE4(qword_4FBF2C0[22]) = 1;
  LODWORD(qword_4FBF2C0[22]) = 0;
  sub_16B8280(qword_4FBF2C0, "sink-into-texture", 17);
  LODWORD(qword_4FBF2C0[20]) = 3;
  BYTE4(qword_4FBF2C0[22]) = 1;
  LODWORD(qword_4FBF2C0[22]) = 3;
  qword_4FBF2C0[6] = 142;
  BYTE4(qword_4FBF2C0[1]) = BYTE4(qword_4FBF2C0[1]) & 0x9F | 0x20;
  qword_4FBF2C0[5] = "Enable sinking into Texture blocks, 1 for cross-block only, 2 for cross and intra-block, 3 for also"
                     " considering instructions used outside only";
  sub_16B88A0(qword_4FBF2C0);
  __cxa_atexit(sub_12EDEA0, qword_4FBF2C0, &qword_4A427C0);
  qword_4FBF1E0[0] = &unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FBF1E0[1]) &= 0xF000u;
  LODWORD(qword_4FBF1E0[1]) = v1;
  qword_4FBF1E0[21] = &unk_49E74C8;
  qword_4FBF1E0[0] = &unk_49EEB70;
  qword_4FBF1E0[23] = &unk_49EEDF0;
  qword_4FBF1E0[9] = qword_4FA01C0;
  qword_4FBF1E0[11] = &qword_4FBF1E0[15];
  qword_4FBF1E0[12] = &qword_4FBF1E0[15];
  qword_4FBF1E0[2] = 0;
  qword_4FBF1E0[3] = 0;
  qword_4FBF1E0[4] = 0;
  qword_4FBF1E0[5] = 0;
  qword_4FBF1E0[6] = 0;
  qword_4FBF1E0[7] = 0;
  qword_4FBF1E0[8] = 0;
  qword_4FBF1E0[10] = 0;
  qword_4FBF1E0[13] = 4;
  LODWORD(qword_4FBF1E0[14]) = 0;
  LOBYTE(qword_4FBF1E0[19]) = 0;
  LODWORD(qword_4FBF1E0[20]) = 0;
  BYTE4(qword_4FBF1E0[22]) = 1;
  LODWORD(qword_4FBF1E0[22]) = 0;
  sub_16B8280(qword_4FBF1E0, "sink-limit", 10);
  LODWORD(qword_4FBF1E0[20]) = 20;
  BYTE4(qword_4FBF1E0[22]) = 1;
  LODWORD(qword_4FBF1E0[22]) = 20;
  qword_4FBF1E0[6] = 38;
  BYTE4(qword_4FBF1E0[1]) = BYTE4(qword_4FBF1E0[1]) & 0x9F | 0x20;
  qword_4FBF1E0[5] = "Control number of instructions to Sink";
  sub_16B88A0(qword_4FBF1E0);
  __cxa_atexit(sub_12EDEA0, qword_4FBF1E0, &qword_4A427C0);
  qword_4FBF100[0] = &unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FBF100[1]) &= 0xF000u;
  LODWORD(qword_4FBF100[1]) = v2;
  qword_4FBF100[11] = &qword_4FBF100[15];
  qword_4FBF100[12] = &qword_4FBF100[15];
  qword_4FBF100[9] = qword_4FA01C0;
  qword_4FBF100[2] = 0;
  qword_4FBF100[21] = &unk_49E74E8;
  LOWORD(qword_4FBF100[22]) = 256;
  qword_4FBF100[3] = 0;
  qword_4FBF100[4] = 0;
  qword_4FBF100[0] = &unk_49EEC70;
  qword_4FBF100[5] = 0;
  qword_4FBF100[6] = 0;
  qword_4FBF100[23] = &unk_49EEDB0;
  qword_4FBF100[7] = 0;
  qword_4FBF100[8] = 0;
  qword_4FBF100[10] = 0;
  qword_4FBF100[13] = 4;
  LODWORD(qword_4FBF100[14]) = 0;
  LOBYTE(qword_4FBF100[19]) = 0;
  LOBYTE(qword_4FBF100[20]) = 0;
  sub_16B8280(qword_4FBF100, "dump-sink2", 10);
  LOWORD(qword_4FBF100[22]) = 256;
  LOBYTE(qword_4FBF100[20]) = 0;
  qword_4FBF100[6] = 33;
  BYTE4(qword_4FBF100[1]) = BYTE4(qword_4FBF100[1]) & 0x9F | 0x20;
  qword_4FBF100[5] = "Dumping information for debugging";
  sub_16B88A0(qword_4FBF100);
  return __cxa_atexit(sub_12EDEC0, qword_4FBF100, &qword_4A427C0);
}
