// Function: ctor_540
// Address: 0x56b7b0
//
int ctor_540()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v10; // [rsp+8h] [rbp-38h]

  qword_50163E0 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_50163EC = word_50163EC & 0x8000;
  qword_5016428[1] = 0x100000000LL;
  unk_50163E8 = v0;
  unk_50163F0 = 0;
  unk_50163F8 = 0;
  unk_5016400 = 0;
  unk_5016408 = 0;
  unk_5016410 = 0;
  unk_5016418 = 0;
  unk_5016420 = 0;
  qword_5016428[0] = &qword_5016428[2];
  qword_5016428[3] = 0;
  qword_5016428[4] = &qword_5016428[7];
  qword_5016428[5] = 1;
  LODWORD(qword_5016428[6]) = 0;
  BYTE4(qword_5016428[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_5016428[1]);
  if ( (unsigned __int64)LODWORD(qword_5016428[1]) + 1 > HIDWORD(qword_5016428[1]) )
  {
    sub_C8D5F0(qword_5016428, &qword_5016428[2], LODWORD(qword_5016428[1]) + 1LL, 8);
    v2 = LODWORD(qword_5016428[1]);
  }
  *(_QWORD *)(qword_5016428[0] + 8 * v2) = v1;
  qword_5016428[9] = &unk_49DA090;
  ++LODWORD(qword_5016428[1]);
  qword_5016428[8] = 0;
  qword_50163E0 = &unk_49DBF90;
  qword_5016428[10] = 0;
  qword_5016428[11] = &unk_49DC230;
  qword_5016428[15] = nullsub_58;
  qword_5016428[14] = sub_B2B5F0;
  sub_C53080(&qword_50163E0, "sink-into-texture", 17);
  LODWORD(qword_5016428[8]) = 3;
  BYTE4(qword_5016428[10]) = 1;
  LODWORD(qword_5016428[10]) = 3;
  unk_5016410 = 142;
  LOBYTE(word_50163EC) = word_50163EC & 0x9F | 0x20;
  unk_5016408 = "Enable sinking into Texture blocks, 1 for cross-block only, 2 for cross and intra-block, 3 for also cons"
                "idering instructions used outside only";
  sub_C53130(&qword_50163E0);
  __cxa_atexit(sub_B2B680, &qword_50163E0, &qword_4A427C0);
  qword_5016300 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501630C = word_501630C & 0x8000;
  qword_5016348[1] = 0x100000000LL;
  unk_5016308 = v3;
  unk_5016310 = 0;
  unk_5016318 = 0;
  unk_5016320 = 0;
  unk_5016328 = 0;
  unk_5016330 = 0;
  unk_5016338 = 0;
  unk_5016340 = 0;
  qword_5016348[0] = &qword_5016348[2];
  qword_5016348[3] = 0;
  qword_5016348[4] = &qword_5016348[7];
  qword_5016348[5] = 1;
  LODWORD(qword_5016348[6]) = 0;
  BYTE4(qword_5016348[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_5016348[1]);
  if ( (unsigned __int64)LODWORD(qword_5016348[1]) + 1 > HIDWORD(qword_5016348[1]) )
  {
    v10 = v4;
    sub_C8D5F0(qword_5016348, &qword_5016348[2], LODWORD(qword_5016348[1]) + 1LL, 8);
    v5 = LODWORD(qword_5016348[1]);
    v4 = v10;
  }
  *(_QWORD *)(qword_5016348[0] + 8 * v5) = v4;
  qword_5016348[9] = &unk_49DA090;
  ++LODWORD(qword_5016348[1]);
  qword_5016348[8] = 0;
  qword_5016300 = &unk_49DBF90;
  qword_5016348[10] = 0;
  qword_5016348[11] = &unk_49DC230;
  qword_5016348[15] = nullsub_58;
  qword_5016348[14] = sub_B2B5F0;
  sub_C53080(&qword_5016300, "sink-limit", 10);
  BYTE4(qword_5016348[10]) = 1;
  LODWORD(qword_5016348[8]) = 20;
  unk_5016330 = 38;
  LODWORD(qword_5016348[10]) = 20;
  LOBYTE(word_501630C) = word_501630C & 0x9F | 0x20;
  unk_5016328 = "Control number of instructions to Sink";
  sub_C53130(&qword_5016300);
  __cxa_atexit(sub_B2B680, &qword_5016300, &qword_4A427C0);
  qword_5016220 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501622C = word_501622C & 0x8000;
  unk_5016228 = v6;
  qword_5016268[1] = 0x100000000LL;
  unk_5016230 = 0;
  unk_5016238 = 0;
  unk_5016240 = 0;
  unk_5016248 = 0;
  unk_5016250 = 0;
  unk_5016258 = 0;
  unk_5016260 = 0;
  qword_5016268[0] = &qword_5016268[2];
  qword_5016268[3] = 0;
  qword_5016268[4] = &qword_5016268[7];
  qword_5016268[5] = 1;
  LODWORD(qword_5016268[6]) = 0;
  BYTE4(qword_5016268[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_5016268[1]);
  if ( (unsigned __int64)LODWORD(qword_5016268[1]) + 1 > HIDWORD(qword_5016268[1]) )
  {
    sub_C8D5F0(qword_5016268, &qword_5016268[2], LODWORD(qword_5016268[1]) + 1LL, 8);
    v8 = LODWORD(qword_5016268[1]);
  }
  *(_QWORD *)(qword_5016268[0] + 8 * v8) = v7;
  ++LODWORD(qword_5016268[1]);
  qword_5016268[8] = 0;
  qword_5016268[9] = &unk_49D9748;
  qword_5016268[10] = 0;
  qword_5016220 = &unk_49DC090;
  qword_5016268[11] = &unk_49DC1D0;
  qword_5016268[15] = nullsub_23;
  qword_5016268[14] = sub_984030;
  sub_C53080(&qword_5016220, "dump-sink2", 10);
  LOBYTE(qword_5016268[8]) = 0;
  unk_5016250 = 33;
  LOBYTE(word_501622C) = word_501622C & 0x9F | 0x20;
  LOWORD(qword_5016268[10]) = 256;
  unk_5016248 = "Dumping information for debugging";
  sub_C53130(&qword_5016220);
  return __cxa_atexit(sub_984900, &qword_5016220, &qword_4A427C0);
}
