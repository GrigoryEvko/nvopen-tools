// Function: ctor_663
// Address: 0x59dae0
//
int __fastcall ctor_663(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax

  qword_503A820 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_503A82C = word_503A82C & 0x8000;
  unk_503A830 = 0;
  qword_503A868[1] = 0x100000000LL;
  unk_503A828 = v4;
  unk_503A838 = 0;
  unk_503A840 = 0;
  unk_503A848 = 0;
  unk_503A850 = 0;
  unk_503A858 = 0;
  unk_503A860 = 0;
  qword_503A868[0] = &qword_503A868[2];
  qword_503A868[3] = 0;
  qword_503A868[4] = &qword_503A868[7];
  qword_503A868[5] = 1;
  LODWORD(qword_503A868[6]) = 0;
  BYTE4(qword_503A868[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_503A868[1]);
  if ( (unsigned __int64)LODWORD(qword_503A868[1]) + 1 > HIDWORD(qword_503A868[1]) )
  {
    sub_C8D5F0(qword_503A868, &qword_503A868[2], LODWORD(qword_503A868[1]) + 1LL, 8);
    v6 = LODWORD(qword_503A868[1]);
  }
  *(_QWORD *)(qword_503A868[0] + 8 * v6) = v5;
  ++LODWORD(qword_503A868[1]);
  qword_503A868[8] = 0;
  qword_503A868[9] = &unk_49D9728;
  qword_503A868[10] = 0;
  qword_503A820 = &unk_49DBF10;
  qword_503A868[11] = &unk_49DC290;
  qword_503A868[15] = nullsub_24;
  qword_503A868[14] = sub_984050;
  sub_C53080(&qword_503A820, "partial-unrolling-threshold", 27);
  LODWORD(qword_503A868[8]) = 0;
  unk_503A848 = "Threshold for partial unrolling";
  BYTE4(qword_503A868[10]) = 1;
  LODWORD(qword_503A868[10]) = 0;
  unk_503A850 = 31;
  LOBYTE(word_503A82C) = word_503A82C & 0x9F | 0x20;
  sub_C53130(&qword_503A820);
  return __cxa_atexit(sub_984970, &qword_503A820, &qword_4A427C0);
}
