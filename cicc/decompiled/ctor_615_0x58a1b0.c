// Function: ctor_615
// Address: 0x58a1b0
//
int ctor_615()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax

  sub_2208040(&unk_502D808);
  __cxa_atexit(sub_2208810, &unk_502D808, &qword_4A427C0);
  qword_502D740 = &unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_2208810, &unk_502D808, v0, v1), 1u);
  *(_DWORD *)&word_502D74C = word_502D74C & 0x8000;
  unk_502D750 = 0;
  qword_502D788[1] = 0x100000000LL;
  unk_502D748 = v2;
  unk_502D758 = 0;
  unk_502D760 = 0;
  unk_502D768 = 0;
  unk_502D770 = 0;
  unk_502D778 = 0;
  unk_502D780 = 0;
  qword_502D788[0] = &qword_502D788[2];
  qword_502D788[3] = 0;
  qword_502D788[4] = &qword_502D788[7];
  qword_502D788[5] = 1;
  LODWORD(qword_502D788[6]) = 0;
  BYTE4(qword_502D788[6]) = 1;
  v3 = sub_C57470();
  v4 = LODWORD(qword_502D788[1]);
  if ( (unsigned __int64)LODWORD(qword_502D788[1]) + 1 > HIDWORD(qword_502D788[1]) )
  {
    sub_C8D5F0(qword_502D788, &qword_502D788[2], LODWORD(qword_502D788[1]) + 1LL, 8);
    v4 = LODWORD(qword_502D788[1]);
  }
  *(_QWORD *)(qword_502D788[0] + 8 * v4) = v3;
  ++LODWORD(qword_502D788[1]);
  qword_502D788[8] = 0;
  qword_502D788[9] = &unk_49D9748;
  qword_502D788[10] = 0;
  qword_502D740 = &unk_49DC090;
  qword_502D788[11] = &unk_49DC1D0;
  qword_502D788[15] = nullsub_23;
  qword_502D788[14] = sub_984030;
  sub_C53080(&qword_502D740, "lnk-disable-allopts", 19);
  unk_502D770 = 35;
  LOBYTE(word_502D74C) = word_502D74C & 0x9F | 0x20;
  unk_502D768 = "Disable all lnk Optimization passes";
  sub_C53130(&qword_502D740);
  return __cxa_atexit(sub_984900, &qword_502D740, &qword_4A427C0);
}
