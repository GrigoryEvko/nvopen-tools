// Function: ctor_052
// Address: 0x490990
//
int ctor_052()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_4F86C60 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F86C6C = word_4F86C6C & 0x8000;
  unk_4F86C70 = 0;
  qword_4F86CA8[1] = 0x100000000LL;
  unk_4F86C68 = v0;
  unk_4F86C78 = 0;
  unk_4F86C80 = 0;
  unk_4F86C88 = 0;
  unk_4F86C90 = 0;
  unk_4F86C98 = 0;
  unk_4F86CA0 = 0;
  qword_4F86CA8[0] = &qword_4F86CA8[2];
  qword_4F86CA8[3] = 0;
  qword_4F86CA8[4] = &qword_4F86CA8[7];
  qword_4F86CA8[5] = 1;
  LODWORD(qword_4F86CA8[6]) = 0;
  BYTE4(qword_4F86CA8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F86CA8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F86CA8[1]) + 1 > HIDWORD(qword_4F86CA8[1]) )
  {
    sub_C8D5F0(qword_4F86CA8, &qword_4F86CA8[2], LODWORD(qword_4F86CA8[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F86CA8[1]);
  }
  *(_QWORD *)(qword_4F86CA8[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F86CA8[1]);
  qword_4F86CA8[8] = 0;
  qword_4F86CA8[9] = &unk_49D9728;
  qword_4F86CA8[10] = 0;
  qword_4F86C60 = &unk_49DBF10;
  qword_4F86CA8[11] = &unk_49DC290;
  qword_4F86CA8[15] = nullsub_24;
  qword_4F86CA8[14] = sub_984050;
  sub_C53080(&qword_4F86C60, "available-load-scan-limit", 25);
  LODWORD(qword_4F86CA8[8]) = 6;
  BYTE4(qword_4F86CA8[10]) = 1;
  LODWORD(qword_4F86CA8[10]) = 6;
  unk_4F86C90 = 147;
  LOBYTE(word_4F86C6C) = word_4F86C6C & 0x9F | 0x20;
  unk_4F86C88 = "Use this to specify the default maximum number of instructions to scan backward from a given instruction"
                ", when searching for available loaded value";
  sub_C53130(&qword_4F86C60);
  return __cxa_atexit(sub_984970, &qword_4F86C60, &qword_4A427C0);
}
