// Function: ctor_065
// Address: 0x4959f0
//
int ctor_065()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_4F8A360 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8A36C = word_4F8A36C & 0x8000;
  unk_4F8A370 = 0;
  qword_4F8A3A8[1] = 0x100000000LL;
  unk_4F8A368 = v0;
  unk_4F8A378 = 0;
  unk_4F8A380 = 0;
  unk_4F8A388 = 0;
  unk_4F8A390 = 0;
  unk_4F8A398 = 0;
  unk_4F8A3A0 = 0;
  qword_4F8A3A8[0] = &qword_4F8A3A8[2];
  qword_4F8A3A8[3] = 0;
  qword_4F8A3A8[4] = &qword_4F8A3A8[7];
  qword_4F8A3A8[5] = 1;
  LODWORD(qword_4F8A3A8[6]) = 0;
  BYTE4(qword_4F8A3A8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F8A3A8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8A3A8[1]) + 1 > HIDWORD(qword_4F8A3A8[1]) )
  {
    sub_C8D5F0(qword_4F8A3A8, &qword_4F8A3A8[2], LODWORD(qword_4F8A3A8[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F8A3A8[1]);
  }
  *(_QWORD *)(qword_4F8A3A8[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F8A3A8[1]);
  qword_4F8A3A8[8] = 0;
  qword_4F8A3A8[9] = &unk_49D9728;
  qword_4F8A3A8[10] = 0;
  qword_4F8A360 = &unk_49DBF10;
  qword_4F8A3A8[11] = &unk_49DC290;
  qword_4F8A3A8[15] = nullsub_24;
  qword_4F8A3A8[14] = sub_984050;
  sub_C53080(&qword_4F8A360, "asm-macro-max-nesting-depth", 27);
  LODWORD(qword_4F8A3A8[8]) = 20;
  BYTE4(qword_4F8A3A8[10]) = 1;
  LODWORD(qword_4F8A3A8[10]) = 20;
  unk_4F8A390 = 54;
  LOBYTE(word_4F8A36C) = word_4F8A36C & 0x9F | 0x20;
  unk_4F8A388 = "The maximum nesting depth allowed for assembly macros.";
  sub_C53130(&qword_4F8A360);
  return __cxa_atexit(sub_984970, &qword_4F8A360, &qword_4A427C0);
}
