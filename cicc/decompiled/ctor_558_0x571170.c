// Function: ctor_558
// Address: 0x571170
//
int ctor_558()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_501EA00 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_501EA0C = word_501EA0C & 0x8000;
  unk_501EA08 = v0;
  qword_501EA48[1] = 0x100000000LL;
  unk_501EA10 = 0;
  unk_501EA18 = 0;
  unk_501EA20 = 0;
  unk_501EA28 = 0;
  unk_501EA30 = 0;
  unk_501EA38 = 0;
  unk_501EA40 = 0;
  qword_501EA48[0] = &qword_501EA48[2];
  qword_501EA48[3] = 0;
  qword_501EA48[4] = &qword_501EA48[7];
  qword_501EA48[5] = 1;
  LODWORD(qword_501EA48[6]) = 0;
  BYTE4(qword_501EA48[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_501EA48[1]);
  if ( (unsigned __int64)LODWORD(qword_501EA48[1]) + 1 > HIDWORD(qword_501EA48[1]) )
  {
    sub_C8D5F0(qword_501EA48, &qword_501EA48[2], LODWORD(qword_501EA48[1]) + 1LL, 8);
    v2 = LODWORD(qword_501EA48[1]);
  }
  *(_QWORD *)(qword_501EA48[0] + 8 * v2) = v1;
  ++LODWORD(qword_501EA48[1]);
  qword_501EA48[8] = 0;
  qword_501EA48[9] = &unk_49D9748;
  qword_501EA48[10] = 0;
  qword_501EA00 = &unk_49DC090;
  qword_501EA48[11] = &unk_49DC1D0;
  qword_501EA48[15] = nullsub_23;
  qword_501EA48[14] = sub_984030;
  sub_C53080(&qword_501EA00, "use-segment-set-for-physregs", 28);
  LOBYTE(qword_501EA48[8]) = 1;
  unk_501EA30 = 67;
  LOBYTE(word_501EA0C) = word_501EA0C & 0x9F | 0x20;
  LOWORD(qword_501EA48[10]) = 257;
  unk_501EA28 = "Use segment set for the computation of the live ranges of physregs.";
  sub_C53130(&qword_501EA00);
  return __cxa_atexit(sub_984900, &qword_501EA00, &qword_4A427C0);
}
