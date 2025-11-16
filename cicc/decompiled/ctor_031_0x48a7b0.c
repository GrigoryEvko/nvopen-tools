// Function: ctor_031
// Address: 0x48a7b0
//
int ctor_031()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_4F82300 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8230C = word_4F8230C & 0x8000 | 1;
  unk_4F82310 = 0;
  qword_4F82348[1] = 0x100000000LL;
  unk_4F82308 = v0;
  unk_4F82318 = 0;
  unk_4F82320 = 0;
  unk_4F82328 = 0;
  unk_4F82330 = 0;
  unk_4F82338 = 0;
  unk_4F82340 = 0;
  qword_4F82348[0] = &qword_4F82348[2];
  qword_4F82348[3] = 0;
  qword_4F82348[4] = &qword_4F82348[7];
  qword_4F82348[5] = 1;
  LODWORD(qword_4F82348[6]) = 0;
  BYTE4(qword_4F82348[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F82348[1]);
  if ( (unsigned __int64)LODWORD(qword_4F82348[1]) + 1 > HIDWORD(qword_4F82348[1]) )
  {
    sub_C8D5F0(qword_4F82348, &qword_4F82348[2], LODWORD(qword_4F82348[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F82348[1]);
  }
  *(_QWORD *)(qword_4F82348[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F82348[1]);
  qword_4F82348[8] = 0;
  qword_4F82300 = &unk_49DAD08;
  qword_4F82348[9] = 0;
  qword_4F82348[10] = 0;
  qword_4F82348[18] = &unk_49DC350;
  qword_4F82348[11] = 0;
  qword_4F82348[22] = nullsub_81;
  qword_4F82348[12] = 0;
  qword_4F82348[21] = sub_BB8600;
  qword_4F82348[13] = 0;
  LOBYTE(qword_4F82348[14]) = 0;
  qword_4F82348[15] = 0;
  qword_4F82348[16] = 0;
  qword_4F82348[17] = 0;
  sub_C53080(&qword_4F82300, "opt-bisect-funcs", 16);
  HIBYTE(word_4F8230C) |= 2u;
  unk_4F82338 = "function names";
  unk_4F82328 = "Only perform opt bisect for functions that are includedin this list and if empty, apply to all functions.";
  unk_4F82340 = 14;
  unk_4F82330 = 105;
  LOBYTE(word_4F8230C) = word_4F8230C & 0x9F | 0x20;
  sub_C53130(&qword_4F82300);
  return __cxa_atexit(sub_BB89D0, &qword_4F82300, &qword_4A427C0);
}
