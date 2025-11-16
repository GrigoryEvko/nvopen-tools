// Function: ctor_072
// Address: 0x499580
//
int ctor_072()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned __int64 v6; // rdx

  qword_4F8C220 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8C22C = word_4F8C22C & 0x8000;
  qword_4F8C268[1] = 0x100000000LL;
  unk_4F8C228 = v0;
  unk_4F8C230 = 0;
  unk_4F8C238 = 0;
  unk_4F8C240 = 0;
  unk_4F8C248 = 0;
  unk_4F8C250 = 0;
  unk_4F8C258 = 0;
  unk_4F8C260 = 0;
  qword_4F8C268[0] = &qword_4F8C268[2];
  qword_4F8C268[3] = 0;
  qword_4F8C268[4] = &qword_4F8C268[7];
  qword_4F8C268[5] = 1;
  LODWORD(qword_4F8C268[6]) = 0;
  BYTE4(qword_4F8C268[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F8C268[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8C268[1]) + 1 > HIDWORD(qword_4F8C268[1]) )
  {
    sub_C8D5F0(qword_4F8C268, &qword_4F8C268[2], LODWORD(qword_4F8C268[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F8C268[1]);
  }
  *(_QWORD *)(qword_4F8C268[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F8C268[1]);
  qword_4F8C268[8] = 0;
  qword_4F8C268[9] = &unk_49D9728;
  qword_4F8C268[10] = 0;
  qword_4F8C220 = &unk_49DBF10;
  qword_4F8C268[11] = &unk_49DC290;
  qword_4F8C268[15] = nullsub_24;
  qword_4F8C268[14] = sub_984050;
  sub_C53080(&qword_4F8C220, "scev-cheap-expansion-budget", 27);
  LODWORD(qword_4F8C268[8]) = 4;
  BYTE4(qword_4F8C268[10]) = 1;
  LODWORD(qword_4F8C268[10]) = 4;
  unk_4F8C250 = 121;
  LOBYTE(word_4F8C22C) = word_4F8C22C & 0x9F | 0x20;
  unk_4F8C248 = "When performing SCEV expansion only if it is cheap to do, this controls the budget that is considered ch"
                "eap (default = 4)";
  sub_C53130(&qword_4F8C220);
  __cxa_atexit(sub_984970, &qword_4F8C220, &qword_4A427C0);
  qword_4F8C140 = (__int64)&unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8C1BC = 1;
  qword_4F8C190 = 0x100000000LL;
  dword_4F8C14C &= 0x8000u;
  qword_4F8C158 = 0;
  qword_4F8C160 = 0;
  qword_4F8C168 = 0;
  dword_4F8C148 = v3;
  word_4F8C150 = 0;
  qword_4F8C170 = 0;
  qword_4F8C178 = 0;
  qword_4F8C180 = 0;
  qword_4F8C188 = (__int64)&unk_4F8C198;
  qword_4F8C1A0 = 0;
  qword_4F8C1A8 = (__int64)&unk_4F8C1C0;
  qword_4F8C1B0 = 1;
  dword_4F8C1B8 = 0;
  v4 = sub_C57470();
  v5 = (unsigned int)qword_4F8C190;
  v6 = (unsigned int)qword_4F8C190 + 1LL;
  if ( v6 > HIDWORD(qword_4F8C190) )
  {
    sub_C8D5F0((char *)&unk_4F8C198 - 16, &unk_4F8C198, v6, 8);
    v5 = (unsigned int)qword_4F8C190;
  }
  *(_QWORD *)(qword_4F8C188 + 8 * v5) = v4;
  LODWORD(qword_4F8C190) = qword_4F8C190 + 1;
  qword_4F8C1C8 = 0;
  qword_4F8C1D0 = (__int64)&unk_49D9748;
  qword_4F8C1D8 = 0;
  qword_4F8C140 = (__int64)&unk_49DC090;
  qword_4F8C1E0 = (__int64)&unk_49DC1D0;
  qword_4F8C200 = (__int64)nullsub_23;
  qword_4F8C1F8 = (__int64)sub_984030;
  sub_C53080(&qword_4F8C140, "scev-avoid-type-conversions-larger", 34);
  LOBYTE(qword_4F8C1C8) = 1;
  qword_4F8C170 = 55;
  LOBYTE(dword_4F8C14C) = dword_4F8C14C & 0x9F | 0x20;
  LOWORD(qword_4F8C1D8) = 257;
  qword_4F8C168 = (__int64)"Avoid type conversions to a larger type performing SCEV";
  sub_C53130(&qword_4F8C140);
  return __cxa_atexit(sub_984900, &qword_4F8C140, &qword_4A427C0);
}
