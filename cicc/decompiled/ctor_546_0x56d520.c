// Function: ctor_546
// Address: 0x56d520
//
int ctor_546()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_501D060 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501D0DC = 1;
  qword_501D0B0 = 0x100000000LL;
  dword_501D06C &= 0x8000u;
  qword_501D078 = 0;
  qword_501D080 = 0;
  qword_501D088 = 0;
  dword_501D068 = v0;
  word_501D070 = 0;
  qword_501D090 = 0;
  qword_501D098 = 0;
  qword_501D0A0 = 0;
  qword_501D0A8 = (__int64)&unk_501D0B8;
  qword_501D0C0 = 0;
  qword_501D0C8 = (__int64)&unk_501D0E0;
  qword_501D0D0 = 1;
  dword_501D0D8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501D0B0;
  v3 = (unsigned int)qword_501D0B0 + 1LL;
  if ( v3 > HIDWORD(qword_501D0B0) )
  {
    sub_C8D5F0((char *)&unk_501D0B8 - 16, &unk_501D0B8, v3, 8);
    v2 = (unsigned int)qword_501D0B0;
  }
  *(_QWORD *)(qword_501D0A8 + 8 * v2) = v1;
  LODWORD(qword_501D0B0) = qword_501D0B0 + 1;
  qword_501D0E8 = 0;
  qword_501D0F0 = (__int64)&unk_49D9728;
  qword_501D0F8 = 0;
  qword_501D060 = (__int64)&unk_49DBF10;
  qword_501D100 = (__int64)&unk_49DC290;
  qword_501D120 = (__int64)nullsub_24;
  qword_501D118 = (__int64)sub_984050;
  sub_C53080(&qword_501D060, "early-ifcvt-limit", 17);
  LODWORD(qword_501D0E8) = 30;
  BYTE4(qword_501D0F8) = 1;
  LODWORD(qword_501D0F8) = 30;
  qword_501D090 = 52;
  LOBYTE(dword_501D06C) = dword_501D06C & 0x9F | 0x20;
  qword_501D088 = (__int64)"Maximum number of instructions per speculated block.";
  sub_C53130(&qword_501D060);
  __cxa_atexit(sub_984970, &qword_501D060, &qword_4A427C0);
  qword_501CF80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501CF8C &= 0x8000u;
  word_501CF90 = 0;
  qword_501CFD0 = 0x100000000LL;
  qword_501CF98 = 0;
  qword_501CFA0 = 0;
  qword_501CFA8 = 0;
  dword_501CF88 = v4;
  qword_501CFB0 = 0;
  qword_501CFB8 = 0;
  qword_501CFC0 = 0;
  qword_501CFC8 = (__int64)&unk_501CFD8;
  qword_501CFE0 = 0;
  qword_501CFE8 = (__int64)&unk_501D000;
  qword_501CFF0 = 1;
  dword_501CFF8 = 0;
  byte_501CFFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_501CFD0;
  v7 = (unsigned int)qword_501CFD0 + 1LL;
  if ( v7 > HIDWORD(qword_501CFD0) )
  {
    sub_C8D5F0((char *)&unk_501CFD8 - 16, &unk_501CFD8, v7, 8);
    v6 = (unsigned int)qword_501CFD0;
  }
  *(_QWORD *)(qword_501CFC8 + 8 * v6) = v5;
  LODWORD(qword_501CFD0) = qword_501CFD0 + 1;
  qword_501D008 = 0;
  qword_501D010 = (__int64)&unk_49D9748;
  qword_501D018 = 0;
  qword_501CF80 = (__int64)&unk_49DC090;
  qword_501D020 = (__int64)&unk_49DC1D0;
  qword_501D040 = (__int64)nullsub_23;
  qword_501D038 = (__int64)sub_984030;
  sub_C53080(&qword_501CF80, "stress-early-ifcvt", 18);
  qword_501CFB0 = 20;
  LOBYTE(dword_501CF8C) = dword_501CF8C & 0x9F | 0x20;
  qword_501CFA8 = (__int64)"Turn all knobs to 11";
  sub_C53130(&qword_501CF80);
  return __cxa_atexit(sub_984900, &qword_501CF80, &qword_4A427C0);
}
