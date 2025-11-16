// Function: ctor_050
// Address: 0x490710
//
int ctor_050()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F86AA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F86AAC &= 0x8000u;
  word_4F86AB0 = 0;
  qword_4F86AF0 = 0x100000000LL;
  qword_4F86AB8 = 0;
  qword_4F86AC0 = 0;
  qword_4F86AC8 = 0;
  dword_4F86AA8 = v0;
  qword_4F86AD0 = 0;
  qword_4F86AD8 = 0;
  qword_4F86AE0 = 0;
  qword_4F86AE8 = (__int64)&unk_4F86AF8;
  qword_4F86B00 = 0;
  qword_4F86B08 = (__int64)&unk_4F86B20;
  qword_4F86B10 = 1;
  dword_4F86B18 = 0;
  byte_4F86B1C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F86AF0;
  v3 = (unsigned int)qword_4F86AF0 + 1LL;
  if ( v3 > HIDWORD(qword_4F86AF0) )
  {
    sub_C8D5F0((char *)&unk_4F86AF8 - 16, &unk_4F86AF8, v3, 8);
    v2 = (unsigned int)qword_4F86AF0;
  }
  *(_QWORD *)(qword_4F86AE8 + 8 * v2) = v1;
  LODWORD(qword_4F86AF0) = qword_4F86AF0 + 1;
  qword_4F86B28 = 0;
  qword_4F86B30 = (__int64)&unk_49D9728;
  qword_4F86B38 = 0;
  qword_4F86AA0 = (__int64)&unk_49DBF10;
  qword_4F86B40 = (__int64)&unk_49DC290;
  qword_4F86B60 = (__int64)nullsub_24;
  qword_4F86B58 = (__int64)sub_984050;
  sub_C53080(&qword_4F86AA0, "capture-tracking-max-uses-to-explore", 36);
  qword_4F86AD0 = 34;
  LODWORD(qword_4F86B28) = 5000;
  BYTE4(qword_4F86B38) = 1;
  LODWORD(qword_4F86B38) = 5000;
  LOBYTE(dword_4F86AAC) = dword_4F86AAC & 0x9F | 0x20;
  qword_4F86AC8 = (__int64)"Maximal number of uses to explore.";
  sub_C53130(&qword_4F86AA0);
  return __cxa_atexit(sub_984970, &qword_4F86AA0, &qword_4A427C0);
}
