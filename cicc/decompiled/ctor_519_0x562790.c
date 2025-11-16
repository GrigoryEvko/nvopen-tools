// Function: ctor_519
// Address: 0x562790
//
int ctor_519()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5010960 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501096C &= 0x8000u;
  word_5010970 = 0;
  qword_50109B0 = 0x100000000LL;
  qword_5010978 = 0;
  qword_5010980 = 0;
  qword_5010988 = 0;
  dword_5010968 = v0;
  qword_5010990 = 0;
  qword_5010998 = 0;
  qword_50109A0 = 0;
  qword_50109A8 = (__int64)&unk_50109B8;
  qword_50109C0 = 0;
  qword_50109C8 = (__int64)&unk_50109E0;
  qword_50109D0 = 1;
  dword_50109D8 = 0;
  byte_50109DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50109B0;
  v3 = (unsigned int)qword_50109B0 + 1LL;
  if ( v3 > HIDWORD(qword_50109B0) )
  {
    sub_C8D5F0((char *)&unk_50109B8 - 16, &unk_50109B8, v3, 8);
    v2 = (unsigned int)qword_50109B0;
  }
  *(_QWORD *)(qword_50109A8 + 8 * v2) = v1;
  LODWORD(qword_50109B0) = qword_50109B0 + 1;
  qword_50109E8 = 0;
  qword_50109F0 = (__int64)&unk_49D9748;
  qword_50109F8 = 0;
  qword_5010960 = (__int64)&unk_49DC090;
  qword_5010A00 = (__int64)&unk_49DC1D0;
  qword_5010A20 = (__int64)nullsub_23;
  qword_5010A18 = (__int64)sub_984030;
  sub_C53080(&qword_5010960, "vplan-print-in-dot-format", 25);
  qword_5010990 = 56;
  LOBYTE(dword_501096C) = dword_501096C & 0x9F | 0x20;
  qword_5010988 = (__int64)"Use dot format instead of plain text when dumping VPlans";
  sub_C53130(&qword_5010960);
  return __cxa_atexit(sub_984900, &qword_5010960, &qword_4A427C0);
}
