// Function: ctor_560
// Address: 0x571390
//
_QWORD *ctor_560()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501EB40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501EBBC = 1;
  qword_501EB90 = 0x100000000LL;
  dword_501EB4C &= 0x8000u;
  qword_501EB58 = 0;
  qword_501EB60 = 0;
  qword_501EB68 = 0;
  dword_501EB48 = v0;
  word_501EB50 = 0;
  qword_501EB70 = 0;
  qword_501EB78 = 0;
  qword_501EB80 = 0;
  qword_501EB88 = (__int64)&unk_501EB98;
  qword_501EBA0 = 0;
  qword_501EBA8 = (__int64)&unk_501EBC0;
  qword_501EBB0 = 1;
  dword_501EBB8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501EB90;
  v3 = (unsigned int)qword_501EB90 + 1LL;
  if ( v3 > HIDWORD(qword_501EB90) )
  {
    sub_C8D5F0((char *)&unk_501EB98 - 16, &unk_501EB98, v3, 8);
    v2 = (unsigned int)qword_501EB90;
  }
  *(_QWORD *)(qword_501EB88 + 8 * v2) = v1;
  LODWORD(qword_501EB90) = qword_501EB90 + 1;
  qword_501EBC8 = 0;
  qword_501EBD0 = (__int64)&unk_49D9748;
  qword_501EBD8 = 0;
  qword_501EB40 = (__int64)&unk_49DC090;
  qword_501EBE0 = (__int64)&unk_49DC1D0;
  qword_501EC00 = (__int64)nullsub_23;
  qword_501EBF8 = (__int64)sub_984030;
  sub_C53080(&qword_501EB40, "print-slotindexes", 17);
  qword_501EB70 = 90;
  qword_501EB68 = (__int64)"When printing machine IR, annotate instructions and blocks with SlotIndexes when available";
  LOWORD(qword_501EBD8) = 257;
  LOBYTE(qword_501EBC8) = 1;
  LOBYTE(dword_501EB4C) = dword_501EB4C & 0x9F | 0x20;
  sub_C53130(&qword_501EB40);
  __cxa_atexit(sub_984900, &qword_501EB40, &qword_4A427C0);
  unk_501EB38 = 2;
  qword_501EB30 = 1;
  return &qword_501EB30;
}
