// Function: ctor_566
// Address: 0x573460
//
int ctor_566()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501FF40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501FF4C &= 0x8000u;
  word_501FF50 = 0;
  qword_501FF90 = 0x100000000LL;
  qword_501FF58 = 0;
  qword_501FF60 = 0;
  qword_501FF68 = 0;
  dword_501FF48 = v0;
  qword_501FF70 = 0;
  qword_501FF78 = 0;
  qword_501FF80 = 0;
  qword_501FF88 = (__int64)&unk_501FF98;
  qword_501FFA0 = 0;
  qword_501FFA8 = (__int64)&unk_501FFC0;
  qword_501FFB0 = 1;
  dword_501FFB8 = 0;
  byte_501FFBC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501FF90;
  v3 = (unsigned int)qword_501FF90 + 1LL;
  if ( v3 > HIDWORD(qword_501FF90) )
  {
    sub_C8D5F0((char *)&unk_501FF98 - 16, &unk_501FF98, v3, 8);
    v2 = (unsigned int)qword_501FF90;
  }
  *(_QWORD *)(qword_501FF88 + 8 * v2) = v1;
  LODWORD(qword_501FF90) = qword_501FF90 + 1;
  qword_501FFC8 = 0;
  qword_501FFD0 = (__int64)&unk_49D9728;
  qword_501FFD8 = 0;
  qword_501FF40 = (__int64)&unk_49DBF10;
  qword_501FFE0 = (__int64)&unk_49DC290;
  qword_5020000 = (__int64)nullsub_24;
  qword_501FFF8 = (__int64)sub_984050;
  sub_C53080(&qword_501FF40, "align-all-functions", 19);
  qword_501FF70 = 91;
  qword_501FF68 = (__int64)"Force the alignment of all functions in log2 format (e.g. 4 means align on 16B boundaries).";
  LODWORD(qword_501FFC8) = 0;
  BYTE4(qword_501FFD8) = 1;
  LODWORD(qword_501FFD8) = 0;
  LOBYTE(dword_501FF4C) = dword_501FF4C & 0x9F | 0x20;
  sub_C53130(&qword_501FF40);
  return __cxa_atexit(sub_984970, &qword_501FF40, &qword_4A427C0);
}
