// Function: ctor_495
// Address: 0x557640
//
int ctor_495()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5009100 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500910C &= 0x8000u;
  word_5009110 = 0;
  qword_5009150 = 0x100000000LL;
  qword_5009118 = 0;
  qword_5009120 = 0;
  qword_5009128 = 0;
  dword_5009108 = v0;
  qword_5009130 = 0;
  qword_5009138 = 0;
  qword_5009140 = 0;
  qword_5009148 = (__int64)&unk_5009158;
  qword_5009160 = 0;
  qword_5009168 = (__int64)&unk_5009180;
  qword_5009170 = 1;
  dword_5009178 = 0;
  byte_500917C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5009150;
  v3 = (unsigned int)qword_5009150 + 1LL;
  if ( v3 > HIDWORD(qword_5009150) )
  {
    sub_C8D5F0((char *)&unk_5009158 - 16, &unk_5009158, v3, 8);
    v2 = (unsigned int)qword_5009150;
  }
  *(_QWORD *)(qword_5009148 + 8 * v2) = v1;
  LODWORD(qword_5009150) = qword_5009150 + 1;
  qword_5009188 = 0;
  qword_5009190 = (__int64)&unk_49D9728;
  qword_5009198 = 0;
  qword_5009100 = (__int64)&unk_49DBF10;
  qword_50091A0 = (__int64)&unk_49DC290;
  qword_50091C0 = (__int64)nullsub_24;
  qword_50091B8 = (__int64)sub_984050;
  sub_C53080(&qword_5009100, "guards-predicate-pass-branch-weight", 35);
  LODWORD(qword_5009188) = 0x100000;
  BYTE4(qword_5009198) = 1;
  LODWORD(qword_5009198) = 0x100000;
  qword_5009130 = 100;
  LOBYTE(dword_500910C) = dword_500910C & 0x9F | 0x20;
  qword_5009128 = (__int64)"The probability of a guard failing is assumed to be the reciprocal of this value (default = 1 << 20)";
  sub_C53130(&qword_5009100);
  return __cxa_atexit(sub_984970, &qword_5009100, &qword_4A427C0);
}
