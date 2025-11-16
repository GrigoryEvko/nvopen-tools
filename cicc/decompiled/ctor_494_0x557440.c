// Function: ctor_494
// Address: 0x557440
//
int ctor_494()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5009020 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500902C &= 0x8000u;
  word_5009030 = 0;
  qword_5009070 = 0x100000000LL;
  qword_5009038 = 0;
  qword_5009040 = 0;
  qword_5009048 = 0;
  dword_5009028 = v0;
  qword_5009050 = 0;
  qword_5009058 = 0;
  qword_5009060 = 0;
  qword_5009068 = (__int64)&unk_5009078;
  qword_5009080 = 0;
  qword_5009088 = (__int64)&unk_50090A0;
  qword_5009090 = 1;
  dword_5009098 = 0;
  byte_500909C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5009070;
  v3 = (unsigned int)qword_5009070 + 1LL;
  if ( v3 > HIDWORD(qword_5009070) )
  {
    sub_C8D5F0((char *)&unk_5009078 - 16, &unk_5009078, v3, 8);
    v2 = (unsigned int)qword_5009070;
  }
  *(_QWORD *)(qword_5009068 + 8 * v2) = v1;
  LODWORD(qword_5009070) = qword_5009070 + 1;
  qword_50090A8 = 0;
  qword_50090B0 = (__int64)&unk_49D9748;
  qword_50090B8 = 0;
  qword_5009020 = (__int64)&unk_49DC090;
  qword_50090C0 = (__int64)&unk_49DC1D0;
  qword_50090E0 = (__int64)nullsub_23;
  qword_50090D8 = (__int64)sub_984030;
  sub_C53080(&qword_5009020, "use-source-filename-for-promoted-locals", 39);
  qword_5009050 = 143;
  LOBYTE(dword_500902C) = dword_500902C & 0x9F | 0x20;
  qword_5009048 = (__int64)"Uses the source file name instead of the Module hash. This requires that the source filename "
                           "has a unique name / path to avoid name collisions.";
  sub_C53130(&qword_5009020);
  return __cxa_atexit(sub_984900, &qword_5009020, &qword_4A427C0);
}
