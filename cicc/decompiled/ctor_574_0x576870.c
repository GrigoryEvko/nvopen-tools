// Function: ctor_574
// Address: 0x576870
//
int ctor_574()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5022380 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50223FC = 1;
  qword_50223D0 = 0x100000000LL;
  dword_502238C &= 0x8000u;
  qword_5022398 = 0;
  qword_50223A0 = 0;
  qword_50223A8 = 0;
  dword_5022388 = v0;
  word_5022390 = 0;
  qword_50223B0 = 0;
  qword_50223B8 = 0;
  qword_50223C0 = 0;
  qword_50223C8 = (__int64)&unk_50223D8;
  qword_50223E0 = 0;
  qword_50223E8 = (__int64)&unk_5022400;
  qword_50223F0 = 1;
  dword_50223F8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50223D0;
  v3 = (unsigned int)qword_50223D0 + 1LL;
  if ( v3 > HIDWORD(qword_50223D0) )
  {
    sub_C8D5F0((char *)&unk_50223D8 - 16, &unk_50223D8, v3, 8);
    v2 = (unsigned int)qword_50223D0;
  }
  *(_QWORD *)(qword_50223C8 + 8 * v2) = v1;
  LODWORD(qword_50223D0) = qword_50223D0 + 1;
  qword_5022408 = 0;
  qword_5022410 = (__int64)&unk_49D9748;
  qword_5022418 = 0;
  qword_5022380 = (__int64)&unk_49DC090;
  qword_5022420 = (__int64)&unk_49DC1D0;
  qword_5022440 = (__int64)nullsub_23;
  qword_5022438 = (__int64)sub_984030;
  sub_C53080(&qword_5022380, "misched-fusion", 14);
  qword_50223B0 = 35;
  LOBYTE(qword_5022408) = 1;
  LOBYTE(dword_502238C) = dword_502238C & 0x9F | 0x20;
  qword_50223A8 = (__int64)"Enable scheduling for macro fusion.";
  LOWORD(qword_5022418) = 257;
  sub_C53130(&qword_5022380);
  return __cxa_atexit(sub_984900, &qword_5022380, &qword_4A427C0);
}
