// Function: ctor_653
// Address: 0x59ae00
//
int __fastcall ctor_653(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5039060 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_503906C &= 0x8000u;
  word_5039070 = 0;
  qword_50390B0 = 0x100000000LL;
  qword_5039078 = 0;
  qword_5039080 = 0;
  qword_5039088 = 0;
  dword_5039068 = v4;
  qword_5039090 = 0;
  qword_5039098 = 0;
  qword_50390A0 = 0;
  qword_50390A8 = (__int64)&unk_50390B8;
  qword_50390C0 = 0;
  qword_50390C8 = (__int64)&unk_50390E0;
  qword_50390D0 = 1;
  dword_50390D8 = 0;
  byte_50390DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50390B0;
  v7 = (unsigned int)qword_50390B0 + 1LL;
  if ( v7 > HIDWORD(qword_50390B0) )
  {
    sub_C8D5F0((char *)&unk_50390B8 - 16, &unk_50390B8, v7, 8);
    v6 = (unsigned int)qword_50390B0;
  }
  *(_QWORD *)(qword_50390A8 + 8 * v6) = v5;
  LODWORD(qword_50390B0) = qword_50390B0 + 1;
  qword_50390E8 = 0;
  qword_50390F0 = (__int64)&unk_49DA090;
  qword_50390F8 = 0;
  qword_5039060 = (__int64)&unk_49DBF90;
  qword_5039100 = (__int64)&unk_49DC230;
  qword_5039120 = (__int64)nullsub_58;
  qword_5039118 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5039060, "sched-high-latency-cycles", 25);
  LODWORD(qword_50390E8) = 10;
  BYTE4(qword_50390F8) = 1;
  LODWORD(qword_50390F8) = 10;
  qword_5039090 = 105;
  LOBYTE(dword_503906C) = dword_503906C & 0x9F | 0x20;
  qword_5039088 = (__int64)"Roughly estimate the number of cycles that 'long latency' instructions take for targets with no itinerary";
  sub_C53130(&qword_5039060);
  return __cxa_atexit(sub_B2B680, &qword_5039060, &qword_4A427C0);
}
