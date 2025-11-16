// Function: ctor_611
// Address: 0x589160
//
int __fastcall ctor_611(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_502D280 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_502D2FC = 1;
  qword_502D2D0 = 0x100000000LL;
  dword_502D28C &= 0x8000u;
  qword_502D298 = 0;
  qword_502D2A0 = 0;
  qword_502D2A8 = 0;
  dword_502D288 = v4;
  word_502D290 = 0;
  qword_502D2B0 = 0;
  qword_502D2B8 = 0;
  qword_502D2C0 = 0;
  qword_502D2C8 = (__int64)&unk_502D2D8;
  qword_502D2E0 = 0;
  qword_502D2E8 = (__int64)&unk_502D300;
  qword_502D2F0 = 1;
  dword_502D2F8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502D2D0;
  v7 = (unsigned int)qword_502D2D0 + 1LL;
  if ( v7 > HIDWORD(qword_502D2D0) )
  {
    sub_C8D5F0((char *)&unk_502D2D8 - 16, &unk_502D2D8, v7, 8);
    v6 = (unsigned int)qword_502D2D0;
  }
  *(_QWORD *)(qword_502D2C8 + 8 * v6) = v5;
  LODWORD(qword_502D2D0) = qword_502D2D0 + 1;
  qword_502D308 = 0;
  qword_502D310 = (__int64)&unk_49D9748;
  qword_502D318 = 0;
  qword_502D280 = (__int64)&unk_49DC090;
  qword_502D320 = (__int64)&unk_49DC1D0;
  qword_502D340 = (__int64)nullsub_23;
  qword_502D338 = (__int64)sub_984030;
  sub_C53080(&qword_502D280, "cta-reconfig-aware-mrpa", 23);
  LOBYTE(qword_502D308) = 1;
  LOWORD(qword_502D318) = 257;
  qword_502D2B0 = 61;
  LOBYTE(dword_502D28C) = dword_502D28C & 0x9F | 0x20;
  qword_502D2A8 = (__int64)"Enable CTA reconfig aware machine register pressure analysis.";
  sub_C53130(&qword_502D280);
  return __cxa_atexit(sub_984900, &qword_502D280, &qword_4A427C0);
}
