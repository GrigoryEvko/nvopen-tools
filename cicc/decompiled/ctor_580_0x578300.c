// Function: ctor_580
// Address: 0x578300
//
int ctor_580()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5023360 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_502336C &= 0x8000u;
  word_5023370 = 0;
  qword_50233B0 = 0x100000000LL;
  qword_5023378 = 0;
  qword_5023380 = 0;
  qword_5023388 = 0;
  dword_5023368 = v0;
  qword_5023390 = 0;
  qword_5023398 = 0;
  qword_50233A0 = 0;
  qword_50233A8 = (__int64)&unk_50233B8;
  qword_50233C0 = 0;
  qword_50233C8 = (__int64)&unk_50233E0;
  qword_50233D0 = 1;
  dword_50233D8 = 0;
  byte_50233DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50233B0;
  v3 = (unsigned int)qword_50233B0 + 1LL;
  if ( v3 > HIDWORD(qword_50233B0) )
  {
    sub_C8D5F0((char *)&unk_50233B8 - 16, &unk_50233B8, v3, 8);
    v2 = (unsigned int)qword_50233B0;
  }
  *(_QWORD *)(qword_50233A8 + 8 * v2) = v1;
  LODWORD(qword_50233B0) = qword_50233B0 + 1;
  byte_5023400 = 0;
  qword_50233F0 = (__int64)&unk_4A2AA48;
  qword_50233E8 = 0;
  qword_50233F8 = 0;
  qword_5023360 = (__int64)&unk_4A2AA68;
  qword_5023408 = (__int64)&unk_49DC260;
  qword_5023428 = (__int64)nullsub_1640;
  qword_5023420 = (__int64)sub_2F3BD10;
  sub_C53080(&qword_5023360, "mem-intrinsic-expand-size", 25);
  qword_5023390 = 46;
  qword_5023388 = (__int64)"Set minimum mem intrinsic size to expand in IR";
  qword_50233E8 = -1;
  byte_5023400 = 1;
  qword_50233F8 = -1;
  LOBYTE(dword_502336C) = dword_502336C & 0x9F | 0x20;
  sub_C53130(&qword_5023360);
  return __cxa_atexit(sub_2F3C4C0, &qword_5023360, &qword_4A427C0);
}
