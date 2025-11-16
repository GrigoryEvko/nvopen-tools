// Function: ctor_585
// Address: 0x57a2d0
//
int __fastcall ctor_585(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_50246A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_50246AC &= 0x8000u;
  word_50246B0 = 0;
  qword_50246F0 = 0x100000000LL;
  qword_50246B8 = 0;
  qword_50246C0 = 0;
  qword_50246C8 = 0;
  dword_50246A8 = v4;
  qword_50246D0 = 0;
  qword_50246D8 = 0;
  qword_50246E0 = 0;
  qword_50246E8 = (__int64)&unk_50246F8;
  qword_5024700 = 0;
  qword_5024708 = (__int64)&unk_5024720;
  qword_5024710 = 1;
  dword_5024718 = 0;
  byte_502471C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50246F0;
  v7 = (unsigned int)qword_50246F0 + 1LL;
  if ( v7 > HIDWORD(qword_50246F0) )
  {
    sub_C8D5F0((char *)&unk_50246F8 - 16, &unk_50246F8, v7, 8);
    v6 = (unsigned int)qword_50246F0;
  }
  *(_QWORD *)(qword_50246E8 + 8 * v6) = v5;
  LODWORD(qword_50246F0) = qword_50246F0 + 1;
  qword_5024728 = 0;
  qword_5024730 = (__int64)&unk_49D9728;
  qword_5024738 = 0;
  qword_50246A0 = (__int64)&unk_49DBF10;
  qword_5024740 = (__int64)&unk_49DC290;
  qword_5024760 = (__int64)nullsub_24;
  qword_5024758 = (__int64)sub_984050;
  sub_C53080(&qword_50246A0, "stress-regalloc", 15);
  LODWORD(qword_5024728) = 0;
  BYTE4(qword_5024738) = 1;
  LODWORD(qword_5024738) = 0;
  qword_50246E0 = 1;
  LOBYTE(dword_50246AC) = dword_50246AC & 0x9F | 0x20;
  qword_50246D8 = (__int64)"N";
  qword_50246C8 = (__int64)"Limit all regclasses to N registers";
  qword_50246D0 = 35;
  sub_C53130(&qword_50246A0);
  return __cxa_atexit(sub_984970, &qword_50246A0, &qword_4A427C0);
}
