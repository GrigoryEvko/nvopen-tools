// Function: ctor_642
// Address: 0x595fb0
//
int __fastcall ctor_642(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_50359C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_50359CC &= 0x8000u;
  word_50359D0 = 0;
  qword_5035A10 = 0x100000000LL;
  qword_50359D8 = 0;
  qword_50359E0 = 0;
  qword_50359E8 = 0;
  dword_50359C8 = v4;
  qword_50359F0 = 0;
  qword_50359F8 = 0;
  qword_5035A00 = 0;
  qword_5035A08 = (__int64)&unk_5035A18;
  qword_5035A20 = 0;
  qword_5035A28 = (__int64)&unk_5035A40;
  qword_5035A30 = 1;
  dword_5035A38 = 0;
  byte_5035A3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5035A10;
  v7 = (unsigned int)qword_5035A10 + 1LL;
  if ( v7 > HIDWORD(qword_5035A10) )
  {
    sub_C8D5F0((char *)&unk_5035A18 - 16, &unk_5035A18, v7, 8);
    v6 = (unsigned int)qword_5035A10;
  }
  *(_QWORD *)(qword_5035A08 + 8 * v6) = v5;
  LODWORD(qword_5035A10) = qword_5035A10 + 1;
  qword_5035A48 = 0;
  qword_5035A50 = (__int64)&unk_49DA090;
  qword_5035A58 = 0;
  qword_50359C0 = (__int64)&unk_49DBF90;
  qword_5035A60 = (__int64)&unk_49DC230;
  qword_5035A80 = (__int64)nullsub_58;
  qword_5035A78 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_50359C0, "sbvec-cost-threshold", 20);
  LODWORD(qword_5035A48) = 0;
  BYTE4(qword_5035A58) = 1;
  LODWORD(qword_5035A58) = 0;
  qword_50359F0 = 29;
  LOBYTE(dword_50359CC) = dword_50359CC & 0x9F | 0x20;
  qword_50359E8 = (__int64)"Vectorization cost threshold.";
  sub_C53130(&qword_50359C0);
  return __cxa_atexit(sub_B2B680, &qword_50359C0, &qword_4A427C0);
}
