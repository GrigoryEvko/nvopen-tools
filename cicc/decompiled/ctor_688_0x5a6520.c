// Function: ctor_688
// Address: 0x5a6520
//
int __fastcall ctor_688(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5040040 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5040090 = 0x100000000LL;
  word_5040050 = 0;
  dword_504004C &= 0x8000u;
  qword_5040058 = 0;
  qword_5040060 = 0;
  dword_5040048 = v4;
  qword_5040068 = 0;
  qword_5040070 = 0;
  qword_5040078 = 0;
  qword_5040080 = 0;
  qword_5040088 = (__int64)&unk_5040098;
  qword_50400A0 = 0;
  qword_50400A8 = (__int64)&unk_50400C0;
  qword_50400B0 = 1;
  dword_50400B8 = 0;
  byte_50400BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040090;
  v7 = (unsigned int)qword_5040090 + 1LL;
  if ( v7 > HIDWORD(qword_5040090) )
  {
    sub_C8D5F0((char *)&unk_5040098 - 16, &unk_5040098, v7, 8);
    v6 = (unsigned int)qword_5040090;
  }
  *(_QWORD *)(qword_5040088 + 8 * v6) = v5;
  LODWORD(qword_5040090) = qword_5040090 + 1;
  qword_50400C8 = 0;
  qword_50400D0 = (__int64)&unk_49DC110;
  qword_50400D8 = 0;
  qword_5040040 = (__int64)&unk_49D97F0;
  qword_50400E0 = (__int64)&unk_49DC200;
  qword_5040100 = (__int64)nullsub_26;
  qword_50400F8 = (__int64)sub_9C26D0;
  sub_C53080(&qword_5040040, "enable-shrink-wrap", 18);
  qword_5040070 = 31;
  LOBYTE(dword_504004C) = dword_504004C & 0x9F | 0x20;
  qword_5040068 = (__int64)"enable the shrink-wrapping pass";
  sub_C53130(&qword_5040040);
  __cxa_atexit(sub_9C44F0, &qword_5040040, &qword_4A427C0);
  qword_503FF60 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_9C44F0, &qword_5040040, v8, v9), 1u);
  byte_503FFDC = 1;
  qword_503FFB0 = 0x100000000LL;
  dword_503FF6C &= 0x8000u;
  qword_503FF78 = 0;
  qword_503FF80 = 0;
  qword_503FF88 = 0;
  dword_503FF68 = v10;
  word_503FF70 = 0;
  qword_503FF90 = 0;
  qword_503FF98 = 0;
  qword_503FFA0 = 0;
  qword_503FFA8 = (__int64)&unk_503FFB8;
  qword_503FFC0 = 0;
  qword_503FFC8 = (__int64)&unk_503FFE0;
  qword_503FFD0 = 1;
  dword_503FFD8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503FFB0;
  v13 = (unsigned int)qword_503FFB0 + 1LL;
  if ( v13 > HIDWORD(qword_503FFB0) )
  {
    sub_C8D5F0((char *)&unk_503FFB8 - 16, &unk_503FFB8, v13, 8);
    v12 = (unsigned int)qword_503FFB0;
  }
  *(_QWORD *)(qword_503FFA8 + 8 * v12) = v11;
  LODWORD(qword_503FFB0) = qword_503FFB0 + 1;
  qword_503FFE8 = 0;
  qword_503FFF0 = (__int64)&unk_49D9748;
  qword_503FFF8 = 0;
  qword_503FF60 = (__int64)&unk_49DC090;
  qword_5040000 = (__int64)&unk_49DC1D0;
  qword_5040020 = (__int64)nullsub_23;
  qword_5040018 = (__int64)sub_984030;
  sub_C53080(&qword_503FF60, "enable-shrink-wrap-region-split", 31);
  LOBYTE(qword_503FFE8) = 1;
  LOWORD(qword_503FFF8) = 257;
  qword_503FF90 = 49;
  LOBYTE(dword_503FF6C) = dword_503FF6C & 0x9F | 0x20;
  qword_503FF88 = (__int64)"enable splitting of the restore block if possible";
  sub_C53130(&qword_503FF60);
  return __cxa_atexit(sub_984900, &qword_503FF60, &qword_4A427C0);
}
