// Function: ctor_665
// Address: 0x59e6b0
//
int __fastcall ctor_665(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_503AE40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503AEBC = 1;
  qword_503AE90 = 0x100000000LL;
  dword_503AE4C &= 0x8000u;
  qword_503AE58 = 0;
  qword_503AE60 = 0;
  qword_503AE68 = 0;
  dword_503AE48 = v4;
  word_503AE50 = 0;
  qword_503AE70 = 0;
  qword_503AE78 = 0;
  qword_503AE80 = 0;
  qword_503AE88 = (__int64)&unk_503AE98;
  qword_503AEA0 = 0;
  qword_503AEA8 = (__int64)&unk_503AEC0;
  qword_503AEB0 = 1;
  dword_503AEB8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503AE90;
  v7 = (unsigned int)qword_503AE90 + 1LL;
  if ( v7 > HIDWORD(qword_503AE90) )
  {
    sub_C8D5F0((char *)&unk_503AE98 - 16, &unk_503AE98, v7, 8);
    v6 = (unsigned int)qword_503AE90;
  }
  *(_QWORD *)(qword_503AE88 + 8 * v6) = v5;
  qword_503AED0 = (__int64)&unk_49D9748;
  LODWORD(qword_503AE90) = qword_503AE90 + 1;
  qword_503AEC8 = 0;
  qword_503AE40 = (__int64)&unk_49DC090;
  qword_503AEE0 = (__int64)&unk_49DC1D0;
  qword_503AED8 = 0;
  qword_503AF00 = (__int64)nullsub_23;
  qword_503AEF8 = (__int64)sub_984030;
  sub_C53080(&qword_503AE40, "trap-unreachable", 16);
  qword_503AE70 = 38;
  LOBYTE(dword_503AE4C) = dword_503AE4C & 0x9F | 0x20;
  qword_503AE68 = (__int64)"Enable generating trap for unreachable";
  sub_C53130(&qword_503AE40);
  __cxa_atexit(sub_984900, &qword_503AE40, &qword_4A427C0);
  qword_503AD60 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503AE40, v8, v9), 1u);
  dword_503AD6C &= 0x8000u;
  word_503AD70 = 0;
  qword_503ADB0 = 0x100000000LL;
  qword_503AD78 = 0;
  qword_503AD80 = 0;
  qword_503AD88 = 0;
  dword_503AD68 = v10;
  qword_503AD90 = 0;
  qword_503AD98 = 0;
  qword_503ADA0 = 0;
  qword_503ADA8 = (__int64)&unk_503ADB8;
  qword_503ADC0 = 0;
  qword_503ADC8 = (__int64)&unk_503ADE0;
  qword_503ADD0 = 1;
  dword_503ADD8 = 0;
  byte_503ADDC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503ADB0;
  v13 = (unsigned int)qword_503ADB0 + 1LL;
  if ( v13 > HIDWORD(qword_503ADB0) )
  {
    sub_C8D5F0((char *)&unk_503ADB8 - 16, &unk_503ADB8, v13, 8);
    v12 = (unsigned int)qword_503ADB0;
  }
  *(_QWORD *)(qword_503ADA8 + 8 * v12) = v11;
  qword_503ADF0 = (__int64)&unk_49D9748;
  LODWORD(qword_503ADB0) = qword_503ADB0 + 1;
  qword_503ADE8 = 0;
  qword_503AD60 = (__int64)&unk_49DC090;
  qword_503AE00 = (__int64)&unk_49DC1D0;
  qword_503ADF8 = 0;
  qword_503AE20 = (__int64)nullsub_23;
  qword_503AE18 = (__int64)sub_984030;
  sub_C53080(&qword_503AD60, "no-trap-after-noreturn", 22);
  qword_503AD90 = 121;
  LOBYTE(dword_503AD6C) = dword_503AD6C & 0x9F | 0x20;
  qword_503AD88 = (__int64)"Do not emit a trap instruction for 'unreachable' IR instructions after noreturn calls, even i"
                           "f --trap-unreachable is set.";
  sub_C53130(&qword_503AD60);
  return __cxa_atexit(sub_984900, &qword_503AD60, &qword_4A427C0);
}
