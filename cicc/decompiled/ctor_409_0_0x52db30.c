// Function: ctor_409_0
// Address: 0x52db30
//
int ctor_409_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+1Ch] [rbp-44h] BYREF
  const char *v22; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+28h] [rbp-38h]

  qword_4FEDC00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEDC50 = 0x100000000LL;
  dword_4FEDC0C &= 0x8000u;
  word_4FEDC10 = 0;
  qword_4FEDC18 = 0;
  qword_4FEDC20 = 0;
  dword_4FEDC08 = v0;
  qword_4FEDC28 = 0;
  qword_4FEDC30 = 0;
  qword_4FEDC38 = 0;
  qword_4FEDC40 = 0;
  qword_4FEDC48 = (__int64)&unk_4FEDC58;
  qword_4FEDC60 = 0;
  qword_4FEDC68 = (__int64)&unk_4FEDC80;
  qword_4FEDC70 = 1;
  dword_4FEDC78 = 0;
  byte_4FEDC7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEDC50;
  v3 = (unsigned int)qword_4FEDC50 + 1LL;
  if ( v3 > HIDWORD(qword_4FEDC50) )
  {
    sub_C8D5F0((char *)&unk_4FEDC58 - 16, &unk_4FEDC58, v3, 8);
    v2 = (unsigned int)qword_4FEDC50;
  }
  *(_QWORD *)(qword_4FEDC48 + 8 * v2) = v1;
  LODWORD(qword_4FEDC50) = qword_4FEDC50 + 1;
  qword_4FEDC88 = 0;
  qword_4FEDC90 = (__int64)&unk_49DA090;
  qword_4FEDC98 = 0;
  qword_4FEDC00 = (__int64)&unk_49DBF90;
  qword_4FEDCA0 = (__int64)&unk_49DC230;
  qword_4FEDCC0 = (__int64)nullsub_58;
  qword_4FEDCB8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FEDC00, "sanitizer-coverage-level", 24);
  qword_4FEDC30 = 92;
  qword_4FEDC28 = (__int64)"Sanitizer Coverage. 0: none, 1: entry block, 2: all blocks, 3: all blocks and critical edges";
  LOBYTE(dword_4FEDC0C) = dword_4FEDC0C & 0x9F | 0x20;
  sub_C53130(&qword_4FEDC00);
  __cxa_atexit(sub_B2B680, &qword_4FEDC00, &qword_4A427C0);
  v22 = "Experimental pc tracing";
  v21 = 1;
  v23 = 23;
  sub_24C49B0(&unk_4FEDB20, "sanitizer-coverage-trace-pc", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FEDB20, &qword_4A427C0);
  v22 = "pc tracing with a guard";
  v21 = 1;
  v23 = 23;
  sub_24C4BA0(&unk_4FEDA40, "sanitizer-coverage-trace-pc-guard", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FEDA40, &qword_4A427C0);
  v22 = "create a static PC table";
  v21 = 1;
  v23 = 24;
  sub_24C49B0(&unk_4FED960, "sanitizer-coverage-pc-table", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED960, &qword_4A427C0);
  qword_4FED880 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FED8D0 = 0x100000000LL;
  dword_4FED88C &= 0x8000u;
  qword_4FED8C8 = (__int64)&unk_4FED8D8;
  word_4FED890 = 0;
  qword_4FED898 = 0;
  dword_4FED888 = v4;
  qword_4FED8A0 = 0;
  qword_4FED8A8 = 0;
  qword_4FED8B0 = 0;
  qword_4FED8B8 = 0;
  qword_4FED8C0 = 0;
  qword_4FED8E0 = 0;
  qword_4FED8E8 = (__int64)&unk_4FED900;
  qword_4FED8F0 = 1;
  dword_4FED8F8 = 0;
  byte_4FED8FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FED8D0;
  v7 = (unsigned int)qword_4FED8D0 + 1LL;
  if ( v7 > HIDWORD(qword_4FED8D0) )
  {
    sub_C8D5F0((char *)&unk_4FED8D8 - 16, &unk_4FED8D8, v7, 8);
    v6 = (unsigned int)qword_4FED8D0;
  }
  *(_QWORD *)(qword_4FED8C8 + 8 * v6) = v5;
  LODWORD(qword_4FED8D0) = qword_4FED8D0 + 1;
  qword_4FED908 = 0;
  qword_4FED910 = (__int64)&unk_49D9748;
  qword_4FED918 = 0;
  qword_4FED880 = (__int64)&unk_49DC090;
  qword_4FED920 = (__int64)&unk_49DC1D0;
  qword_4FED940 = (__int64)nullsub_23;
  qword_4FED938 = (__int64)sub_984030;
  sub_C53080(&qword_4FED880, "sanitizer-coverage-inline-8bit-counters", 39);
  qword_4FED8B0 = 39;
  qword_4FED8A8 = (__int64)"increments 8-bit counter for every edge";
  LOBYTE(dword_4FED88C) = dword_4FED88C & 0x9F | 0x20;
  sub_C53130(&qword_4FED880);
  __cxa_atexit(sub_984900, &qword_4FED880, &qword_4A427C0);
  qword_4FED7A0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FED7F0 = 0x100000000LL;
  dword_4FED7AC &= 0x8000u;
  qword_4FED7E8 = (__int64)&unk_4FED7F8;
  word_4FED7B0 = 0;
  qword_4FED7B8 = 0;
  dword_4FED7A8 = v8;
  qword_4FED7C0 = 0;
  qword_4FED7C8 = 0;
  qword_4FED7D0 = 0;
  qword_4FED7D8 = 0;
  qword_4FED7E0 = 0;
  qword_4FED800 = 0;
  qword_4FED808 = (__int64)&unk_4FED820;
  qword_4FED810 = 1;
  dword_4FED818 = 0;
  byte_4FED81C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FED7F0;
  if ( (unsigned __int64)(unsigned int)qword_4FED7F0 + 1 > HIDWORD(qword_4FED7F0) )
  {
    v19 = v9;
    sub_C8D5F0((char *)&unk_4FED7F8 - 16, &unk_4FED7F8, (unsigned int)qword_4FED7F0 + 1LL, 8);
    v10 = (unsigned int)qword_4FED7F0;
    v9 = v19;
  }
  *(_QWORD *)(qword_4FED7E8 + 8 * v10) = v9;
  qword_4FED830 = (__int64)&unk_49D9748;
  LODWORD(qword_4FED7F0) = qword_4FED7F0 + 1;
  qword_4FED828 = 0;
  qword_4FED7A0 = (__int64)&unk_49DC090;
  qword_4FED838 = 0;
  qword_4FED840 = (__int64)&unk_49DC1D0;
  qword_4FED860 = (__int64)nullsub_23;
  qword_4FED858 = (__int64)sub_984030;
  sub_C53080(&qword_4FED7A0, "sanitizer-coverage-inline-bool-flag", 35);
  qword_4FED7D0 = 34;
  qword_4FED7C8 = (__int64)"sets a boolean flag for every edge";
  LOBYTE(dword_4FED7AC) = dword_4FED7AC & 0x9F | 0x20;
  sub_C53130(&qword_4FED7A0);
  __cxa_atexit(sub_984900, &qword_4FED7A0, &qword_4A427C0);
  v22 = "Tracing of CMP and similar instructions";
  v21 = 1;
  v23 = 39;
  sub_24C4BA0(&unk_4FED6C0, "sanitizer-coverage-trace-compares", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED6C0, &qword_4A427C0);
  v22 = "Tracing of DIV instructions";
  v21 = 1;
  v23 = 27;
  sub_24C4D90(&unk_4FED5E0, "sanitizer-coverage-trace-divs", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED5E0, &qword_4A427C0);
  v22 = "Tracing of load instructions";
  v21 = 1;
  v23 = 28;
  sub_24C4F80(&unk_4FED500, "sanitizer-coverage-trace-loads", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED500, &qword_4A427C0);
  v22 = "Tracing of store instructions";
  v21 = 1;
  v23 = 29;
  sub_24C5170(&unk_4FED420, "sanitizer-coverage-trace-stores", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED420, &qword_4A427C0);
  v22 = "Tracing of GEP instructions";
  v21 = 1;
  v23 = 27;
  sub_24C4D90(&unk_4FED340, "sanitizer-coverage-trace-geps", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED340, &qword_4A427C0);
  qword_4FED260 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FED2DC = 1;
  word_4FED270 = 0;
  qword_4FED2B0 = 0x100000000LL;
  dword_4FED26C &= 0x8000u;
  qword_4FED2A8 = (__int64)&unk_4FED2B8;
  qword_4FED278 = 0;
  dword_4FED268 = v11;
  qword_4FED280 = 0;
  qword_4FED288 = 0;
  qword_4FED290 = 0;
  qword_4FED298 = 0;
  qword_4FED2A0 = 0;
  qword_4FED2C0 = 0;
  qword_4FED2C8 = (__int64)&unk_4FED2E0;
  qword_4FED2D0 = 1;
  dword_4FED2D8 = 0;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FED2B0;
  if ( (unsigned __int64)(unsigned int)qword_4FED2B0 + 1 > HIDWORD(qword_4FED2B0) )
  {
    v20 = v12;
    sub_C8D5F0((char *)&unk_4FED2B8 - 16, &unk_4FED2B8, (unsigned int)qword_4FED2B0 + 1LL, 8);
    v13 = (unsigned int)qword_4FED2B0;
    v12 = v20;
  }
  *(_QWORD *)(qword_4FED2A8 + 8 * v13) = v12;
  qword_4FED2F0 = (__int64)&unk_49D9748;
  LODWORD(qword_4FED2B0) = qword_4FED2B0 + 1;
  qword_4FED2E8 = 0;
  qword_4FED260 = (__int64)&unk_49DC090;
  qword_4FED2F8 = 0;
  qword_4FED300 = (__int64)&unk_49DC1D0;
  qword_4FED320 = (__int64)nullsub_23;
  qword_4FED318 = (__int64)sub_984030;
  sub_C53080(&qword_4FED260, "sanitizer-coverage-prune-blocks", 31);
  qword_4FED288 = (__int64)"Reduce the number of instrumented blocks";
  LOWORD(qword_4FED2F8) = 257;
  LOBYTE(qword_4FED2E8) = 1;
  qword_4FED290 = 40;
  LOBYTE(dword_4FED26C) = dword_4FED26C & 0x9F | 0x20;
  sub_C53130(&qword_4FED260);
  __cxa_atexit(sub_984900, &qword_4FED260, &qword_4A427C0);
  v22 = "max stack depth tracing";
  v21 = 1;
  v23 = 23;
  sub_24C4F80(&unk_4FED180, "sanitizer-coverage-stack-depth", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED180, &qword_4A427C0);
  v22 = "collect control flow for each function";
  v21 = 1;
  v23 = 38;
  sub_24C5170(&unk_4FED0A0, "sanitizer-coverage-control-flow", &v22, &v21);
  __cxa_atexit(sub_984900, &unk_4FED0A0, &qword_4A427C0);
  qword_4FECFC0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FED010 = 0x100000000LL;
  word_4FECFD0 = 0;
  dword_4FECFCC &= 0x8000u;
  qword_4FECFD8 = 0;
  qword_4FECFE0 = 0;
  dword_4FECFC8 = v14;
  qword_4FECFE8 = 0;
  qword_4FECFF0 = 0;
  qword_4FECFF8 = 0;
  qword_4FED000 = 0;
  qword_4FED008 = (__int64)&unk_4FED018;
  qword_4FED020 = 0;
  qword_4FED028 = (__int64)&unk_4FED040;
  qword_4FED030 = 1;
  dword_4FED038 = 0;
  byte_4FED03C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FED010;
  v17 = (unsigned int)qword_4FED010 + 1LL;
  if ( v17 > HIDWORD(qword_4FED010) )
  {
    sub_C8D5F0((char *)&unk_4FED018 - 16, &unk_4FED018, v17, 8);
    v16 = (unsigned int)qword_4FED010;
  }
  *(_QWORD *)(qword_4FED008 + 8 * v16) = v15;
  qword_4FED050 = (__int64)&unk_49D9748;
  LODWORD(qword_4FED010) = qword_4FED010 + 1;
  qword_4FED048 = 0;
  qword_4FECFC0 = (__int64)&unk_49DC090;
  qword_4FED058 = 0;
  qword_4FED060 = (__int64)&unk_49DC1D0;
  qword_4FED080 = (__int64)nullsub_23;
  qword_4FED078 = (__int64)sub_984030;
  sub_C53080(&qword_4FECFC0, "sanitizer-coverage-gated-trace-callbacks", 40);
  qword_4FECFF0 = 125;
  qword_4FECFE8 = (__int64)"Gate the invocation of the tracing callbacks on a global variable. Currently only supported f"
                           "or trace-pc-guard and trace-cmp.";
  LOBYTE(qword_4FED048) = 0;
  LOBYTE(dword_4FECFCC) = dword_4FECFCC & 0x9F | 0x20;
  LOWORD(qword_4FED058) = 256;
  sub_C53130(&qword_4FECFC0);
  return __cxa_atexit(sub_984900, &qword_4FECFC0, &qword_4A427C0);
}
