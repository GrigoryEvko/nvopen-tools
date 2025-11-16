// Function: ctor_623
// Address: 0x58c350
//
int __fastcall ctor_623(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // edx
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  qword_502F040 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_502F04C = word_502F04C & 0x8000;
  qword_502F088[1] = 0x100000000LL;
  unk_502F048 = v4;
  unk_502F050 = 0;
  unk_502F058 = 0;
  unk_502F060 = 0;
  unk_502F068 = 0;
  unk_502F070 = 0;
  unk_502F078 = 0;
  unk_502F080 = 0;
  qword_502F088[0] = &qword_502F088[2];
  qword_502F088[3] = 0;
  qword_502F088[4] = &qword_502F088[7];
  qword_502F088[5] = 1;
  LODWORD(qword_502F088[6]) = 0;
  BYTE4(qword_502F088[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_502F088[1]);
  if ( (unsigned __int64)LODWORD(qword_502F088[1]) + 1 > HIDWORD(qword_502F088[1]) )
  {
    sub_C8D5F0(qword_502F088, &qword_502F088[2], LODWORD(qword_502F088[1]) + 1LL, 8);
    v6 = LODWORD(qword_502F088[1]);
  }
  *(_QWORD *)(qword_502F088[0] + 8 * v6) = v5;
  ++LODWORD(qword_502F088[1]);
  qword_502F088[8] = 0;
  qword_502F088[9] = &unk_49D9748;
  qword_502F088[10] = 0;
  qword_502F040 = &unk_49DC090;
  qword_502F088[11] = &unk_49DC1D0;
  qword_502F088[15] = nullsub_23;
  qword_502F088[14] = sub_984030;
  sub_C53080(&qword_502F040, "enable-detailed-function-properties", 35);
  LOWORD(qword_502F088[10]) = 256;
  LOBYTE(qword_502F088[8]) = 0;
  unk_502F070 = 55;
  LOBYTE(word_502F04C) = word_502F04C & 0x9F | 0x20;
  unk_502F068 = "Whether or not to compute detailed function properties.";
  sub_C53130(&qword_502F040);
  __cxa_atexit(sub_984900, &qword_502F040, &qword_4A427C0);
  qword_502EF60 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502F040, v7, v8), 1u);
  qword_502EFB0 = 0x100000000LL;
  dword_502EF6C &= 0x8000u;
  word_502EF70 = 0;
  qword_502EF78 = 0;
  qword_502EF80 = 0;
  dword_502EF68 = v9;
  qword_502EF88 = 0;
  qword_502EF90 = 0;
  qword_502EF98 = 0;
  qword_502EFA0 = 0;
  qword_502EFA8 = (__int64)&unk_502EFB8;
  qword_502EFC0 = 0;
  qword_502EFC8 = (__int64)&unk_502EFE0;
  qword_502EFD0 = 1;
  dword_502EFD8 = 0;
  byte_502EFDC = 1;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_502EFB0;
  v12 = (unsigned int)qword_502EFB0 + 1LL;
  if ( v12 > HIDWORD(qword_502EFB0) )
  {
    sub_C8D5F0((char *)&unk_502EFB8 - 16, &unk_502EFB8, v12, 8);
    v11 = (unsigned int)qword_502EFB0;
  }
  *(_QWORD *)(qword_502EFA8 + 8 * v11) = v10;
  qword_502EFF0 = (__int64)&unk_49D9728;
  qword_502EF60 = (__int64)&unk_49DBF10;
  qword_502F000 = (__int64)&unk_49DC290;
  LODWORD(qword_502EFB0) = qword_502EFB0 + 1;
  qword_502F020 = (__int64)nullsub_24;
  qword_502EFE8 = 0;
  qword_502F018 = (__int64)sub_984050;
  qword_502EFF8 = 0;
  sub_C53080(&qword_502EF60, "big-basic-block-instruction-threshold", 37);
  LODWORD(qword_502EFE8) = 500;
  BYTE4(qword_502EFF8) = 1;
  LODWORD(qword_502EFF8) = 500;
  qword_502EF90 = 92;
  LOBYTE(dword_502EF6C) = dword_502EF6C & 0x9F | 0x20;
  qword_502EF88 = (__int64)"The minimum number of instructions a basic block should contain before being considered big.";
  sub_C53130(&qword_502EF60);
  __cxa_atexit(sub_984970, &qword_502EF60, &qword_4A427C0);
  qword_502EE80 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_502EF60, v13, v14), 1u);
  byte_502EEFC = 1;
  qword_502EED0 = 0x100000000LL;
  dword_502EE8C &= 0x8000u;
  qword_502EEC8 = (__int64)&unk_502EED8;
  qword_502EE98 = 0;
  qword_502EEA0 = 0;
  dword_502EE88 = v15;
  word_502EE90 = 0;
  qword_502EEA8 = 0;
  qword_502EEB0 = 0;
  qword_502EEB8 = 0;
  qword_502EEC0 = 0;
  qword_502EEE0 = 0;
  qword_502EEE8 = (__int64)&unk_502EF00;
  qword_502EEF0 = 1;
  dword_502EEF8 = 0;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_502EED0;
  if ( (unsigned __int64)(unsigned int)qword_502EED0 + 1 > HIDWORD(qword_502EED0) )
  {
    v25 = v16;
    sub_C8D5F0((char *)&unk_502EED8 - 16, &unk_502EED8, (unsigned int)qword_502EED0 + 1LL, 8);
    v17 = (unsigned int)qword_502EED0;
    v16 = v25;
  }
  *(_QWORD *)(qword_502EEC8 + 8 * v17) = v16;
  qword_502EF10 = (__int64)&unk_49D9728;
  qword_502EE80 = (__int64)&unk_49DBF10;
  qword_502EF20 = (__int64)&unk_49DC290;
  LODWORD(qword_502EED0) = qword_502EED0 + 1;
  qword_502EF40 = (__int64)nullsub_24;
  qword_502EF08 = 0;
  qword_502EF38 = (__int64)sub_984050;
  qword_502EF18 = 0;
  sub_C53080(&qword_502EE80, "medium-basic-block-instruction-threshold", 40);
  LODWORD(qword_502EF08) = 15;
  BYTE4(qword_502EF18) = 1;
  LODWORD(qword_502EF18) = 15;
  qword_502EEB0 = 101;
  LOBYTE(dword_502EE8C) = dword_502EE8C & 0x9F | 0x20;
  qword_502EEA8 = (__int64)"The minimum number of instructions a basic block should contain before being considered medium-sized.";
  sub_C53130(&qword_502EE80);
  __cxa_atexit(sub_984970, &qword_502EE80, &qword_4A427C0);
  qword_502EDA0 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_502EE80, v18, v19), 1u);
  dword_502EDAC &= 0x8000u;
  word_502EDB0 = 0;
  qword_502EDF0 = 0x100000000LL;
  qword_502EDE8 = (__int64)&unk_502EDF8;
  qword_502EDB8 = 0;
  qword_502EDC0 = 0;
  dword_502EDA8 = v20;
  qword_502EDC8 = 0;
  qword_502EDD0 = 0;
  qword_502EDD8 = 0;
  qword_502EDE0 = 0;
  qword_502EE00 = 0;
  qword_502EE08 = (__int64)&unk_502EE20;
  qword_502EE10 = 1;
  dword_502EE18 = 0;
  byte_502EE1C = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_502EDF0;
  v23 = (unsigned int)qword_502EDF0 + 1LL;
  if ( v23 > HIDWORD(qword_502EDF0) )
  {
    sub_C8D5F0((char *)&unk_502EDF8 - 16, &unk_502EDF8, v23, 8);
    v22 = (unsigned int)qword_502EDF0;
  }
  *(_QWORD *)(qword_502EDE8 + 8 * v22) = v21;
  qword_502EE30 = (__int64)&unk_49D9728;
  qword_502EDA0 = (__int64)&unk_49DBF10;
  qword_502EE40 = (__int64)&unk_49DC290;
  LODWORD(qword_502EDF0) = qword_502EDF0 + 1;
  qword_502EE60 = (__int64)nullsub_24;
  qword_502EE28 = 0;
  qword_502EE58 = (__int64)sub_984050;
  qword_502EE38 = 0;
  sub_C53080(&qword_502EDA0, "call-with-many-arguments-threshold", 34);
  LODWORD(qword_502EE28) = 4;
  BYTE4(qword_502EE38) = 1;
  LODWORD(qword_502EE38) = 4;
  qword_502EDD0 = 104;
  LOBYTE(dword_502EDAC) = dword_502EDAC & 0x9F | 0x20;
  qword_502EDC8 = (__int64)"The minimum number of arguments a function call must have before it is considered having many arguments.";
  sub_C53130(&qword_502EDA0);
  return __cxa_atexit(sub_984970, &qword_502EDA0, &qword_4A427C0);
}
