// Function: ctor_423
// Address: 0x533780
//
int ctor_423()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+14h] [rbp-4Ch] BYREF
  const char *v22; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v23[8]; // [rsp+20h] [rbp-40h] BYREF

  qword_4FF1D00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF1D50 = 0x100000000LL;
  dword_4FF1D0C &= 0x8000u;
  word_4FF1D10 = 0;
  qword_4FF1D18 = 0;
  qword_4FF1D20 = 0;
  dword_4FF1D08 = v0;
  qword_4FF1D28 = 0;
  qword_4FF1D30 = 0;
  qword_4FF1D38 = 0;
  qword_4FF1D40 = 0;
  qword_4FF1D48 = (__int64)&unk_4FF1D58;
  qword_4FF1D60 = 0;
  qword_4FF1D68 = (__int64)&unk_4FF1D80;
  qword_4FF1D70 = 1;
  dword_4FF1D78 = 0;
  byte_4FF1D7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF1D50;
  v3 = (unsigned int)qword_4FF1D50 + 1LL;
  if ( v3 > HIDWORD(qword_4FF1D50) )
  {
    sub_C8D5F0((char *)&unk_4FF1D58 - 16, &unk_4FF1D58, v3, 8);
    v2 = (unsigned int)qword_4FF1D50;
  }
  *(_QWORD *)(qword_4FF1D48 + 8 * v2) = v1;
  LODWORD(qword_4FF1D50) = qword_4FF1D50 + 1;
  qword_4FF1D88 = 0;
  qword_4FF1D90 = (__int64)&unk_49D9748;
  qword_4FF1D98 = 0;
  qword_4FF1D00 = (__int64)&unk_49DC090;
  qword_4FF1DA0 = (__int64)&unk_49DC1D0;
  qword_4FF1DC0 = (__int64)nullsub_23;
  qword_4FF1DB8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF1D00, "hot-cold-static-analysis", 24);
  LOBYTE(qword_4FF1D88) = 1;
  LOWORD(qword_4FF1D98) = 257;
  LOBYTE(dword_4FF1D0C) = dword_4FF1D0C & 0x9F | 0x20;
  sub_C53130(&qword_4FF1D00);
  __cxa_atexit(sub_984900, &qword_4FF1D00, &qword_4A427C0);
  qword_4FF1C20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF1C70 = 0x100000000LL;
  dword_4FF1C2C &= 0x8000u;
  word_4FF1C30 = 0;
  qword_4FF1C38 = 0;
  qword_4FF1C40 = 0;
  dword_4FF1C28 = v4;
  qword_4FF1C48 = 0;
  qword_4FF1C50 = 0;
  qword_4FF1C58 = 0;
  qword_4FF1C60 = 0;
  qword_4FF1C68 = (__int64)&unk_4FF1C78;
  qword_4FF1C80 = 0;
  qword_4FF1C88 = (__int64)&unk_4FF1CA0;
  qword_4FF1C90 = 1;
  dword_4FF1C98 = 0;
  byte_4FF1C9C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF1C70;
  v7 = (unsigned int)qword_4FF1C70 + 1LL;
  if ( v7 > HIDWORD(qword_4FF1C70) )
  {
    sub_C8D5F0((char *)&unk_4FF1C78 - 16, &unk_4FF1C78, v7, 8);
    v6 = (unsigned int)qword_4FF1C70;
  }
  *(_QWORD *)(qword_4FF1C68 + 8 * v6) = v5;
  qword_4FF1CB0 = (__int64)&unk_49DA090;
  qword_4FF1C20 = (__int64)&unk_49DBF90;
  LODWORD(qword_4FF1C70) = qword_4FF1C70 + 1;
  qword_4FF1CA8 = 0;
  qword_4FF1CC0 = (__int64)&unk_49DC230;
  qword_4FF1CB8 = 0;
  qword_4FF1CE0 = (__int64)nullsub_58;
  qword_4FF1CD8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF1C20, "hotcoldsplit-threshold", 22);
  LODWORD(qword_4FF1CA8) = 2;
  BYTE4(qword_4FF1CB8) = 1;
  LODWORD(qword_4FF1CB8) = 2;
  qword_4FF1C50 = 65;
  LOBYTE(dword_4FF1C2C) = dword_4FF1C2C & 0x9F | 0x20;
  qword_4FF1C48 = (__int64)"Base penalty for splitting cold code (as a multiple of TCC_Basic)";
  sub_C53130(&qword_4FF1C20);
  __cxa_atexit(sub_B2B680, &qword_4FF1C20, &qword_4A427C0);
  qword_4FF1B40 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF1BBC = 1;
  word_4FF1B50 = 0;
  qword_4FF1B90 = 0x100000000LL;
  dword_4FF1B4C &= 0x8000u;
  qword_4FF1B88 = (__int64)&unk_4FF1B98;
  qword_4FF1B58 = 0;
  dword_4FF1B48 = v8;
  qword_4FF1B60 = 0;
  qword_4FF1B68 = 0;
  qword_4FF1B70 = 0;
  qword_4FF1B78 = 0;
  qword_4FF1B80 = 0;
  qword_4FF1BA0 = 0;
  qword_4FF1BA8 = (__int64)&unk_4FF1BC0;
  qword_4FF1BB0 = 1;
  dword_4FF1BB8 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FF1B90;
  if ( (unsigned __int64)(unsigned int)qword_4FF1B90 + 1 > HIDWORD(qword_4FF1B90) )
  {
    v20 = v9;
    sub_C8D5F0((char *)&unk_4FF1B98 - 16, &unk_4FF1B98, (unsigned int)qword_4FF1B90 + 1LL, 8);
    v10 = (unsigned int)qword_4FF1B90;
    v9 = v20;
  }
  *(_QWORD *)(qword_4FF1B88 + 8 * v10) = v9;
  LODWORD(qword_4FF1B90) = qword_4FF1B90 + 1;
  qword_4FF1BC8 = 0;
  qword_4FF1BD0 = (__int64)&unk_49D9748;
  qword_4FF1BD8 = 0;
  qword_4FF1B40 = (__int64)&unk_49DC090;
  qword_4FF1BE0 = (__int64)&unk_49DC1D0;
  qword_4FF1C00 = (__int64)nullsub_23;
  qword_4FF1BF8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF1B40, "enable-cold-section", 19);
  LOWORD(qword_4FF1BD8) = 256;
  LOBYTE(qword_4FF1BC8) = 0;
  qword_4FF1B70 = 94;
  LOBYTE(dword_4FF1B4C) = dword_4FF1B4C & 0x9F | 0x20;
  qword_4FF1B68 = (__int64)"Enable placement of extracted cold functions into a separate section after hot-cold splitting.";
  sub_C53130(&qword_4FF1B40);
  __cxa_atexit(sub_984900, &qword_4FF1B40, &qword_4A427C0);
  v23[1] = 79;
  v23[0] = "Name for the section containing cold functions extracted by hot-cold splitting.";
  v21 = 1;
  v22 = "__llvm_cold";
  sub_25F1470(&unk_4FF1A40, "hotcoldsplit-cold-section-name", &v22, &v21, v23);
  __cxa_atexit(sub_BC5A40, &unk_4FF1A40, &qword_4A427C0);
  qword_4FF1960 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF19B0 = 0x100000000LL;
  dword_4FF196C &= 0x8000u;
  word_4FF1970 = 0;
  qword_4FF19A8 = (__int64)&unk_4FF19B8;
  qword_4FF1978 = 0;
  dword_4FF1968 = v11;
  qword_4FF1980 = 0;
  qword_4FF1988 = 0;
  qword_4FF1990 = 0;
  qword_4FF1998 = 0;
  qword_4FF19A0 = 0;
  qword_4FF19C0 = 0;
  qword_4FF19C8 = (__int64)&unk_4FF19E0;
  qword_4FF19D0 = 1;
  dword_4FF19D8 = 0;
  byte_4FF19DC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FF19B0;
  v14 = (unsigned int)qword_4FF19B0 + 1LL;
  if ( v14 > HIDWORD(qword_4FF19B0) )
  {
    sub_C8D5F0((char *)&unk_4FF19B8 - 16, &unk_4FF19B8, v14, 8);
    v13 = (unsigned int)qword_4FF19B0;
  }
  *(_QWORD *)(qword_4FF19A8 + 8 * v13) = v12;
  qword_4FF19F0 = (__int64)&unk_49DA090;
  qword_4FF1960 = (__int64)&unk_49DBF90;
  LODWORD(qword_4FF19B0) = qword_4FF19B0 + 1;
  qword_4FF19E8 = 0;
  qword_4FF1A00 = (__int64)&unk_49DC230;
  qword_4FF19F8 = 0;
  qword_4FF1A20 = (__int64)nullsub_58;
  qword_4FF1A18 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF1960, "hotcoldsplit-max-params", 23);
  LODWORD(qword_4FF19E8) = 4;
  BYTE4(qword_4FF19F8) = 1;
  LODWORD(qword_4FF19F8) = 4;
  qword_4FF1990 = 49;
  LOBYTE(dword_4FF196C) = dword_4FF196C & 0x9F | 0x20;
  qword_4FF1988 = (__int64)"Maximum number of parameters for a split function";
  sub_C53130(&qword_4FF1960);
  __cxa_atexit(sub_B2B680, &qword_4FF1960, &qword_4A427C0);
  qword_4FF1880 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF188C &= 0x8000u;
  word_4FF1890 = 0;
  qword_4FF18D0 = 0x100000000LL;
  qword_4FF1898 = 0;
  qword_4FF18A0 = 0;
  qword_4FF18A8 = 0;
  dword_4FF1888 = v15;
  qword_4FF18B0 = 0;
  qword_4FF18B8 = 0;
  qword_4FF18C0 = 0;
  qword_4FF18C8 = (__int64)&unk_4FF18D8;
  qword_4FF18E0 = 0;
  qword_4FF18E8 = (__int64)&unk_4FF1900;
  qword_4FF18F0 = 1;
  dword_4FF18F8 = 0;
  byte_4FF18FC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_4FF18D0;
  v18 = (unsigned int)qword_4FF18D0 + 1LL;
  if ( v18 > HIDWORD(qword_4FF18D0) )
  {
    sub_C8D5F0((char *)&unk_4FF18D8 - 16, &unk_4FF18D8, v18, 8);
    v17 = (unsigned int)qword_4FF18D0;
  }
  *(_QWORD *)(qword_4FF18C8 + 8 * v17) = v16;
  qword_4FF1910 = (__int64)&unk_49DA090;
  qword_4FF1880 = (__int64)&unk_49DBF90;
  LODWORD(qword_4FF18D0) = qword_4FF18D0 + 1;
  qword_4FF1908 = 0;
  qword_4FF1920 = (__int64)&unk_49DC230;
  qword_4FF1918 = 0;
  qword_4FF1940 = (__int64)nullsub_58;
  qword_4FF1938 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FF1880, "hotcoldsplit-cold-probability-denom", 35);
  LODWORD(qword_4FF1908) = 100;
  BYTE4(qword_4FF1918) = 1;
  LODWORD(qword_4FF1918) = 100;
  qword_4FF18B0 = 76;
  LOBYTE(dword_4FF188C) = dword_4FF188C & 0x9F | 0x20;
  qword_4FF18A8 = (__int64)"Divisor of cold branch probability.BranchProbability = 1/ColdBranchProbDenom";
  sub_C53130(&qword_4FF1880);
  return __cxa_atexit(sub_B2B680, &qword_4FF1880, &qword_4A427C0);
}
