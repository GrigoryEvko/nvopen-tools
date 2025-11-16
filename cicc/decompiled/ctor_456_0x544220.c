// Function: ctor_456
// Address: 0x544220
//
int ctor_456()
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
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_4FFDC80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFDCD0 = 0x100000000LL;
  dword_4FFDC8C &= 0x8000u;
  word_4FFDC90 = 0;
  qword_4FFDC98 = 0;
  qword_4FFDCA0 = 0;
  dword_4FFDC88 = v0;
  qword_4FFDCA8 = 0;
  qword_4FFDCB0 = 0;
  qword_4FFDCB8 = 0;
  qword_4FFDCC0 = 0;
  qword_4FFDCC8 = (__int64)&unk_4FFDCD8;
  qword_4FFDCE0 = 0;
  qword_4FFDCE8 = (__int64)&unk_4FFDD00;
  qword_4FFDCF0 = 1;
  dword_4FFDCF8 = 0;
  byte_4FFDCFC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFDCD0;
  v3 = (unsigned int)qword_4FFDCD0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFDCD0) )
  {
    sub_C8D5F0((char *)&unk_4FFDCD8 - 16, &unk_4FFDCD8, v3, 8);
    v2 = (unsigned int)qword_4FFDCD0;
  }
  *(_QWORD *)(qword_4FFDCC8 + 8 * v2) = v1;
  LODWORD(qword_4FFDCD0) = qword_4FFDCD0 + 1;
  qword_4FFDD08 = 0;
  qword_4FFDD10 = (__int64)&unk_49D9748;
  qword_4FFDD18 = 0;
  qword_4FFDC80 = (__int64)&unk_49DC090;
  qword_4FFDD20 = (__int64)&unk_49DC1D0;
  qword_4FFDD40 = (__int64)nullsub_23;
  qword_4FFDD38 = (__int64)sub_984030;
  sub_C53080(&qword_4FFDC80, "jump-threading-disable-select-unfolding", 39);
  qword_4FFDCB0 = 41;
  qword_4FFDCA8 = (__int64)"Disables unfolding of select instructions";
  LOWORD(qword_4FFDD18) = 257;
  LOBYTE(qword_4FFDD08) = 1;
  LOBYTE(dword_4FFDC8C) = dword_4FFDC8C & 0x9F | 0x20;
  sub_C53130(&qword_4FFDC80);
  __cxa_atexit(sub_984900, &qword_4FFDC80, &qword_4A427C0);
  qword_4FFDBA0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFDBF0 = 0x100000000LL;
  dword_4FFDBAC &= 0x8000u;
  word_4FFDBB0 = 0;
  qword_4FFDBB8 = 0;
  qword_4FFDBC0 = 0;
  dword_4FFDBA8 = v4;
  qword_4FFDBC8 = 0;
  qword_4FFDBD0 = 0;
  qword_4FFDBD8 = 0;
  qword_4FFDBE0 = 0;
  qword_4FFDBE8 = (__int64)&unk_4FFDBF8;
  qword_4FFDC00 = 0;
  qword_4FFDC08 = (__int64)&unk_4FFDC20;
  qword_4FFDC10 = 1;
  dword_4FFDC18 = 0;
  byte_4FFDC1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFDBF0;
  v7 = (unsigned int)qword_4FFDBF0 + 1LL;
  if ( v7 > HIDWORD(qword_4FFDBF0) )
  {
    sub_C8D5F0((char *)&unk_4FFDBF8 - 16, &unk_4FFDBF8, v7, 8);
    v6 = (unsigned int)qword_4FFDBF0;
  }
  *(_QWORD *)(qword_4FFDBE8 + 8 * v6) = v5;
  qword_4FFDC30 = (__int64)&unk_49D9728;
  qword_4FFDBA0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFDBF0) = qword_4FFDBF0 + 1;
  qword_4FFDC28 = 0;
  qword_4FFDC40 = (__int64)&unk_49DC290;
  qword_4FFDC38 = 0;
  qword_4FFDC60 = (__int64)nullsub_24;
  qword_4FFDC58 = (__int64)sub_984050;
  sub_C53080(&qword_4FFDBA0, "jump-threading-threshold", 24);
  qword_4FFDBD0 = 46;
  qword_4FFDBC8 = (__int64)"Max block size to duplicate for jump threading";
  LODWORD(qword_4FFDC28) = 6;
  BYTE4(qword_4FFDC38) = 1;
  LODWORD(qword_4FFDC38) = 6;
  LOBYTE(dword_4FFDBAC) = dword_4FFDBAC & 0x9F | 0x20;
  sub_C53130(&qword_4FFDBA0);
  __cxa_atexit(sub_984970, &qword_4FFDBA0, &qword_4A427C0);
  qword_4FFDAC0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFDB3C = 1;
  word_4FFDAD0 = 0;
  qword_4FFDB10 = 0x100000000LL;
  dword_4FFDACC &= 0x8000u;
  qword_4FFDB08 = (__int64)&unk_4FFDB18;
  qword_4FFDAD8 = 0;
  dword_4FFDAC8 = v8;
  qword_4FFDAE0 = 0;
  qword_4FFDAE8 = 0;
  qword_4FFDAF0 = 0;
  qword_4FFDAF8 = 0;
  qword_4FFDB00 = 0;
  qword_4FFDB20 = 0;
  qword_4FFDB28 = (__int64)&unk_4FFDB40;
  qword_4FFDB30 = 1;
  dword_4FFDB38 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FFDB10;
  if ( (unsigned __int64)(unsigned int)qword_4FFDB10 + 1 > HIDWORD(qword_4FFDB10) )
  {
    v19 = v9;
    sub_C8D5F0((char *)&unk_4FFDB18 - 16, &unk_4FFDB18, (unsigned int)qword_4FFDB10 + 1LL, 8);
    v10 = (unsigned int)qword_4FFDB10;
    v9 = v19;
  }
  *(_QWORD *)(qword_4FFDB08 + 8 * v10) = v9;
  qword_4FFDB50 = (__int64)&unk_49D9728;
  qword_4FFDAC0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFDB10) = qword_4FFDB10 + 1;
  qword_4FFDB48 = 0;
  qword_4FFDB60 = (__int64)&unk_49DC290;
  qword_4FFDB58 = 0;
  qword_4FFDB80 = (__int64)nullsub_24;
  qword_4FFDB78 = (__int64)sub_984050;
  sub_C53080(&qword_4FFDAC0, "jump-threading-implication-search-threshold", 43);
  qword_4FFDAF0 = 102;
  qword_4FFDAE8 = (__int64)"The number of predecessors to search for a stronger condition to use to thread over a weaker condition";
  LODWORD(qword_4FFDB48) = 3;
  BYTE4(qword_4FFDB58) = 1;
  LODWORD(qword_4FFDB58) = 3;
  LOBYTE(dword_4FFDACC) = dword_4FFDACC & 0x9F | 0x20;
  sub_C53130(&qword_4FFDAC0);
  __cxa_atexit(sub_984970, &qword_4FFDAC0, &qword_4A427C0);
  qword_4FFD9E0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFDA30 = 0x100000000LL;
  dword_4FFD9EC &= 0x8000u;
  word_4FFD9F0 = 0;
  qword_4FFDA28 = (__int64)&unk_4FFDA38;
  qword_4FFD9F8 = 0;
  dword_4FFD9E8 = v11;
  qword_4FFDA00 = 0;
  qword_4FFDA08 = 0;
  qword_4FFDA10 = 0;
  qword_4FFDA18 = 0;
  qword_4FFDA20 = 0;
  qword_4FFDA40 = 0;
  qword_4FFDA48 = (__int64)&unk_4FFDA60;
  qword_4FFDA50 = 1;
  dword_4FFDA58 = 0;
  byte_4FFDA5C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FFDA30;
  if ( (unsigned __int64)(unsigned int)qword_4FFDA30 + 1 > HIDWORD(qword_4FFDA30) )
  {
    v20 = v12;
    sub_C8D5F0((char *)&unk_4FFDA38 - 16, &unk_4FFDA38, (unsigned int)qword_4FFDA30 + 1LL, 8);
    v13 = (unsigned int)qword_4FFDA30;
    v12 = v20;
  }
  *(_QWORD *)(qword_4FFDA28 + 8 * v13) = v12;
  qword_4FFDA70 = (__int64)&unk_49D9728;
  qword_4FFD9E0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FFDA30) = qword_4FFDA30 + 1;
  qword_4FFDA68 = 0;
  qword_4FFDA80 = (__int64)&unk_49DC290;
  qword_4FFDA78 = 0;
  qword_4FFDAA0 = (__int64)nullsub_24;
  qword_4FFDA98 = (__int64)sub_984050;
  sub_C53080(&qword_4FFD9E0, "jump-threading-phi-threshold", 28);
  qword_4FFDA10 = 46;
  qword_4FFDA08 = (__int64)"Max PHIs in BB to duplicate for jump threading";
  LODWORD(qword_4FFDA68) = 76;
  BYTE4(qword_4FFDA78) = 1;
  LODWORD(qword_4FFDA78) = 76;
  LOBYTE(dword_4FFD9EC) = dword_4FFD9EC & 0x9F | 0x20;
  sub_C53130(&qword_4FFD9E0);
  __cxa_atexit(sub_984970, &qword_4FFD9E0, &qword_4A427C0);
  qword_4FFD900 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFD97C = 1;
  qword_4FFD950 = 0x100000000LL;
  dword_4FFD90C &= 0x8000u;
  qword_4FFD918 = 0;
  qword_4FFD920 = 0;
  qword_4FFD928 = 0;
  dword_4FFD908 = v14;
  word_4FFD910 = 0;
  qword_4FFD930 = 0;
  qword_4FFD938 = 0;
  qword_4FFD940 = 0;
  qword_4FFD948 = (__int64)&unk_4FFD958;
  qword_4FFD960 = 0;
  qword_4FFD968 = (__int64)&unk_4FFD980;
  qword_4FFD970 = 1;
  dword_4FFD978 = 0;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FFD950;
  v17 = (unsigned int)qword_4FFD950 + 1LL;
  if ( v17 > HIDWORD(qword_4FFD950) )
  {
    sub_C8D5F0((char *)&unk_4FFD958 - 16, &unk_4FFD958, v17, 8);
    v16 = (unsigned int)qword_4FFD950;
  }
  *(_QWORD *)(qword_4FFD948 + 8 * v16) = v15;
  LODWORD(qword_4FFD950) = qword_4FFD950 + 1;
  qword_4FFD988 = 0;
  qword_4FFD990 = (__int64)&unk_49D9748;
  qword_4FFD998 = 0;
  qword_4FFD900 = (__int64)&unk_49DC090;
  qword_4FFD9A0 = (__int64)&unk_49DC1D0;
  qword_4FFD9C0 = (__int64)nullsub_23;
  qword_4FFD9B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFD900, "jump-threading-across-loop-headers", 34);
  qword_4FFD930 = 62;
  qword_4FFD928 = (__int64)"Allow JumpThreading to thread across loop headers, for testing";
  LOWORD(qword_4FFD998) = 256;
  LOBYTE(qword_4FFD988) = 0;
  LOBYTE(dword_4FFD90C) = dword_4FFD90C & 0x9F | 0x20;
  sub_C53130(&qword_4FFD900);
  return __cxa_atexit(sub_984900, &qword_4FFD900, &qword_4A427C0);
}
