// Function: ctor_552
// Address: 0x56ed40
//
int ctor_552()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // edx
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]
  __int64 v29; // [rsp+8h] [rbp-38h]
  __int64 v30; // [rsp+8h] [rbp-38h]
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  qword_501E060 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501E06C &= 0x8000u;
  word_501E070 = 0;
  qword_501E0B0 = 0x100000000LL;
  qword_501E078 = 0;
  qword_501E080 = 0;
  qword_501E088 = 0;
  dword_501E068 = v0;
  qword_501E090 = 0;
  qword_501E098 = 0;
  qword_501E0A0 = 0;
  qword_501E0A8 = (__int64)&unk_501E0B8;
  qword_501E0C0 = 0;
  qword_501E0C8 = (__int64)&unk_501E0E0;
  qword_501E0D0 = 1;
  dword_501E0D8 = 0;
  byte_501E0DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501E0B0;
  v3 = (unsigned int)qword_501E0B0 + 1LL;
  if ( v3 > HIDWORD(qword_501E0B0) )
  {
    sub_C8D5F0((char *)&unk_501E0B8 - 16, &unk_501E0B8, v3, 8);
    v2 = (unsigned int)qword_501E0B0;
  }
  *(_QWORD *)(qword_501E0A8 + 8 * v2) = v1;
  qword_501E0F0 = (__int64)&unk_49D9748;
  qword_501E060 = (__int64)&unk_49DC090;
  qword_501E100 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501E0B0) = qword_501E0B0 + 1;
  qword_501E120 = (__int64)nullsub_23;
  qword_501E0E8 = 0;
  qword_501E118 = (__int64)sub_984030;
  qword_501E0F8 = 0;
  sub_C53080(&qword_501E060, "enable-global-merge", 19);
  qword_501E090 = 28;
  LOBYTE(qword_501E0E8) = 1;
  LOBYTE(dword_501E06C) = dword_501E06C & 0x9F | 0x20;
  qword_501E088 = (__int64)"Enable the global merge pass";
  LOWORD(qword_501E0F8) = 257;
  sub_C53130(&qword_501E060);
  __cxa_atexit(sub_984900, &qword_501E060, &qword_4A427C0);
  qword_501DF80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501DF8C &= 0x8000u;
  word_501DF90 = 0;
  qword_501DFD0 = 0x100000000LL;
  qword_501DFC8 = (__int64)&unk_501DFD8;
  qword_501DF98 = 0;
  qword_501DFA0 = 0;
  dword_501DF88 = v4;
  qword_501DFA8 = 0;
  qword_501DFB0 = 0;
  qword_501DFB8 = 0;
  qword_501DFC0 = 0;
  qword_501DFE0 = 0;
  qword_501DFE8 = (__int64)&unk_501E000;
  qword_501DFF0 = 1;
  dword_501DFF8 = 0;
  byte_501DFFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_501DFD0;
  if ( (unsigned __int64)(unsigned int)qword_501DFD0 + 1 > HIDWORD(qword_501DFD0) )
  {
    v28 = v5;
    sub_C8D5F0((char *)&unk_501DFD8 - 16, &unk_501DFD8, (unsigned int)qword_501DFD0 + 1LL, 8);
    v6 = (unsigned int)qword_501DFD0;
    v5 = v28;
  }
  *(_QWORD *)(qword_501DFC8 + 8 * v6) = v5;
  LODWORD(qword_501DFD0) = qword_501DFD0 + 1;
  qword_501E008 = 0;
  qword_501E010 = (__int64)&unk_49D9728;
  qword_501E018 = 0;
  qword_501DF80 = (__int64)&unk_49DBF10;
  qword_501E020 = (__int64)&unk_49DC290;
  qword_501E040 = (__int64)nullsub_24;
  qword_501E038 = (__int64)sub_984050;
  sub_C53080(&qword_501DF80, "global-merge-max-offset", 23);
  qword_501DFB0 = 40;
  LODWORD(qword_501E008) = 0;
  BYTE4(qword_501E018) = 1;
  LODWORD(qword_501E018) = 0;
  LOBYTE(dword_501DF8C) = dword_501DF8C & 0x9F | 0x20;
  qword_501DFA8 = (__int64)"Set maximum offset for global merge pass";
  sub_C53130(&qword_501DF80);
  __cxa_atexit(sub_984970, &qword_501DF80, &qword_4A427C0);
  qword_501DEA0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501DEAC &= 0x8000u;
  word_501DEB0 = 0;
  qword_501DEF0 = 0x100000000LL;
  qword_501DEE8 = (__int64)&unk_501DEF8;
  qword_501DEB8 = 0;
  qword_501DEC0 = 0;
  dword_501DEA8 = v7;
  qword_501DEC8 = 0;
  qword_501DED0 = 0;
  qword_501DED8 = 0;
  qword_501DEE0 = 0;
  qword_501DF00 = 0;
  qword_501DF08 = (__int64)&unk_501DF20;
  qword_501DF10 = 1;
  dword_501DF18 = 0;
  byte_501DF1C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_501DEF0;
  if ( (unsigned __int64)(unsigned int)qword_501DEF0 + 1 > HIDWORD(qword_501DEF0) )
  {
    v29 = v8;
    sub_C8D5F0((char *)&unk_501DEF8 - 16, &unk_501DEF8, (unsigned int)qword_501DEF0 + 1LL, 8);
    v9 = (unsigned int)qword_501DEF0;
    v8 = v29;
  }
  *(_QWORD *)(qword_501DEE8 + 8 * v9) = v8;
  qword_501DF30 = (__int64)&unk_49D9748;
  qword_501DEA0 = (__int64)&unk_49DC090;
  qword_501DF40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501DEF0) = qword_501DEF0 + 1;
  qword_501DF60 = (__int64)nullsub_23;
  qword_501DF28 = 0;
  qword_501DF58 = (__int64)sub_984030;
  qword_501DF38 = 0;
  sub_C53080(&qword_501DEA0, "global-merge-group-by-use", 25);
  LOWORD(qword_501DF38) = 257;
  LOBYTE(qword_501DF28) = 1;
  qword_501DED0 = 41;
  LOBYTE(dword_501DEAC) = dword_501DEAC & 0x9F | 0x20;
  qword_501DEC8 = (__int64)"Improve global merge pass to look at uses";
  sub_C53130(&qword_501DEA0);
  __cxa_atexit(sub_984900, &qword_501DEA0, &qword_4A427C0);
  qword_501DDC0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501DE10 = 0x100000000LL;
  dword_501DDCC &= 0x8000u;
  qword_501DE08 = (__int64)&unk_501DE18;
  word_501DDD0 = 0;
  qword_501DDD8 = 0;
  dword_501DDC8 = v10;
  qword_501DDE0 = 0;
  qword_501DDE8 = 0;
  qword_501DDF0 = 0;
  qword_501DDF8 = 0;
  qword_501DE00 = 0;
  qword_501DE20 = 0;
  qword_501DE28 = (__int64)&unk_501DE40;
  qword_501DE30 = 1;
  dword_501DE38 = 0;
  byte_501DE3C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_501DE10;
  if ( (unsigned __int64)(unsigned int)qword_501DE10 + 1 > HIDWORD(qword_501DE10) )
  {
    v30 = v11;
    sub_C8D5F0((char *)&unk_501DE18 - 16, &unk_501DE18, (unsigned int)qword_501DE10 + 1LL, 8);
    v12 = (unsigned int)qword_501DE10;
    v11 = v30;
  }
  *(_QWORD *)(qword_501DE08 + 8 * v12) = v11;
  qword_501DE50 = (__int64)&unk_49D9748;
  qword_501DDC0 = (__int64)&unk_49DC090;
  qword_501DE60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501DE10) = qword_501DE10 + 1;
  qword_501DE80 = (__int64)nullsub_23;
  qword_501DE48 = 0;
  qword_501DE78 = (__int64)sub_984030;
  qword_501DE58 = 0;
  sub_C53080(&qword_501DDC0, "global-merge-all-const", 22);
  LOWORD(qword_501DE58) = 256;
  LOBYTE(qword_501DE48) = 0;
  qword_501DDF0 = 47;
  LOBYTE(dword_501DDCC) = dword_501DDCC & 0x9F | 0x20;
  qword_501DDE8 = (__int64)"Merge all const globals without looking at uses";
  sub_C53130(&qword_501DDC0);
  __cxa_atexit(sub_984900, &qword_501DDC0, &qword_4A427C0);
  qword_501DCE0 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501DD30 = 0x100000000LL;
  dword_501DCEC &= 0x8000u;
  qword_501DD28 = (__int64)&unk_501DD38;
  word_501DCF0 = 0;
  qword_501DCF8 = 0;
  dword_501DCE8 = v13;
  qword_501DD00 = 0;
  qword_501DD08 = 0;
  qword_501DD10 = 0;
  qword_501DD18 = 0;
  qword_501DD20 = 0;
  qword_501DD40 = 0;
  qword_501DD48 = (__int64)&unk_501DD60;
  qword_501DD50 = 1;
  dword_501DD58 = 0;
  byte_501DD5C = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_501DD30;
  if ( (unsigned __int64)(unsigned int)qword_501DD30 + 1 > HIDWORD(qword_501DD30) )
  {
    v31 = v14;
    sub_C8D5F0((char *)&unk_501DD38 - 16, &unk_501DD38, (unsigned int)qword_501DD30 + 1LL, 8);
    v15 = (unsigned int)qword_501DD30;
    v14 = v31;
  }
  *(_QWORD *)(qword_501DD28 + 8 * v15) = v14;
  qword_501DD70 = (__int64)&unk_49D9748;
  qword_501DCE0 = (__int64)&unk_49DC090;
  qword_501DD80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501DD30) = qword_501DD30 + 1;
  qword_501DDA0 = (__int64)nullsub_23;
  qword_501DD68 = 0;
  qword_501DD98 = (__int64)sub_984030;
  qword_501DD78 = 0;
  sub_C53080(&qword_501DCE0, "global-merge-ignore-single-use", 30);
  LOWORD(qword_501DD78) = 257;
  LOBYTE(qword_501DD68) = 1;
  qword_501DD10 = 59;
  LOBYTE(dword_501DCEC) = dword_501DCEC & 0x9F | 0x20;
  qword_501DD08 = (__int64)"Improve global merge pass to ignore globals only used alone";
  sub_C53130(&qword_501DCE0);
  __cxa_atexit(sub_984900, &qword_501DCE0, &qword_4A427C0);
  qword_501DC00 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501DC50 = 0x100000000LL;
  dword_501DC0C &= 0x8000u;
  qword_501DC48 = (__int64)&unk_501DC58;
  word_501DC10 = 0;
  qword_501DC18 = 0;
  dword_501DC08 = v16;
  qword_501DC20 = 0;
  qword_501DC28 = 0;
  qword_501DC30 = 0;
  qword_501DC38 = 0;
  qword_501DC40 = 0;
  qword_501DC60 = 0;
  qword_501DC68 = (__int64)&unk_501DC80;
  qword_501DC70 = 1;
  dword_501DC78 = 0;
  byte_501DC7C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_501DC50;
  if ( (unsigned __int64)(unsigned int)qword_501DC50 + 1 > HIDWORD(qword_501DC50) )
  {
    v32 = v17;
    sub_C8D5F0((char *)&unk_501DC58 - 16, &unk_501DC58, (unsigned int)qword_501DC50 + 1LL, 8);
    v18 = (unsigned int)qword_501DC50;
    v17 = v32;
  }
  *(_QWORD *)(qword_501DC48 + 8 * v18) = v17;
  qword_501DC90 = (__int64)&unk_49D9748;
  qword_501DC00 = (__int64)&unk_49DC090;
  qword_501DCA0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501DC50) = qword_501DC50 + 1;
  qword_501DCC0 = (__int64)nullsub_23;
  qword_501DC88 = 0;
  qword_501DCB8 = (__int64)sub_984030;
  qword_501DC98 = 0;
  sub_C53080(&qword_501DC00, "global-merge-on-const", 21);
  LOWORD(qword_501DC98) = 256;
  LOBYTE(qword_501DC88) = 0;
  qword_501DC30 = 37;
  LOBYTE(dword_501DC0C) = dword_501DC0C & 0x9F | 0x20;
  qword_501DC28 = (__int64)"Enable global merge pass on constants";
  sub_C53130(&qword_501DC00);
  __cxa_atexit(sub_984900, &qword_501DC00, &qword_4A427C0);
  qword_501DB20 = (__int64)&unk_49DC150;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501DB70 = 0x100000000LL;
  word_501DB30 = 0;
  dword_501DB2C &= 0x8000u;
  qword_501DB38 = 0;
  qword_501DB40 = 0;
  dword_501DB28 = v19;
  qword_501DB48 = 0;
  qword_501DB50 = 0;
  qword_501DB58 = 0;
  qword_501DB60 = 0;
  qword_501DB68 = (__int64)&unk_501DB78;
  qword_501DB80 = 0;
  qword_501DB88 = (__int64)&unk_501DBA0;
  qword_501DB90 = 1;
  dword_501DB98 = 0;
  byte_501DB9C = 1;
  v20 = sub_C57470();
  v21 = (unsigned int)qword_501DB70;
  v22 = (unsigned int)qword_501DB70 + 1LL;
  if ( v22 > HIDWORD(qword_501DB70) )
  {
    sub_C8D5F0((char *)&unk_501DB78 - 16, &unk_501DB78, v22, 8);
    v21 = (unsigned int)qword_501DB70;
  }
  *(_QWORD *)(qword_501DB68 + 8 * v21) = v20;
  LODWORD(qword_501DB70) = qword_501DB70 + 1;
  qword_501DBA8 = 0;
  qword_501DBB0 = (__int64)&unk_49DC110;
  qword_501DBB8 = 0;
  qword_501DB20 = (__int64)&unk_49D97F0;
  qword_501DBC0 = (__int64)&unk_49DC200;
  qword_501DBE0 = (__int64)nullsub_26;
  qword_501DBD8 = (__int64)sub_9C26D0;
  sub_C53080(&qword_501DB20, "global-merge-on-external", 24);
  qword_501DB50 = 44;
  LOBYTE(dword_501DB2C) = dword_501DB2C & 0x9F | 0x20;
  qword_501DB48 = (__int64)"Enable global merge pass on external linkage";
  sub_C53130(&qword_501DB20);
  __cxa_atexit(sub_9C44F0, &qword_501DB20, &qword_4A427C0);
  qword_501DA40 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501DA4C &= 0x8000u;
  word_501DA50 = 0;
  qword_501DA90 = 0x100000000LL;
  qword_501DA58 = 0;
  qword_501DA60 = 0;
  qword_501DA68 = 0;
  dword_501DA48 = v23;
  qword_501DA70 = 0;
  qword_501DA78 = 0;
  qword_501DA80 = 0;
  qword_501DA88 = (__int64)&unk_501DA98;
  qword_501DAA0 = 0;
  qword_501DAA8 = (__int64)&unk_501DAC0;
  qword_501DAB0 = 1;
  dword_501DAB8 = 0;
  byte_501DABC = 1;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_501DA90;
  v26 = (unsigned int)qword_501DA90 + 1LL;
  if ( v26 > HIDWORD(qword_501DA90) )
  {
    sub_C8D5F0((char *)&unk_501DA98 - 16, &unk_501DA98, v26, 8);
    v25 = (unsigned int)qword_501DA90;
  }
  *(_QWORD *)(qword_501DA88 + 8 * v25) = v24;
  LODWORD(qword_501DA90) = qword_501DA90 + 1;
  qword_501DAC8 = 0;
  qword_501DAD0 = (__int64)&unk_49D9728;
  qword_501DAD8 = 0;
  qword_501DA40 = (__int64)&unk_49DBF10;
  qword_501DAE0 = (__int64)&unk_49DC290;
  qword_501DB00 = (__int64)nullsub_24;
  qword_501DAF8 = (__int64)sub_984050;
  sub_C53080(&qword_501DA40, "global-merge-min-data-size", 26);
  qword_501DA70 = 75;
  qword_501DA68 = (__int64)"The minimum size in bytes of each global that should considered in merging.";
  LODWORD(qword_501DAC8) = 0;
  BYTE4(qword_501DAD8) = 1;
  LODWORD(qword_501DAD8) = 0;
  LOBYTE(dword_501DA4C) = dword_501DA4C & 0x9F | 0x20;
  sub_C53130(&qword_501DA40);
  return __cxa_atexit(sub_984970, &qword_501DA40, &qword_4A427C0);
}
