// Function: ctor_675
// Address: 0x5a2820
//
int __fastcall ctor_675(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  int v30; // edx
  __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]

  qword_503DC00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503DC50 = 0x100000000LL;
  word_503DC10 = 0;
  dword_503DC0C &= 0x8000u;
  qword_503DC18 = 0;
  qword_503DC20 = 0;
  dword_503DC08 = v4;
  qword_503DC28 = 0;
  qword_503DC30 = 0;
  qword_503DC38 = 0;
  qword_503DC40 = 0;
  qword_503DC48 = (__int64)&unk_503DC58;
  qword_503DC60 = 0;
  qword_503DC68 = (__int64)&unk_503DC80;
  qword_503DC70 = 1;
  dword_503DC78 = 0;
  byte_503DC7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503DC50;
  v7 = (unsigned int)qword_503DC50 + 1LL;
  if ( v7 > HIDWORD(qword_503DC50) )
  {
    sub_C8D5F0((char *)&unk_503DC58 - 16, &unk_503DC58, v7, 8);
    v6 = (unsigned int)qword_503DC50;
  }
  *(_QWORD *)(qword_503DC48 + 8 * v6) = v5;
  qword_503DC90 = (__int64)&unk_49D9748;
  qword_503DC00 = (__int64)&unk_49DC090;
  qword_503DCA0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_503DC50) = qword_503DC50 + 1;
  qword_503DCC0 = (__int64)nullsub_23;
  qword_503DC88 = 0;
  qword_503DCB8 = (__int64)sub_984030;
  qword_503DC98 = 0;
  sub_C53080(&qword_503DC00, "enable-linkonceodr-outlining", 28);
  LOWORD(qword_503DC98) = 256;
  LOBYTE(qword_503DC88) = 0;
  qword_503DC30 = 52;
  LOBYTE(dword_503DC0C) = dword_503DC0C & 0x9F | 0x20;
  qword_503DC28 = (__int64)"Enable the machine outliner on linkonceodr functions";
  sub_C53130(&qword_503DC00);
  __cxa_atexit(sub_984900, &qword_503DC00, &qword_4A427C0);
  qword_503DB20 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503DC00, v8, v9), 1u);
  qword_503DB70 = 0x100000000LL;
  dword_503DB2C &= 0x8000u;
  qword_503DB68 = (__int64)&unk_503DB78;
  word_503DB30 = 0;
  qword_503DB38 = 0;
  dword_503DB28 = v10;
  qword_503DB40 = 0;
  qword_503DB48 = 0;
  qword_503DB50 = 0;
  qword_503DB58 = 0;
  qword_503DB60 = 0;
  qword_503DB80 = 0;
  qword_503DB88 = (__int64)&unk_503DBA0;
  qword_503DB90 = 1;
  dword_503DB98 = 0;
  byte_503DB9C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503DB70;
  if ( (unsigned __int64)(unsigned int)qword_503DB70 + 1 > HIDWORD(qword_503DB70) )
  {
    v35 = v11;
    sub_C8D5F0((char *)&unk_503DB78 - 16, &unk_503DB78, (unsigned int)qword_503DB70 + 1LL, 8);
    v12 = (unsigned int)qword_503DB70;
    v11 = v35;
  }
  *(_QWORD *)(qword_503DB68 + 8 * v12) = v11;
  LODWORD(qword_503DB70) = qword_503DB70 + 1;
  qword_503DBA8 = 0;
  qword_503DBB0 = (__int64)&unk_49D9728;
  qword_503DBB8 = 0;
  qword_503DB20 = (__int64)&unk_49DBF10;
  qword_503DBC0 = (__int64)&unk_49DC290;
  qword_503DBE0 = (__int64)nullsub_24;
  qword_503DBD8 = (__int64)sub_984050;
  sub_C53080(&qword_503DB20, "machine-outliner-reruns", 23);
  LODWORD(qword_503DBA8) = 0;
  BYTE4(qword_503DBB8) = 1;
  LODWORD(qword_503DBB8) = 0;
  qword_503DB50 = 63;
  LOBYTE(dword_503DB2C) = dword_503DB2C & 0x9F | 0x20;
  qword_503DB48 = (__int64)"Number of times to rerun the outliner after the initial outline";
  sub_C53130(&qword_503DB20);
  __cxa_atexit(sub_984970, &qword_503DB20, &qword_4A427C0);
  qword_503DA40 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503DB20, v13, v14), 1u);
  qword_503DA90 = 0x100000000LL;
  dword_503DA4C &= 0x8000u;
  qword_503DA88 = (__int64)&unk_503DA98;
  word_503DA50 = 0;
  qword_503DA58 = 0;
  dword_503DA48 = v15;
  qword_503DA60 = 0;
  qword_503DA68 = 0;
  qword_503DA70 = 0;
  qword_503DA78 = 0;
  qword_503DA80 = 0;
  qword_503DAA0 = 0;
  qword_503DAA8 = (__int64)&unk_503DAC0;
  qword_503DAB0 = 1;
  dword_503DAB8 = 0;
  byte_503DABC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_503DA90;
  if ( (unsigned __int64)(unsigned int)qword_503DA90 + 1 > HIDWORD(qword_503DA90) )
  {
    v36 = v16;
    sub_C8D5F0((char *)&unk_503DA98 - 16, &unk_503DA98, (unsigned int)qword_503DA90 + 1LL, 8);
    v17 = (unsigned int)qword_503DA90;
    v16 = v36;
  }
  *(_QWORD *)(qword_503DA88 + 8 * v17) = v16;
  LODWORD(qword_503DA90) = qword_503DA90 + 1;
  qword_503DAC8 = 0;
  qword_503DAD0 = (__int64)&unk_49D9728;
  qword_503DAD8 = 0;
  qword_503DA40 = (__int64)&unk_49DBF10;
  qword_503DAE0 = (__int64)&unk_49DC290;
  qword_503DB00 = (__int64)nullsub_24;
  qword_503DAF8 = (__int64)sub_984050;
  sub_C53080(&qword_503DA40, "outliner-benefit-threshold", 26);
  LODWORD(qword_503DAC8) = 1;
  BYTE4(qword_503DAD8) = 1;
  LODWORD(qword_503DAD8) = 1;
  qword_503DA70 = 67;
  LOBYTE(dword_503DA4C) = dword_503DA4C & 0x9F | 0x20;
  qword_503DA68 = (__int64)"The minimum size in bytes before an outlining candidate is accepted";
  sub_C53130(&qword_503DA40);
  __cxa_atexit(sub_984970, &qword_503DA40, &qword_4A427C0);
  qword_503D960 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503DA40, v18, v19), 1u);
  qword_503D9B0 = 0x100000000LL;
  dword_503D96C &= 0x8000u;
  word_503D970 = 0;
  qword_503D9A8 = (__int64)&unk_503D9B8;
  qword_503D978 = 0;
  dword_503D968 = v20;
  qword_503D980 = 0;
  qword_503D988 = 0;
  qword_503D990 = 0;
  qword_503D998 = 0;
  qword_503D9A0 = 0;
  qword_503D9C0 = 0;
  qword_503D9C8 = (__int64)&unk_503D9E0;
  qword_503D9D0 = 1;
  dword_503D9D8 = 0;
  byte_503D9DC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_503D9B0;
  if ( (unsigned __int64)(unsigned int)qword_503D9B0 + 1 > HIDWORD(qword_503D9B0) )
  {
    v37 = v21;
    sub_C8D5F0((char *)&unk_503D9B8 - 16, &unk_503D9B8, (unsigned int)qword_503D9B0 + 1LL, 8);
    v22 = (unsigned int)qword_503D9B0;
    v21 = v37;
  }
  *(_QWORD *)(qword_503D9A8 + 8 * v22) = v21;
  qword_503D9F0 = (__int64)&unk_49D9748;
  qword_503D960 = (__int64)&unk_49DC090;
  qword_503DA00 = (__int64)&unk_49DC1D0;
  LODWORD(qword_503D9B0) = qword_503D9B0 + 1;
  qword_503DA20 = (__int64)nullsub_23;
  qword_503D9E8 = 0;
  qword_503DA18 = (__int64)sub_984030;
  qword_503D9F8 = 0;
  sub_C53080(&qword_503D960, "outliner-leaf-descendants", 25);
  LOWORD(qword_503D9F8) = 257;
  LOBYTE(qword_503D9E8) = 1;
  qword_503D990 = 140;
  LOBYTE(dword_503D96C) = dword_503D96C & 0x9F | 0x20;
  qword_503D988 = (__int64)"Consider all leaf descendants of internal nodes of the suffix tree as candidates for outlinin"
                           "g (if false, only leaf children are considered)";
  sub_C53130(&qword_503D960);
  __cxa_atexit(sub_984900, &qword_503D960, &qword_4A427C0);
  qword_503D880 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503D960, v23, v24), 1u);
  qword_503D8D0 = 0x100000000LL;
  dword_503D88C &= 0x8000u;
  qword_503D8C8 = (__int64)&unk_503D8D8;
  word_503D890 = 0;
  qword_503D898 = 0;
  dword_503D888 = v25;
  qword_503D8A0 = 0;
  qword_503D8A8 = 0;
  qword_503D8B0 = 0;
  qword_503D8B8 = 0;
  qword_503D8C0 = 0;
  qword_503D8E0 = 0;
  qword_503D8E8 = (__int64)&unk_503D900;
  qword_503D8F0 = 1;
  dword_503D8F8 = 0;
  byte_503D8FC = 1;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_503D8D0;
  if ( (unsigned __int64)(unsigned int)qword_503D8D0 + 1 > HIDWORD(qword_503D8D0) )
  {
    v38 = v26;
    sub_C8D5F0((char *)&unk_503D8D8 - 16, &unk_503D8D8, (unsigned int)qword_503D8D0 + 1LL, 8);
    v27 = (unsigned int)qword_503D8D0;
    v26 = v38;
  }
  *(_QWORD *)(qword_503D8C8 + 8 * v27) = v26;
  qword_503D910 = (__int64)&unk_49D9748;
  qword_503D880 = (__int64)&unk_49DC090;
  qword_503D920 = (__int64)&unk_49DC1D0;
  LODWORD(qword_503D8D0) = qword_503D8D0 + 1;
  qword_503D940 = (__int64)nullsub_23;
  qword_503D908 = 0;
  qword_503D938 = (__int64)sub_984030;
  qword_503D918 = 0;
  sub_C53080(&qword_503D880, "disable-global-outlining", 24);
  LOWORD(qword_503D918) = 256;
  LOBYTE(qword_503D908) = 0;
  qword_503D8B0 = 76;
  LOBYTE(dword_503D88C) = dword_503D88C & 0x9F | 0x20;
  qword_503D8A8 = (__int64)"Disable global outlining only by ignoring the codegen data generation or use";
  sub_C53130(&qword_503D880);
  __cxa_atexit(sub_984900, &qword_503D880, &qword_4A427C0);
  qword_503D7A0 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503D880, v28, v29), 1u);
  qword_503D7F0 = 0x100000000LL;
  dword_503D7AC &= 0x8000u;
  word_503D7B0 = 0;
  qword_503D7E8 = (__int64)&unk_503D7F8;
  qword_503D7B8 = 0;
  dword_503D7A8 = v30;
  qword_503D7C0 = 0;
  qword_503D7C8 = 0;
  qword_503D7D0 = 0;
  qword_503D7D8 = 0;
  qword_503D7E0 = 0;
  qword_503D800 = 0;
  qword_503D808 = (__int64)&unk_503D820;
  qword_503D810 = 1;
  dword_503D818 = 0;
  byte_503D81C = 1;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_503D7F0;
  v33 = (unsigned int)qword_503D7F0 + 1LL;
  if ( v33 > HIDWORD(qword_503D7F0) )
  {
    sub_C8D5F0((char *)&unk_503D7F8 - 16, &unk_503D7F8, v33, 8);
    v32 = (unsigned int)qword_503D7F0;
  }
  *(_QWORD *)(qword_503D7E8 + 8 * v32) = v31;
  qword_503D830 = (__int64)&unk_49D9748;
  qword_503D7A0 = (__int64)&unk_49DC090;
  qword_503D840 = (__int64)&unk_49DC1D0;
  LODWORD(qword_503D7F0) = qword_503D7F0 + 1;
  qword_503D860 = (__int64)nullsub_23;
  qword_503D828 = 0;
  qword_503D858 = (__int64)sub_984030;
  qword_503D838 = 0;
  sub_C53080(&qword_503D7A0, "append-content-hash-outlined-name", 33);
  qword_503D7D0 = 173;
  LOBYTE(qword_503D828) = 1;
  LOBYTE(dword_503D7AC) = dword_503D7AC & 0x9F | 0x20;
  qword_503D7C8 = (__int64)"This appends the content hash to the globally outlined function name. It's beneficial for enh"
                           "ancing the precision of the stable hash and for ordering the outlined functions.";
  LOWORD(qword_503D838) = 257;
  sub_C53130(&qword_503D7A0);
  return __cxa_atexit(sub_984900, &qword_503D7A0, &qword_4A427C0);
}
