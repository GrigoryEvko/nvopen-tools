// Function: ctor_617
// Address: 0x58a3f0
//
int __fastcall ctor_617(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
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
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // edx
  __int64 v41; // r15
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+8h] [rbp-48h]
  __int64 v50; // [rsp+8h] [rbp-48h]
  _DWORD v51[13]; // [rsp+1Ch] [rbp-34h] BYREF

  qword_502E000 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_502E050 = 0x100000000LL;
  dword_502E00C &= 0x8000u;
  word_502E010 = 0;
  qword_502E018 = 0;
  qword_502E020 = 0;
  dword_502E008 = v4;
  qword_502E028 = 0;
  qword_502E030 = 0;
  qword_502E038 = 0;
  qword_502E040 = 0;
  qword_502E048 = (__int64)&unk_502E058;
  qword_502E060 = 0;
  qword_502E068 = (__int64)&unk_502E080;
  qword_502E070 = 1;
  dword_502E078 = 0;
  byte_502E07C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502E050;
  v7 = (unsigned int)qword_502E050 + 1LL;
  if ( v7 > HIDWORD(qword_502E050) )
  {
    sub_C8D5F0((char *)&unk_502E058 - 16, &unk_502E058, v7, 8);
    v6 = (unsigned int)qword_502E050;
  }
  *(_QWORD *)(qword_502E048 + 8 * v6) = v5;
  qword_502E090 = (__int64)&unk_49D9748;
  qword_502E000 = (__int64)&unk_49DC090;
  qword_502E0A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502E050) = qword_502E050 + 1;
  qword_502E0C0 = (__int64)nullsub_23;
  qword_502E088 = 0;
  qword_502E0B8 = (__int64)sub_984030;
  qword_502E098 = 0;
  sub_C53080(&qword_502E000, "print-all-alias-modref-info", 27);
  LOBYTE(dword_502E00C) = dword_502E00C & 0x9F | 0x40;
  sub_C53130(&qword_502E000);
  __cxa_atexit(sub_984900, &qword_502E000, &qword_4A427C0);
  qword_502DF20 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502E000, v8, v9), 1u);
  qword_502DF70 = 0x100000000LL;
  dword_502DF2C &= 0x8000u;
  qword_502DF68 = (__int64)&unk_502DF78;
  word_502DF30 = 0;
  qword_502DF38 = 0;
  dword_502DF28 = v10;
  qword_502DF40 = 0;
  qword_502DF48 = 0;
  qword_502DF50 = 0;
  qword_502DF58 = 0;
  qword_502DF60 = 0;
  qword_502DF80 = 0;
  qword_502DF88 = (__int64)&unk_502DFA0;
  qword_502DF90 = 1;
  dword_502DF98 = 0;
  byte_502DF9C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502DF70;
  if ( (unsigned __int64)(unsigned int)qword_502DF70 + 1 > HIDWORD(qword_502DF70) )
  {
    v45 = v11;
    sub_C8D5F0((char *)&unk_502DF78 - 16, &unk_502DF78, (unsigned int)qword_502DF70 + 1LL, 8);
    v12 = (unsigned int)qword_502DF70;
    v11 = v45;
  }
  *(_QWORD *)(qword_502DF68 + 8 * v12) = v11;
  qword_502DFB0 = (__int64)&unk_49D9748;
  qword_502DF20 = (__int64)&unk_49DC090;
  qword_502DFC0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502DF70) = qword_502DF70 + 1;
  qword_502DFE0 = (__int64)nullsub_23;
  qword_502DFA8 = 0;
  qword_502DFD8 = (__int64)sub_984030;
  qword_502DFB8 = 0;
  sub_C53080(&qword_502DF20, "print-no-aliases", 16);
  LOBYTE(dword_502DF2C) = dword_502DF2C & 0x9F | 0x40;
  sub_C53130(&qword_502DF20);
  __cxa_atexit(sub_984900, &qword_502DF20, &qword_4A427C0);
  qword_502DE40 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502DF20, v13, v14), 1u);
  qword_502DE90 = 0x100000000LL;
  dword_502DE4C &= 0x8000u;
  word_502DE50 = 0;
  qword_502DE88 = (__int64)&unk_502DE98;
  qword_502DE58 = 0;
  dword_502DE48 = v15;
  qword_502DE60 = 0;
  qword_502DE68 = 0;
  qword_502DE70 = 0;
  qword_502DE78 = 0;
  qword_502DE80 = 0;
  qword_502DEA0 = 0;
  qword_502DEA8 = (__int64)&unk_502DEC0;
  qword_502DEB0 = 1;
  dword_502DEB8 = 0;
  byte_502DEBC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_502DE90;
  if ( (unsigned __int64)(unsigned int)qword_502DE90 + 1 > HIDWORD(qword_502DE90) )
  {
    v46 = v16;
    sub_C8D5F0((char *)&unk_502DE98 - 16, &unk_502DE98, (unsigned int)qword_502DE90 + 1LL, 8);
    v17 = (unsigned int)qword_502DE90;
    v16 = v46;
  }
  *(_QWORD *)(qword_502DE88 + 8 * v17) = v16;
  qword_502DED0 = (__int64)&unk_49D9748;
  qword_502DE40 = (__int64)&unk_49DC090;
  qword_502DEE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502DE90) = qword_502DE90 + 1;
  qword_502DF00 = (__int64)nullsub_23;
  qword_502DEC8 = 0;
  qword_502DEF8 = (__int64)sub_984030;
  qword_502DED8 = 0;
  sub_C53080(&qword_502DE40, "print-may-aliases", 17);
  LOBYTE(dword_502DE4C) = dword_502DE4C & 0x9F | 0x40;
  sub_C53130(&qword_502DE40);
  __cxa_atexit(sub_984900, &qword_502DE40, &qword_4A427C0);
  qword_502DD60 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502DE40, v18, v19), 1u);
  qword_502DDB0 = 0x100000000LL;
  dword_502DD6C &= 0x8000u;
  qword_502DDA8 = (__int64)&unk_502DDB8;
  word_502DD70 = 0;
  qword_502DD78 = 0;
  dword_502DD68 = v20;
  qword_502DD80 = 0;
  qword_502DD88 = 0;
  qword_502DD90 = 0;
  qword_502DD98 = 0;
  qword_502DDA0 = 0;
  qword_502DDC0 = 0;
  qword_502DDC8 = (__int64)&unk_502DDE0;
  qword_502DDD0 = 1;
  dword_502DDD8 = 0;
  byte_502DDDC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_502DDB0;
  if ( (unsigned __int64)(unsigned int)qword_502DDB0 + 1 > HIDWORD(qword_502DDB0) )
  {
    v47 = v21;
    sub_C8D5F0((char *)&unk_502DDB8 - 16, &unk_502DDB8, (unsigned int)qword_502DDB0 + 1LL, 8);
    v22 = (unsigned int)qword_502DDB0;
    v21 = v47;
  }
  *(_QWORD *)(qword_502DDA8 + 8 * v22) = v21;
  qword_502DDF0 = (__int64)&unk_49D9748;
  qword_502DD60 = (__int64)&unk_49DC090;
  qword_502DE00 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502DDB0) = qword_502DDB0 + 1;
  qword_502DE20 = (__int64)nullsub_23;
  qword_502DDE8 = 0;
  qword_502DE18 = (__int64)sub_984030;
  qword_502DDF8 = 0;
  sub_C53080(&qword_502DD60, "print-partial-aliases", 21);
  LOBYTE(dword_502DD6C) = dword_502DD6C & 0x9F | 0x40;
  sub_C53130(&qword_502DD60);
  __cxa_atexit(sub_984900, &qword_502DD60, &qword_4A427C0);
  qword_502DC80 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502DD60, v23, v24), 1u);
  byte_502DCFC = 1;
  word_502DC90 = 0;
  qword_502DCD0 = 0x100000000LL;
  dword_502DC8C &= 0x8000u;
  qword_502DCC8 = (__int64)&unk_502DCD8;
  qword_502DC98 = 0;
  dword_502DC88 = v25;
  qword_502DCA0 = 0;
  qword_502DCA8 = 0;
  qword_502DCB0 = 0;
  qword_502DCB8 = 0;
  qword_502DCC0 = 0;
  qword_502DCE0 = 0;
  qword_502DCE8 = (__int64)&unk_502DD00;
  qword_502DCF0 = 1;
  dword_502DCF8 = 0;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_502DCD0;
  if ( (unsigned __int64)(unsigned int)qword_502DCD0 + 1 > HIDWORD(qword_502DCD0) )
  {
    v48 = v26;
    sub_C8D5F0((char *)&unk_502DCD8 - 16, &unk_502DCD8, (unsigned int)qword_502DCD0 + 1LL, 8);
    v27 = (unsigned int)qword_502DCD0;
    v26 = v48;
  }
  *(_QWORD *)(qword_502DCC8 + 8 * v27) = v26;
  qword_502DD10 = (__int64)&unk_49D9748;
  qword_502DC80 = (__int64)&unk_49DC090;
  qword_502DD20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502DCD0) = qword_502DCD0 + 1;
  qword_502DD40 = (__int64)nullsub_23;
  qword_502DD08 = 0;
  qword_502DD38 = (__int64)sub_984030;
  qword_502DD18 = 0;
  sub_C53080(&qword_502DC80, "print-must-aliases", 18);
  LOBYTE(dword_502DC8C) = dword_502DC8C & 0x9F | 0x40;
  sub_C53130(&qword_502DC80);
  __cxa_atexit(sub_984900, &qword_502DC80, &qword_4A427C0);
  qword_502DBA0 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502DC80, v28, v29), 1u);
  qword_502DBF0 = 0x100000000LL;
  dword_502DBAC &= 0x8000u;
  word_502DBB0 = 0;
  qword_502DBE8 = (__int64)&unk_502DBF8;
  qword_502DBB8 = 0;
  dword_502DBA8 = v30;
  qword_502DBC0 = 0;
  qword_502DBC8 = 0;
  qword_502DBD0 = 0;
  qword_502DBD8 = 0;
  qword_502DBE0 = 0;
  qword_502DC00 = 0;
  qword_502DC08 = (__int64)&unk_502DC20;
  qword_502DC10 = 1;
  dword_502DC18 = 0;
  byte_502DC1C = 1;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_502DBF0;
  if ( (unsigned __int64)(unsigned int)qword_502DBF0 + 1 > HIDWORD(qword_502DBF0) )
  {
    v49 = v31;
    sub_C8D5F0((char *)&unk_502DBF8 - 16, &unk_502DBF8, (unsigned int)qword_502DBF0 + 1LL, 8);
    v32 = (unsigned int)qword_502DBF0;
    v31 = v49;
  }
  *(_QWORD *)(qword_502DBE8 + 8 * v32) = v31;
  qword_502DC30 = (__int64)&unk_49D9748;
  qword_502DBA0 = (__int64)&unk_49DC090;
  qword_502DC40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502DBF0) = qword_502DBF0 + 1;
  qword_502DC60 = (__int64)nullsub_23;
  qword_502DC28 = 0;
  qword_502DC58 = (__int64)sub_984030;
  qword_502DC38 = 0;
  sub_C53080(&qword_502DBA0, "print-no-modref", 15);
  LOBYTE(dword_502DBAC) = dword_502DBAC & 0x9F | 0x40;
  sub_C53130(&qword_502DBA0);
  __cxa_atexit(sub_984900, &qword_502DBA0, &qword_4A427C0);
  v51[0] = 2;
  sub_30A06F0(&unk_502DAC0, "print-ref", v51);
  __cxa_atexit(sub_984900, &unk_502DAC0, &qword_4A427C0);
  v51[0] = 2;
  sub_30A06F0(&unk_502D9E0, "print-mod", v51);
  __cxa_atexit(sub_984900, &unk_502D9E0, &qword_4A427C0);
  qword_502D900 = (__int64)&unk_49DC150;
  v35 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_502D9E0, v33, v34), 1u);
  byte_502D97C = 1;
  qword_502D950 = 0x100000000LL;
  dword_502D90C &= 0x8000u;
  qword_502D948 = (__int64)&unk_502D958;
  qword_502D918 = 0;
  qword_502D920 = 0;
  dword_502D908 = v35;
  word_502D910 = 0;
  qword_502D928 = 0;
  qword_502D930 = 0;
  qword_502D938 = 0;
  qword_502D940 = 0;
  qword_502D960 = 0;
  qword_502D968 = (__int64)&unk_502D980;
  qword_502D970 = 1;
  dword_502D978 = 0;
  v36 = sub_C57470();
  v37 = (unsigned int)qword_502D950;
  if ( (unsigned __int64)(unsigned int)qword_502D950 + 1 > HIDWORD(qword_502D950) )
  {
    v50 = v36;
    sub_C8D5F0((char *)&unk_502D958 - 16, &unk_502D958, (unsigned int)qword_502D950 + 1LL, 8);
    v37 = (unsigned int)qword_502D950;
    v36 = v50;
  }
  *(_QWORD *)(qword_502D948 + 8 * v37) = v36;
  qword_502D990 = (__int64)&unk_49D9748;
  qword_502D900 = (__int64)&unk_49DC090;
  qword_502D9A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502D950) = qword_502D950 + 1;
  qword_502D9C0 = (__int64)nullsub_23;
  qword_502D988 = 0;
  qword_502D9B8 = (__int64)sub_984030;
  qword_502D998 = 0;
  sub_C53080(&qword_502D900, "print-modref", 12);
  LOBYTE(dword_502D90C) = dword_502D90C & 0x9F | 0x40;
  sub_C53130(&qword_502D900);
  __cxa_atexit(sub_984900, &qword_502D900, &qword_4A427C0);
  qword_502D820 = (__int64)&unk_49DC150;
  v40 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502D900, v38, v39), 1u);
  dword_502D82C &= 0x8000u;
  word_502D830 = 0;
  qword_502D870 = 0x100000000LL;
  qword_502D868 = (__int64)&unk_502D878;
  qword_502D838 = 0;
  qword_502D840 = 0;
  dword_502D828 = v40;
  qword_502D848 = 0;
  qword_502D850 = 0;
  qword_502D858 = 0;
  qword_502D860 = 0;
  qword_502D880 = 0;
  qword_502D888 = (__int64)&unk_502D8A0;
  qword_502D890 = 1;
  dword_502D898 = 0;
  byte_502D89C = 1;
  v41 = sub_C57470();
  v42 = (unsigned int)qword_502D870;
  v43 = (unsigned int)qword_502D870 + 1LL;
  if ( v43 > HIDWORD(qword_502D870) )
  {
    sub_C8D5F0((char *)&unk_502D878 - 16, &unk_502D878, v43, 8);
    v42 = (unsigned int)qword_502D870;
  }
  *(_QWORD *)(qword_502D868 + 8 * v42) = v41;
  qword_502D8B0 = (__int64)&unk_49D9748;
  qword_502D820 = (__int64)&unk_49DC090;
  qword_502D8C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502D870) = qword_502D870 + 1;
  qword_502D8E0 = (__int64)nullsub_23;
  qword_502D8A8 = 0;
  qword_502D8D8 = (__int64)sub_984030;
  qword_502D8B8 = 0;
  sub_C53080(&qword_502D820, "evaluate-aa-metadata", 20);
  LOBYTE(dword_502D82C) = dword_502D82C & 0x9F | 0x40;
  sub_C53130(&qword_502D820);
  return __cxa_atexit(sub_984900, &qword_502D820, &qword_4A427C0);
}
