// Function: ctor_638_0
// Address: 0x5937c0
//
int __fastcall ctor_638_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  int v28; // edx
  __int64 v29; // r15
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // edx
  __int64 v35; // r15
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rcx
  int v45; // edx
  __int64 v46; // r15
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // edx
  __int64 v52; // r12
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rcx
  int v57; // edx
  __int64 v58; // rbx
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  __int64 v62; // [rsp+10h] [rbp-60h]
  int v63; // [rsp+20h] [rbp-50h] BYREF
  int v64; // [rsp+24h] [rbp-4Ch] BYREF
  int *v65; // [rsp+28h] [rbp-48h] BYREF
  const char *v66; // [rsp+30h] [rbp-40h] BYREF
  __int64 v67; // [rsp+38h] [rbp-38h]

  qword_5034DC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_5034DCC &= 0x8000u;
  word_5034DD0 = 0;
  qword_5034E10 = 0x100000000LL;
  qword_5034DD8 = 0;
  qword_5034DE0 = 0;
  qword_5034DE8 = 0;
  dword_5034DC8 = v4;
  qword_5034DF0 = 0;
  qword_5034DF8 = 0;
  qword_5034E00 = 0;
  qword_5034E08 = (__int64)&unk_5034E18;
  qword_5034E20 = 0;
  qword_5034E28 = (__int64)&unk_5034E40;
  qword_5034E30 = 1;
  dword_5034E38 = 0;
  byte_5034E3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5034E10;
  v7 = (unsigned int)qword_5034E10 + 1LL;
  if ( v7 > HIDWORD(qword_5034E10) )
  {
    sub_C8D5F0((char *)&unk_5034E18 - 16, &unk_5034E18, v7, 8);
    v6 = (unsigned int)qword_5034E10;
  }
  *(_QWORD *)(qword_5034E08 + 8 * v6) = v5;
  LODWORD(qword_5034E10) = qword_5034E10 + 1;
  qword_5034E48 = 0;
  qword_5034E50 = (__int64)&unk_49D9748;
  qword_5034E58 = 0;
  qword_5034DC0 = (__int64)&unk_49DC090;
  qword_5034E60 = (__int64)&unk_49DC1D0;
  qword_5034E80 = (__int64)nullsub_23;
  qword_5034E78 = (__int64)sub_984030;
  sub_C53080(&qword_5034DC0, "force-specialization", 20);
  LOBYTE(qword_5034E48) = 0;
  LOWORD(qword_5034E58) = 256;
  qword_5034DF0 = 74;
  LOBYTE(dword_5034DCC) = dword_5034DCC & 0x9F | 0x20;
  qword_5034DE8 = (__int64)"Force function specialization for every call site with a constant argument";
  sub_C53130(&qword_5034DC0);
  __cxa_atexit(sub_984900, &qword_5034DC0, &qword_4A427C0);
  qword_5034CE0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5034DC0, v8, v9), 1u);
  dword_5034CEC &= 0x8000u;
  word_5034CF0 = 0;
  qword_5034D30 = 0x100000000LL;
  qword_5034CF8 = 0;
  qword_5034D00 = 0;
  qword_5034D08 = 0;
  dword_5034CE8 = v10;
  qword_5034D10 = 0;
  qword_5034D18 = 0;
  qword_5034D20 = 0;
  qword_5034D28 = (__int64)&unk_5034D38;
  qword_5034D40 = 0;
  qword_5034D48 = (__int64)&unk_5034D60;
  qword_5034D50 = 1;
  dword_5034D58 = 0;
  byte_5034D5C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5034D30;
  v13 = (unsigned int)qword_5034D30 + 1LL;
  if ( v13 > HIDWORD(qword_5034D30) )
  {
    sub_C8D5F0((char *)&unk_5034D38 - 16, &unk_5034D38, v13, 8);
    v12 = (unsigned int)qword_5034D30;
  }
  *(_QWORD *)(qword_5034D28 + 8 * v12) = v11;
  qword_5034D70 = (__int64)&unk_49D9728;
  qword_5034CE0 = (__int64)&unk_49DBF10;
  LODWORD(qword_5034D30) = qword_5034D30 + 1;
  qword_5034D68 = 0;
  qword_5034D80 = (__int64)&unk_49DC290;
  qword_5034D78 = 0;
  qword_5034DA0 = (__int64)nullsub_24;
  qword_5034D98 = (__int64)sub_984050;
  sub_C53080(&qword_5034CE0, "funcspec-max-clones", 19);
  LODWORD(qword_5034D68) = 3;
  BYTE4(qword_5034D78) = 1;
  LODWORD(qword_5034D78) = 3;
  qword_5034D10 = 73;
  LOBYTE(dword_5034CEC) = dword_5034CEC & 0x9F | 0x20;
  qword_5034D08 = (__int64)"The maximum number of clones allowed for a single function specialization";
  sub_C53130(&qword_5034CE0);
  __cxa_atexit(sub_984970, &qword_5034CE0, &qword_4A427C0);
  qword_5034C00 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5034CE0, v14, v15), 1u);
  qword_5034C50 = 0x100000000LL;
  dword_5034C0C &= 0x8000u;
  word_5034C10 = 0;
  qword_5034C48 = (__int64)&unk_5034C58;
  qword_5034C18 = 0;
  dword_5034C08 = v16;
  qword_5034C20 = 0;
  qword_5034C28 = 0;
  qword_5034C30 = 0;
  qword_5034C38 = 0;
  qword_5034C40 = 0;
  qword_5034C60 = 0;
  qword_5034C68 = (__int64)&unk_5034C80;
  qword_5034C70 = 1;
  dword_5034C78 = 0;
  byte_5034C7C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_5034C50;
  v19 = (unsigned int)qword_5034C50 + 1LL;
  if ( v19 > HIDWORD(qword_5034C50) )
  {
    sub_C8D5F0((char *)&unk_5034C58 - 16, &unk_5034C58, v19, 8);
    v18 = (unsigned int)qword_5034C50;
  }
  *(_QWORD *)(qword_5034C48 + 8 * v18) = v17;
  qword_5034C90 = (__int64)&unk_49D9728;
  qword_5034C00 = (__int64)&unk_49DBF10;
  LODWORD(qword_5034C50) = qword_5034C50 + 1;
  qword_5034C88 = 0;
  qword_5034CA0 = (__int64)&unk_49DC290;
  qword_5034C98 = 0;
  qword_5034CC0 = (__int64)nullsub_24;
  qword_5034CB8 = (__int64)sub_984050;
  sub_C53080(&qword_5034C00, "funcspec-max-discovery-iterations", 33);
  LODWORD(qword_5034C88) = 100;
  BYTE4(qword_5034C98) = 1;
  LODWORD(qword_5034C98) = 100;
  qword_5034C30 = 75;
  LOBYTE(dword_5034C0C) = dword_5034C0C & 0x9F | 0x20;
  qword_5034C28 = (__int64)"The maximum number of iterations allowed when searching for transitive phis";
  sub_C53130(&qword_5034C00);
  __cxa_atexit(sub_984970, &qword_5034C00, &qword_4A427C0);
  qword_5034B20 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5034C00, v20, v21), 1u);
  qword_5034B70 = 0x100000000LL;
  dword_5034B2C &= 0x8000u;
  qword_5034B68 = (__int64)&unk_5034B78;
  word_5034B30 = 0;
  qword_5034B38 = 0;
  dword_5034B28 = v22;
  qword_5034B40 = 0;
  qword_5034B48 = 0;
  qword_5034B50 = 0;
  qword_5034B58 = 0;
  qword_5034B60 = 0;
  qword_5034B80 = 0;
  qword_5034B88 = (__int64)&unk_5034BA0;
  qword_5034B90 = 1;
  dword_5034B98 = 0;
  byte_5034B9C = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_5034B70;
  v25 = (unsigned int)qword_5034B70 + 1LL;
  if ( v25 > HIDWORD(qword_5034B70) )
  {
    sub_C8D5F0((char *)&unk_5034B78 - 16, &unk_5034B78, v25, 8);
    v24 = (unsigned int)qword_5034B70;
  }
  *(_QWORD *)(qword_5034B68 + 8 * v24) = v23;
  qword_5034BB0 = (__int64)&unk_49D9728;
  qword_5034B20 = (__int64)&unk_49DBF10;
  LODWORD(qword_5034B70) = qword_5034B70 + 1;
  qword_5034BA8 = 0;
  qword_5034BC0 = (__int64)&unk_49DC290;
  qword_5034BB8 = 0;
  qword_5034BE0 = (__int64)nullsub_24;
  qword_5034BD8 = (__int64)sub_984050;
  sub_C53080(&qword_5034B20, "funcspec-max-incoming-phi-values", 32);
  LODWORD(qword_5034BA8) = 8;
  BYTE4(qword_5034BB8) = 1;
  LODWORD(qword_5034BB8) = 8;
  qword_5034B50 = 117;
  LOBYTE(dword_5034B2C) = dword_5034B2C & 0x9F | 0x20;
  qword_5034B48 = (__int64)"The maximum number of incoming values a PHI node can have to be considered during the special"
                           "ization bonus estimation";
  sub_C53130(&qword_5034B20);
  __cxa_atexit(sub_984970, &qword_5034B20, &qword_4A427C0);
  qword_5034A40 = (__int64)&unk_49DC150;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5034B20, v26, v27), 1u);
  qword_5034A90 = 0x100000000LL;
  dword_5034A4C &= 0x8000u;
  qword_5034A88 = (__int64)&unk_5034A98;
  word_5034A50 = 0;
  qword_5034A58 = 0;
  dword_5034A48 = v28;
  qword_5034A60 = 0;
  qword_5034A68 = 0;
  qword_5034A70 = 0;
  qword_5034A78 = 0;
  qword_5034A80 = 0;
  qword_5034AA0 = 0;
  qword_5034AA8 = (__int64)&unk_5034AC0;
  qword_5034AB0 = 1;
  dword_5034AB8 = 0;
  byte_5034ABC = 1;
  v29 = sub_C57470();
  v30 = (unsigned int)qword_5034A90;
  v31 = (unsigned int)qword_5034A90 + 1LL;
  if ( v31 > HIDWORD(qword_5034A90) )
  {
    sub_C8D5F0((char *)&unk_5034A98 - 16, &unk_5034A98, v31, 8);
    v30 = (unsigned int)qword_5034A90;
  }
  *(_QWORD *)(qword_5034A88 + 8 * v30) = v29;
  qword_5034AD0 = (__int64)&unk_49D9728;
  qword_5034A40 = (__int64)&unk_49DBF10;
  LODWORD(qword_5034A90) = qword_5034A90 + 1;
  qword_5034AC8 = 0;
  qword_5034AE0 = (__int64)&unk_49DC290;
  qword_5034AD8 = 0;
  qword_5034B00 = (__int64)nullsub_24;
  qword_5034AF8 = (__int64)sub_984050;
  sub_C53080(&qword_5034A40, "funcspec-max-block-predecessors", 31);
  LODWORD(qword_5034AC8) = 2;
  BYTE4(qword_5034AD8) = 1;
  LODWORD(qword_5034AD8) = 2;
  qword_5034A70 = 109;
  LOBYTE(dword_5034A4C) = dword_5034A4C & 0x9F | 0x20;
  qword_5034A68 = (__int64)"The maximum number of predecessors a basic block can have to be considered during the estimat"
                           "ion of dead code";
  sub_C53130(&qword_5034A40);
  __cxa_atexit(sub_984970, &qword_5034A40, &qword_4A427C0);
  qword_5034960 = (__int64)&unk_49DC150;
  v34 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5034A40, v32, v33), 1u);
  qword_50349B0 = 0x100000000LL;
  dword_503496C &= 0x8000u;
  qword_50349A8 = (__int64)&unk_50349B8;
  word_5034970 = 0;
  qword_5034978 = 0;
  dword_5034968 = v34;
  qword_5034980 = 0;
  qword_5034988 = 0;
  qword_5034990 = 0;
  qword_5034998 = 0;
  qword_50349A0 = 0;
  qword_50349C0 = 0;
  qword_50349C8 = (__int64)&unk_50349E0;
  qword_50349D0 = 1;
  dword_50349D8 = 0;
  byte_50349DC = 1;
  v35 = sub_C57470();
  v36 = (unsigned int)qword_50349B0;
  v37 = (unsigned int)qword_50349B0 + 1LL;
  if ( v37 > HIDWORD(qword_50349B0) )
  {
    sub_C8D5F0((char *)&unk_50349B8 - 16, &unk_50349B8, v37, 8);
    v36 = (unsigned int)qword_50349B0;
  }
  *(_QWORD *)(qword_50349A8 + 8 * v36) = v35;
  qword_50349F0 = (__int64)&unk_49D9728;
  qword_5034960 = (__int64)&unk_49DBF10;
  LODWORD(qword_50349B0) = qword_50349B0 + 1;
  qword_50349E8 = 0;
  qword_5034A00 = (__int64)&unk_49DC290;
  qword_50349F8 = 0;
  qword_5034A20 = (__int64)nullsub_24;
  qword_5034A18 = (__int64)sub_984050;
  sub_C53080(&qword_5034960, "funcspec-min-function-size", 26);
  LODWORD(qword_50349E8) = 500;
  BYTE4(qword_50349F8) = 1;
  LODWORD(qword_50349F8) = 500;
  qword_5034990 = 74;
  LOBYTE(dword_503496C) = dword_503496C & 0x9F | 0x20;
  qword_5034988 = (__int64)"Don't specialize functions that have less than this number of instructions";
  sub_C53130(&qword_5034960);
  __cxa_atexit(sub_984970, &qword_5034960, &qword_4A427C0);
  v66 = "Maximum codesize growth allowed per function";
  v65 = &v63;
  v67 = 44;
  v64 = 1;
  v63 = 3;
  sub_2ABC530(&unk_5034880, "funcspec-max-codesize-growth", &v65, &v64, &v66);
  __cxa_atexit(sub_984970, &unk_5034880, &qword_4A427C0);
  qword_50347A0 = (__int64)&unk_49DC150;
  v40 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &unk_5034880, v38, v39), 1u);
  qword_50347F0 = 0x100000000LL;
  dword_50347AC &= 0x8000u;
  word_50347B0 = 0;
  qword_50347E8 = (__int64)&unk_50347F8;
  qword_50347B8 = 0;
  dword_50347A8 = v40;
  qword_50347C0 = 0;
  qword_50347C8 = 0;
  qword_50347D0 = 0;
  qword_50347D8 = 0;
  qword_50347E0 = 0;
  qword_5034800 = 0;
  qword_5034808 = (__int64)&unk_5034820;
  qword_5034810 = 1;
  dword_5034818 = 0;
  byte_503481C = 1;
  v41 = sub_C57470();
  v42 = (unsigned int)qword_50347F0;
  if ( (unsigned __int64)(unsigned int)qword_50347F0 + 1 > HIDWORD(qword_50347F0) )
  {
    v62 = v41;
    sub_C8D5F0((char *)&unk_50347F8 - 16, &unk_50347F8, (unsigned int)qword_50347F0 + 1LL, 8);
    v42 = (unsigned int)qword_50347F0;
    v41 = v62;
  }
  *(_QWORD *)(qword_50347E8 + 8 * v42) = v41;
  qword_5034830 = (__int64)&unk_49D9728;
  qword_50347A0 = (__int64)&unk_49DBF10;
  LODWORD(qword_50347F0) = qword_50347F0 + 1;
  qword_5034828 = 0;
  qword_5034840 = (__int64)&unk_49DC290;
  qword_5034838 = 0;
  qword_5034860 = (__int64)nullsub_24;
  qword_5034858 = (__int64)sub_984050;
  sub_C53080(&qword_50347A0, "funcspec-min-codesize-savings", 29);
  LODWORD(qword_5034828) = 20;
  BYTE4(qword_5034838) = 1;
  LODWORD(qword_5034838) = 20;
  qword_50347D0 = 107;
  LOBYTE(dword_50347AC) = dword_50347AC & 0x9F | 0x20;
  qword_50347C8 = (__int64)"Reject specializations whose codesize savings are less than this much percent of the original function size";
  sub_C53130(&qword_50347A0);
  __cxa_atexit(sub_984970, &qword_50347A0, &qword_4A427C0);
  v67 = 106;
  v66 = "Reject specializations whose latency savings are less than this much percent of the original function size";
  v64 = 1;
  v65 = &v63;
  v63 = 40;
  sub_2ABC530(&unk_50346C0, "funcspec-min-latency-savings", &v65, &v64, &v66);
  __cxa_atexit(sub_984970, &unk_50346C0, &qword_4A427C0);
  qword_50345E0 = (__int64)&unk_49DC150;
  v45 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &unk_50346C0, v43, v44), 1u);
  qword_5034630 = 0x100000000LL;
  dword_50345EC &= 0x8000u;
  qword_5034628 = (__int64)&unk_5034638;
  word_50345F0 = 0;
  qword_50345F8 = 0;
  dword_50345E8 = v45;
  qword_5034600 = 0;
  qword_5034608 = 0;
  qword_5034610 = 0;
  qword_5034618 = 0;
  qword_5034620 = 0;
  qword_5034640 = 0;
  qword_5034648 = (__int64)&unk_5034660;
  qword_5034650 = 1;
  dword_5034658 = 0;
  byte_503465C = 1;
  v46 = sub_C57470();
  v47 = (unsigned int)qword_5034630;
  v48 = (unsigned int)qword_5034630 + 1LL;
  if ( v48 > HIDWORD(qword_5034630) )
  {
    sub_C8D5F0((char *)&unk_5034638 - 16, &unk_5034638, v48, 8);
    v47 = (unsigned int)qword_5034630;
  }
  *(_QWORD *)(qword_5034628 + 8 * v47) = v46;
  qword_5034670 = (__int64)&unk_49D9728;
  qword_50345E0 = (__int64)&unk_49DBF10;
  LODWORD(qword_5034630) = qword_5034630 + 1;
  qword_5034668 = 0;
  qword_5034680 = (__int64)&unk_49DC290;
  qword_5034678 = 0;
  qword_50346A0 = (__int64)nullsub_24;
  qword_5034698 = (__int64)sub_984050;
  sub_C53080(&qword_50345E0, "funcspec-min-inlining-bonus", 27);
  LODWORD(qword_5034668) = 300;
  BYTE4(qword_5034678) = 1;
  LODWORD(qword_5034678) = 300;
  qword_5034610 = 104;
  LOBYTE(dword_50345EC) = dword_50345EC & 0x9F | 0x20;
  qword_5034608 = (__int64)"Reject specializations whose inlining bonus is less than this much percent of the original function size";
  sub_C53130(&qword_50345E0);
  __cxa_atexit(sub_984970, &qword_50345E0, &qword_4A427C0);
  qword_5034500 = (__int64)&unk_49DC150;
  v51 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_50345E0, v49, v50), 1u);
  qword_5034550 = 0x100000000LL;
  dword_503450C &= 0x8000u;
  word_5034510 = 0;
  qword_5034518 = 0;
  qword_5034520 = 0;
  dword_5034508 = v51;
  qword_5034528 = 0;
  qword_5034530 = 0;
  qword_5034538 = 0;
  qword_5034540 = 0;
  qword_5034548 = (__int64)&unk_5034558;
  qword_5034560 = 0;
  qword_5034568 = (__int64)&unk_5034580;
  qword_5034570 = 1;
  dword_5034578 = 0;
  byte_503457C = 1;
  v52 = sub_C57470();
  v53 = (unsigned int)qword_5034550;
  v54 = (unsigned int)qword_5034550 + 1LL;
  if ( v54 > HIDWORD(qword_5034550) )
  {
    sub_C8D5F0((char *)&unk_5034558 - 16, &unk_5034558, v54, 8);
    v53 = (unsigned int)qword_5034550;
  }
  *(_QWORD *)(qword_5034548 + 8 * v53) = v52;
  LODWORD(qword_5034550) = qword_5034550 + 1;
  qword_5034588 = 0;
  qword_5034590 = (__int64)&unk_49D9748;
  qword_5034598 = 0;
  qword_5034500 = (__int64)&unk_49DC090;
  qword_50345A0 = (__int64)&unk_49DC1D0;
  qword_50345C0 = (__int64)nullsub_23;
  qword_50345B8 = (__int64)sub_984030;
  sub_C53080(&qword_5034500, "funcspec-on-address", 19);
  LOWORD(qword_5034598) = 256;
  LOBYTE(qword_5034588) = 0;
  qword_5034530 = 62;
  LOBYTE(dword_503450C) = dword_503450C & 0x9F | 0x20;
  qword_5034528 = (__int64)"Enable function specialization on the address of global values";
  sub_C53130(&qword_5034500);
  __cxa_atexit(sub_984900, &qword_5034500, &qword_4A427C0);
  qword_5034420 = (__int64)&unk_49DC150;
  v57 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5034500, v55, v56), 1u);
  qword_5034470 = 0x100000000LL;
  word_5034430 = 0;
  dword_503442C &= 0x8000u;
  qword_5034438 = 0;
  qword_5034440 = 0;
  dword_5034428 = v57;
  qword_5034448 = 0;
  qword_5034450 = 0;
  qword_5034458 = 0;
  qword_5034460 = 0;
  qword_5034468 = (__int64)&unk_5034478;
  qword_5034480 = 0;
  qword_5034488 = (__int64)&unk_50344A0;
  qword_5034490 = 1;
  dword_5034498 = 0;
  byte_503449C = 1;
  v58 = sub_C57470();
  v59 = (unsigned int)qword_5034470;
  v60 = (unsigned int)qword_5034470 + 1LL;
  if ( v60 > HIDWORD(qword_5034470) )
  {
    sub_C8D5F0((char *)&unk_5034478 - 16, &unk_5034478, v60, 8);
    v59 = (unsigned int)qword_5034470;
  }
  *(_QWORD *)(qword_5034468 + 8 * v59) = v58;
  LODWORD(qword_5034470) = qword_5034470 + 1;
  qword_50344A8 = 0;
  qword_50344B0 = (__int64)&unk_49D9748;
  qword_50344B8 = 0;
  qword_5034420 = (__int64)&unk_49DC090;
  qword_50344C0 = (__int64)&unk_49DC1D0;
  qword_50344E0 = (__int64)nullsub_23;
  qword_50344D8 = (__int64)sub_984030;
  sub_C53080(&qword_5034420, "funcspec-for-literal-constant", 29);
  LOBYTE(qword_50344A8) = 1;
  LOWORD(qword_50344B8) = 257;
  qword_5034450 = 78;
  LOBYTE(dword_503442C) = dword_503442C & 0x9F | 0x20;
  qword_5034448 = (__int64)"Enable specialization of functions that take a literal constant as an argument";
  sub_C53130(&qword_5034420);
  return __cxa_atexit(sub_984900, &qword_5034420, &qword_4A427C0);
}
