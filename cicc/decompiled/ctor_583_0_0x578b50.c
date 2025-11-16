// Function: ctor_583_0
// Address: 0x578b50
//
int ctor_583_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v33; // [rsp+8h] [rbp-108h]
  __int64 v34; // [rsp+8h] [rbp-108h]
  __int64 v35; // [rsp+8h] [rbp-108h]
  __int64 v36; // [rsp+8h] [rbp-108h]
  __int64 v37; // [rsp+8h] [rbp-108h]
  int v38; // [rsp+10h] [rbp-100h] BYREF
  int v39; // [rsp+14h] [rbp-FCh] BYREF
  int *v40; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v41[2]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v42[2]; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD v43[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v44; // [rsp+50h] [rbp-C0h]
  char *v45; // [rsp+58h] [rbp-B8h]
  __int64 v46; // [rsp+60h] [rbp-B0h]
  char *v47; // [rsp+68h] [rbp-A8h]
  __int64 v48; // [rsp+70h] [rbp-A0h]
  int v49; // [rsp+78h] [rbp-98h]
  const char *v50; // [rsp+80h] [rbp-90h]
  __int64 v51; // [rsp+88h] [rbp-88h]
  char *v52; // [rsp+90h] [rbp-80h]
  __int64 v53; // [rsp+98h] [rbp-78h]
  int v54; // [rsp+A0h] [rbp-70h]
  const char *v55; // [rsp+A8h] [rbp-68h]
  __int64 v56; // [rsp+B0h] [rbp-60h]

  v40 = &v39;
  v43[0] = "default";
  v45 = "Default";
  v47 = "size";
  v50 = "Optimize for size";
  v52 = "speed";
  v55 = "Optimize for speed";
  v42[1] = 0x400000003LL;
  v39 = 2;
  v42[0] = v43;
  v43[1] = 7;
  v44 = 0;
  v46 = 7;
  v48 = 4;
  v49 = 1;
  v51 = 17;
  v53 = 5;
  v54 = 2;
  v56 = 18;
  v41[0] = "Spill mode for splitting live ranges";
  v41[1] = 36;
  v38 = 1;
  sub_2F5E300(&unk_50241C0, "split-spill-mode", &v38, v41, v42, &v40);
  if ( (_QWORD *)v42[0] != v43 )
    _libc_free(v42[0], "split-spill-mode");
  __cxa_atexit(sub_2F4DF50, &unk_50241C0, &qword_4A427C0);
  qword_50240E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5024130 = 0x100000000LL;
  word_50240F0 = 0;
  dword_50240EC &= 0x8000u;
  qword_50240F8 = 0;
  qword_5024100 = 0;
  dword_50240E8 = v0;
  qword_5024108 = 0;
  qword_5024110 = 0;
  qword_5024118 = 0;
  qword_5024120 = 0;
  qword_5024128 = (__int64)&unk_5024138;
  qword_5024140 = 0;
  qword_5024148 = (__int64)&unk_5024160;
  qword_5024150 = 1;
  dword_5024158 = 0;
  byte_502415C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5024130;
  v3 = (unsigned int)qword_5024130 + 1LL;
  if ( v3 > HIDWORD(qword_5024130) )
  {
    sub_C8D5F0((char *)&unk_5024138 - 16, &unk_5024138, v3, 8);
    v2 = (unsigned int)qword_5024130;
  }
  *(_QWORD *)(qword_5024128 + 8 * v2) = v1;
  LODWORD(qword_5024130) = qword_5024130 + 1;
  qword_5024168 = 0;
  qword_5024170 = (__int64)&unk_49D9728;
  qword_5024178 = 0;
  qword_50240E0 = (__int64)&unk_49DBF10;
  qword_5024180 = (__int64)&unk_49DC290;
  qword_50241A0 = (__int64)nullsub_24;
  qword_5024198 = (__int64)sub_984050;
  sub_C53080(&qword_50240E0, "lcr-max-depth", 13);
  qword_5024110 = 32;
  LODWORD(qword_5024168) = 5;
  BYTE4(qword_5024178) = 1;
  LODWORD(qword_5024178) = 5;
  LOBYTE(dword_50240EC) = dword_50240EC & 0x9F | 0x20;
  qword_5024108 = (__int64)"Last chance recoloring max depth";
  sub_C53130(&qword_50240E0);
  __cxa_atexit(sub_984970, &qword_50240E0, &qword_4A427C0);
  qword_5024000 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5024050 = 0x100000000LL;
  dword_502400C &= 0x8000u;
  word_5024010 = 0;
  qword_5024018 = 0;
  qword_5024020 = 0;
  dword_5024008 = v4;
  qword_5024028 = 0;
  qword_5024030 = 0;
  qword_5024038 = 0;
  qword_5024040 = 0;
  qword_5024048 = (__int64)&unk_5024058;
  qword_5024060 = 0;
  qword_5024068 = (__int64)&unk_5024080;
  qword_5024070 = 1;
  dword_5024078 = 0;
  byte_502407C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5024050;
  v7 = (unsigned int)qword_5024050 + 1LL;
  if ( v7 > HIDWORD(qword_5024050) )
  {
    sub_C8D5F0((char *)&unk_5024058 - 16, &unk_5024058, v7, 8);
    v6 = (unsigned int)qword_5024050;
  }
  *(_QWORD *)(qword_5024048 + 8 * v6) = v5;
  LODWORD(qword_5024050) = qword_5024050 + 1;
  qword_5024088 = 0;
  qword_5024090 = (__int64)&unk_49D9728;
  qword_5024098 = 0;
  qword_5024000 = (__int64)&unk_49DBF10;
  qword_50240A0 = (__int64)&unk_49DC290;
  qword_50240C0 = (__int64)nullsub_24;
  qword_50240B8 = (__int64)sub_984050;
  sub_C53080(&qword_5024000, "lcr-max-interf", 14);
  qword_5024030 = 74;
  LODWORD(qword_5024088) = 8;
  BYTE4(qword_5024098) = 1;
  LODWORD(qword_5024098) = 8;
  LOBYTE(dword_502400C) = dword_502400C & 0x9F | 0x20;
  qword_5024028 = (__int64)"Last chance recoloring maximum number of considered interference at a time";
  sub_C53130(&qword_5024000);
  __cxa_atexit(sub_984970, &qword_5024000, &qword_4A427C0);
  qword_5023F20 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5023F70 = 0x100000000LL;
  dword_5023F2C &= 0x8000u;
  word_5023F30 = 0;
  qword_5023F38 = 0;
  qword_5023F40 = 0;
  dword_5023F28 = v8;
  qword_5023F48 = 0;
  qword_5023F50 = 0;
  qword_5023F58 = 0;
  qword_5023F60 = 0;
  qword_5023F68 = (__int64)&unk_5023F78;
  qword_5023F80 = 0;
  qword_5023F88 = (__int64)&unk_5023FA0;
  qword_5023F90 = 1;
  dword_5023F98 = 0;
  byte_5023F9C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5023F70;
  v11 = (unsigned int)qword_5023F70 + 1LL;
  if ( v11 > HIDWORD(qword_5023F70) )
  {
    sub_C8D5F0((char *)&unk_5023F78 - 16, &unk_5023F78, v11, 8);
    v10 = (unsigned int)qword_5023F70;
  }
  *(_QWORD *)(qword_5023F68 + 8 * v10) = v9;
  qword_5023FB0 = (__int64)&unk_49D9748;
  qword_5023F20 = (__int64)&unk_49DC090;
  LODWORD(qword_5023F70) = qword_5023F70 + 1;
  qword_5023FA8 = 0;
  qword_5023FC0 = (__int64)&unk_49DC1D0;
  qword_5023FB8 = 0;
  qword_5023FE0 = (__int64)nullsub_23;
  qword_5023FD8 = (__int64)sub_984030;
  sub_C53080(&qword_5023F20, "exhaustive-register-search", 26);
  qword_5023F50 = 102;
  qword_5023F48 = (__int64)"Exhaustive Search for registers bypassing the depth and interference cutoffs of last chance recoloring";
  LOBYTE(dword_5023F2C) = dword_5023F2C & 0x9F | 0x20;
  sub_C53130(&qword_5023F20);
  __cxa_atexit(sub_984900, &qword_5023F20, &qword_4A427C0);
  qword_5023E40 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5023E90 = 0x100000000LL;
  dword_5023E4C &= 0x8000u;
  qword_5023E88 = (__int64)&unk_5023E98;
  word_5023E50 = 0;
  qword_5023E58 = 0;
  dword_5023E48 = v12;
  qword_5023E60 = 0;
  qword_5023E68 = 0;
  qword_5023E70 = 0;
  qword_5023E78 = 0;
  qword_5023E80 = 0;
  qword_5023EA0 = 0;
  qword_5023EA8 = (__int64)&unk_5023EC0;
  qword_5023EB0 = 1;
  dword_5023EB8 = 0;
  byte_5023EBC = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_5023E90;
  if ( (unsigned __int64)(unsigned int)qword_5023E90 + 1 > HIDWORD(qword_5023E90) )
  {
    v33 = v13;
    sub_C8D5F0((char *)&unk_5023E98 - 16, &unk_5023E98, (unsigned int)qword_5023E90 + 1LL, 8);
    v14 = (unsigned int)qword_5023E90;
    v13 = v33;
  }
  *(_QWORD *)(qword_5023E88 + 8 * v14) = v13;
  qword_5023ED0 = (__int64)&unk_49D9748;
  qword_5023E40 = (__int64)&unk_49DC090;
  LODWORD(qword_5023E90) = qword_5023E90 + 1;
  qword_5023EC8 = 0;
  qword_5023EE0 = (__int64)&unk_49DC1D0;
  qword_5023ED8 = 0;
  qword_5023F00 = (__int64)nullsub_23;
  qword_5023EF8 = (__int64)sub_984030;
  sub_C53080(&qword_5023E40, "enable-deferred-spilling", 24);
  qword_5023E70 = 218;
  LOWORD(qword_5023ED8) = 256;
  LOBYTE(qword_5023EC8) = 0;
  LOBYTE(dword_5023E4C) = dword_5023E4C & 0x9F | 0x20;
  qword_5023E68 = (__int64)"Instead of spilling a variable right away, defer the actual code insertion to the end of the "
                           "allocation. That way the allocator might still find a suitable coloring for this variable bec"
                           "ause of other evicted variables.";
  sub_C53130(&qword_5023E40);
  __cxa_atexit(sub_984900, &qword_5023E40, &qword_4A427C0);
  qword_5023D60 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5023DB0 = 0x100000000LL;
  dword_5023D6C &= 0x8000u;
  word_5023D70 = 0;
  qword_5023DA8 = (__int64)&unk_5023DB8;
  qword_5023D78 = 0;
  dword_5023D68 = v15;
  qword_5023D80 = 0;
  qword_5023D88 = 0;
  qword_5023D90 = 0;
  qword_5023D98 = 0;
  qword_5023DA0 = 0;
  qword_5023DC0 = 0;
  qword_5023DC8 = (__int64)&unk_5023DE0;
  qword_5023DD0 = 1;
  dword_5023DD8 = 0;
  byte_5023DDC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5023DB0;
  if ( (unsigned __int64)(unsigned int)qword_5023DB0 + 1 > HIDWORD(qword_5023DB0) )
  {
    v34 = v16;
    sub_C8D5F0((char *)&unk_5023DB8 - 16, &unk_5023DB8, (unsigned int)qword_5023DB0 + 1LL, 8);
    v17 = (unsigned int)qword_5023DB0;
    v16 = v34;
  }
  *(_QWORD *)(qword_5023DA8 + 8 * v17) = v16;
  LODWORD(qword_5023DB0) = qword_5023DB0 + 1;
  qword_5023DE8 = 0;
  qword_5023DF0 = (__int64)&unk_49D9728;
  qword_5023DF8 = 0;
  qword_5023D60 = (__int64)&unk_49DBF10;
  qword_5023E00 = (__int64)&unk_49DC290;
  qword_5023E20 = (__int64)nullsub_24;
  qword_5023E18 = (__int64)sub_984050;
  sub_C53080(&qword_5023D60, "regalloc-csr-first-time-cost", 28);
  qword_5023D90 = 49;
  qword_5023D88 = (__int64)"Cost for first time use of callee-saved register.";
  LODWORD(qword_5023DE8) = 0;
  BYTE4(qword_5023DF8) = 1;
  LODWORD(qword_5023DF8) = 0;
  LOBYTE(dword_5023D6C) = dword_5023D6C & 0x9F | 0x20;
  sub_C53130(&qword_5023D60);
  __cxa_atexit(sub_984970, &qword_5023D60, &qword_4A427C0);
  qword_5023C80 = (__int64)&unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5023CFC = 1;
  word_5023C90 = 0;
  qword_5023CD0 = 0x100000000LL;
  dword_5023C8C &= 0x8000u;
  qword_5023CC8 = (__int64)&unk_5023CD8;
  qword_5023C98 = 0;
  dword_5023C88 = v18;
  qword_5023CA0 = 0;
  qword_5023CA8 = 0;
  qword_5023CB0 = 0;
  qword_5023CB8 = 0;
  qword_5023CC0 = 0;
  qword_5023CE0 = 0;
  qword_5023CE8 = (__int64)&unk_5023D00;
  qword_5023CF0 = 1;
  dword_5023CF8 = 0;
  v19 = sub_C57470();
  v20 = (unsigned int)qword_5023CD0;
  if ( (unsigned __int64)(unsigned int)qword_5023CD0 + 1 > HIDWORD(qword_5023CD0) )
  {
    v35 = v19;
    sub_C8D5F0((char *)&unk_5023CD8 - 16, &unk_5023CD8, (unsigned int)qword_5023CD0 + 1LL, 8);
    v20 = (unsigned int)qword_5023CD0;
    v19 = v35;
  }
  *(_QWORD *)(qword_5023CC8 + 8 * v20) = v19;
  LODWORD(qword_5023CD0) = qword_5023CD0 + 1;
  byte_5023D20 = 0;
  qword_5023D10 = (__int64)&unk_49DB998;
  qword_5023D08 = 0;
  qword_5023D18 = 0;
  qword_5023C80 = (__int64)&unk_49DB9B8;
  qword_5023D28 = (__int64)&unk_49DC2C0;
  qword_5023D48 = (__int64)nullsub_121;
  qword_5023D40 = (__int64)sub_C1A370;
  sub_C53080(&qword_5023C80, "grow-region-complexity-budget", 29);
  qword_5023CB0 = 114;
  qword_5023CA8 = (__int64)"growRegion() does not scale with the number of BB edges, so limit its budget and bail out onc"
                           "e we reach the limit.";
  qword_5023D08 = 10000;
  byte_5023D20 = 1;
  qword_5023D18 = 10000;
  LOBYTE(dword_5023C8C) = dword_5023C8C & 0x9F | 0x20;
  sub_C53130(&qword_5023C80);
  __cxa_atexit(sub_C1A610, &qword_5023C80, &qword_4A427C0);
  qword_5023BA0 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5023BF0 = 0x100000000LL;
  dword_5023BAC &= 0x8000u;
  word_5023BB0 = 0;
  qword_5023BE8 = (__int64)&unk_5023BF8;
  qword_5023BB8 = 0;
  dword_5023BA8 = v21;
  qword_5023BC0 = 0;
  qword_5023BC8 = 0;
  qword_5023BD0 = 0;
  qword_5023BD8 = 0;
  qword_5023BE0 = 0;
  qword_5023C00 = 0;
  qword_5023C08 = (__int64)&unk_5023C20;
  qword_5023C10 = 1;
  dword_5023C18 = 0;
  byte_5023C1C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_5023BF0;
  if ( (unsigned __int64)(unsigned int)qword_5023BF0 + 1 > HIDWORD(qword_5023BF0) )
  {
    v36 = v22;
    sub_C8D5F0((char *)&unk_5023BF8 - 16, &unk_5023BF8, (unsigned int)qword_5023BF0 + 1LL, 8);
    v23 = (unsigned int)qword_5023BF0;
    v22 = v36;
  }
  *(_QWORD *)(qword_5023BE8 + 8 * v23) = v22;
  qword_5023C30 = (__int64)&unk_49D9748;
  qword_5023BA0 = (__int64)&unk_49DC090;
  LODWORD(qword_5023BF0) = qword_5023BF0 + 1;
  qword_5023C28 = 0;
  qword_5023C40 = (__int64)&unk_49DC1D0;
  qword_5023C38 = 0;
  qword_5023C60 = (__int64)nullsub_23;
  qword_5023C58 = (__int64)sub_984030;
  sub_C53080(&qword_5023BA0, "greedy-regclass-priority-trumps-globalness", 42);
  qword_5023BD0 = 171;
  qword_5023BC8 = (__int64)"Change the greedy register allocator's live range priority calculation to make the Allocation"
                           "Priority of the register class more important then whether the range is global";
  LOBYTE(dword_5023BAC) = dword_5023BAC & 0x9F | 0x20;
  sub_C53130(&qword_5023BA0);
  __cxa_atexit(sub_984900, &qword_5023BA0, &qword_4A427C0);
  qword_5023AC0 = (__int64)&unk_49DC150;
  v24 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5023B3C = 1;
  qword_5023B10 = 0x100000000LL;
  dword_5023ACC &= 0x8000u;
  qword_5023B08 = (__int64)&unk_5023B18;
  qword_5023AD8 = 0;
  qword_5023AE0 = 0;
  dword_5023AC8 = v24;
  word_5023AD0 = 0;
  qword_5023AE8 = 0;
  qword_5023AF0 = 0;
  qword_5023AF8 = 0;
  qword_5023B00 = 0;
  qword_5023B20 = 0;
  qword_5023B28 = (__int64)&unk_5023B40;
  qword_5023B30 = 1;
  dword_5023B38 = 0;
  v25 = sub_C57470();
  v26 = (unsigned int)qword_5023B10;
  if ( (unsigned __int64)(unsigned int)qword_5023B10 + 1 > HIDWORD(qword_5023B10) )
  {
    v37 = v25;
    sub_C8D5F0((char *)&unk_5023B18 - 16, &unk_5023B18, (unsigned int)qword_5023B10 + 1LL, 8);
    v26 = (unsigned int)qword_5023B10;
    v25 = v37;
  }
  *(_QWORD *)(qword_5023B08 + 8 * v26) = v25;
  qword_5023B50 = (__int64)&unk_49D9748;
  qword_5023AC0 = (__int64)&unk_49DC090;
  LODWORD(qword_5023B10) = qword_5023B10 + 1;
  qword_5023B48 = 0;
  qword_5023B60 = (__int64)&unk_49DC1D0;
  qword_5023B58 = 0;
  qword_5023B80 = (__int64)nullsub_23;
  qword_5023B78 = (__int64)sub_984030;
  sub_C53080(&qword_5023AC0, "greedy-reverse-local-assignment", 31);
  qword_5023AF0 = 114;
  qword_5023AE8 = (__int64)"Reverse allocation order of local live ranges, such that shorter local live ranges will tend "
                           "to be allocated first";
  LOBYTE(dword_5023ACC) = dword_5023ACC & 0x9F | 0x20;
  sub_C53130(&qword_5023AC0);
  __cxa_atexit(sub_984900, &qword_5023AC0, &qword_4A427C0);
  qword_50239E0 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50239EC &= 0x8000u;
  word_50239F0 = 0;
  qword_5023A30 = 0x100000000LL;
  qword_50239F8 = 0;
  qword_5023A00 = 0;
  qword_5023A08 = 0;
  dword_50239E8 = v27;
  qword_5023A10 = 0;
  qword_5023A18 = 0;
  qword_5023A20 = 0;
  qword_5023A28 = (__int64)&unk_5023A38;
  qword_5023A40 = 0;
  qword_5023A48 = (__int64)&unk_5023A60;
  qword_5023A50 = 1;
  dword_5023A58 = 0;
  byte_5023A5C = 1;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_5023A30;
  v30 = (unsigned int)qword_5023A30 + 1LL;
  if ( v30 > HIDWORD(qword_5023A30) )
  {
    sub_C8D5F0((char *)&unk_5023A38 - 16, &unk_5023A38, v30, 8);
    v29 = (unsigned int)qword_5023A30;
  }
  *(_QWORD *)(qword_5023A28 + 8 * v29) = v28;
  LODWORD(qword_5023A30) = qword_5023A30 + 1;
  qword_5023A68 = 0;
  qword_5023A70 = (__int64)&unk_49D9728;
  qword_5023A78 = 0;
  qword_50239E0 = (__int64)&unk_49DBF10;
  qword_5023A80 = (__int64)&unk_49DC290;
  qword_5023AA0 = (__int64)nullsub_24;
  qword_5023A98 = (__int64)sub_984050;
  sub_C53080(&qword_50239E0, "split-threshold-for-reg-with-hint", 33);
  qword_5023A10 = 73;
  qword_5023A08 = (__int64)"The threshold for splitting a virtual register with a hint, in percentage";
  LODWORD(qword_5023A68) = 75;
  BYTE4(qword_5023A78) = 1;
  LODWORD(qword_5023A78) = 75;
  LOBYTE(dword_50239EC) = dword_50239EC & 0x9F | 0x20;
  sub_C53130(&qword_50239E0);
  __cxa_atexit(sub_984970, &qword_50239E0, &qword_4A427C0);
  qword_50239A8 = (__int64)"greedy";
  v31 = unk_5023860;
  unk_5023860 = &qword_50239A0;
  qword_50239B0 = 6;
  qword_50239B8 = (__int64)"greedy register allocator";
  qword_50239C0 = 25;
  qword_50239C8 = (__int64)sub_2F504C0;
  qword_50239A0 = v31;
  if ( qword_5023870 )
    (*(void (__fastcall **)(_QWORD *, const char *, __int64, __int64 (__fastcall *)(), const char *, __int64))(*qword_5023870 + 24LL))(
      qword_5023870,
      "greedy",
      6,
      sub_2F504C0,
      "greedy register allocator",
      25);
  return __cxa_atexit(sub_2F41140, &qword_50239A0, &qword_4A427C0);
}
