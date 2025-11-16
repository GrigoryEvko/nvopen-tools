// Function: ctor_097
// Address: 0x4a3de0
//
int ctor_097()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  char v25; // [rsp+23h] [rbp-4Dh] BYREF
  int v26; // [rsp+24h] [rbp-4Ch]
  char *v27; // [rsp+28h] [rbp-48h] BYREF
  const char *v28; // [rsp+30h] [rbp-40h]
  __int64 v29; // [rsp+38h] [rbp-38h]

  qword_4F92280 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F922D0 = 0x100000000LL;
  dword_4F9228C &= 0x8000u;
  word_4F92290 = 0;
  qword_4F92298 = 0;
  qword_4F922A0 = 0;
  dword_4F92288 = v0;
  qword_4F922A8 = 0;
  qword_4F922B0 = 0;
  qword_4F922B8 = 0;
  qword_4F922C0 = 0;
  qword_4F922C8 = (__int64)&unk_4F922D8;
  qword_4F922E0 = 0;
  qword_4F922E8 = (__int64)&unk_4F92300;
  qword_4F922F0 = 1;
  dword_4F922F8 = 0;
  byte_4F922FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F922D0;
  v3 = (unsigned int)qword_4F922D0 + 1LL;
  if ( v3 > HIDWORD(qword_4F922D0) )
  {
    sub_C8D5F0((char *)&unk_4F922D8 - 16, &unk_4F922D8, v3, 8);
    v2 = (unsigned int)qword_4F922D0;
  }
  *(_QWORD *)(qword_4F922C8 + 8 * v2) = v1;
  qword_4F92308 = (__int64)&byte_4F92318;
  qword_4F92330 = (__int64)&byte_4F92340;
  qword_4F92328 = (__int64)&unk_49DC130;
  qword_4F92280 = (__int64)&unk_49DC010;
  LODWORD(qword_4F922D0) = qword_4F922D0 + 1;
  qword_4F92310 = 0;
  qword_4F92358 = (__int64)&unk_49DC350;
  byte_4F92318 = 0;
  qword_4F92378 = (__int64)nullsub_92;
  qword_4F92338 = 0;
  qword_4F92370 = (__int64)sub_BC4D70;
  byte_4F92340 = 0;
  byte_4F92350 = 0;
  sub_C53080(&qword_4F92280, "cfg-func-name", 13);
  qword_4F922B0 = 70;
  LOBYTE(dword_4F9228C) = dword_4F9228C & 0x9F | 0x20;
  qword_4F922A8 = (__int64)"The name of a function (or its substring) whose CFG is viewed/printed.";
  sub_C53130(&qword_4F92280);
  __cxa_atexit(sub_BC5A40, &qword_4F92280, &qword_4A427C0);
  qword_4F92180 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F921D0 = 0x100000000LL;
  dword_4F9218C &= 0x8000u;
  word_4F92190 = 0;
  qword_4F92198 = 0;
  qword_4F921A0 = 0;
  dword_4F92188 = v4;
  qword_4F921A8 = 0;
  qword_4F921B0 = 0;
  qword_4F921B8 = 0;
  qword_4F921C0 = 0;
  qword_4F921C8 = (__int64)&unk_4F921D8;
  qword_4F921E0 = 0;
  qword_4F921E8 = (__int64)&unk_4F92200;
  qword_4F921F0 = 1;
  dword_4F921F8 = 0;
  byte_4F921FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F921D0;
  if ( (unsigned __int64)(unsigned int)qword_4F921D0 + 1 > HIDWORD(qword_4F921D0) )
  {
    v22 = v5;
    sub_C8D5F0((char *)&unk_4F921D8 - 16, &unk_4F921D8, (unsigned int)qword_4F921D0 + 1LL, 8);
    v6 = (unsigned int)qword_4F921D0;
    v5 = v22;
  }
  *(_QWORD *)(qword_4F921C8 + 8 * v6) = v5;
  qword_4F92208 = (__int64)&byte_4F92218;
  qword_4F92230 = (__int64)&byte_4F92240;
  qword_4F92228 = (__int64)&unk_49DC130;
  qword_4F92180 = (__int64)&unk_49DC010;
  LODWORD(qword_4F921D0) = qword_4F921D0 + 1;
  qword_4F92210 = 0;
  qword_4F92258 = (__int64)&unk_49DC350;
  byte_4F92218 = 0;
  qword_4F92278 = (__int64)nullsub_92;
  qword_4F92238 = 0;
  qword_4F92270 = (__int64)sub_BC4D70;
  byte_4F92240 = 0;
  byte_4F92250 = 0;
  sub_C53080(&qword_4F92180, "cfg-dot-filename-prefix", 23);
  qword_4F921B0 = 43;
  LOBYTE(dword_4F9218C) = dword_4F9218C & 0x9F | 0x20;
  qword_4F921A8 = (__int64)"The prefix used for the CFG dot file names.";
  sub_C53130(&qword_4F92180);
  __cxa_atexit(sub_BC5A40, &qword_4F92180, &qword_4A427C0);
  qword_4F920A0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F920F0 = 0x100000000LL;
  dword_4F920AC &= 0x8000u;
  word_4F920B0 = 0;
  qword_4F920B8 = 0;
  qword_4F920C0 = 0;
  dword_4F920A8 = v7;
  qword_4F920C8 = 0;
  qword_4F920D0 = 0;
  qword_4F920D8 = 0;
  qword_4F920E0 = 0;
  qword_4F920E8 = (__int64)&unk_4F920F8;
  qword_4F92100 = 0;
  qword_4F92108 = (__int64)&unk_4F92120;
  qword_4F92110 = 1;
  dword_4F92118 = 0;
  byte_4F9211C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F920F0;
  v10 = (unsigned int)qword_4F920F0 + 1LL;
  if ( v10 > HIDWORD(qword_4F920F0) )
  {
    sub_C8D5F0((char *)&unk_4F920F8 - 16, &unk_4F920F8, v10, 8);
    v9 = (unsigned int)qword_4F920F0;
  }
  *(_QWORD *)(qword_4F920E8 + 8 * v9) = v8;
  qword_4F92130 = (__int64)&unk_49D9748;
  qword_4F920A0 = (__int64)&unk_49DC090;
  qword_4F92140 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F920F0) = qword_4F920F0 + 1;
  qword_4F92160 = (__int64)nullsub_23;
  qword_4F92128 = 0;
  qword_4F92158 = (__int64)sub_984030;
  qword_4F92138 = 0;
  sub_C53080(&qword_4F920A0, "cfg-hide-unreachable-paths", 26);
  LOBYTE(qword_4F92128) = 0;
  LOWORD(qword_4F92138) = 256;
  sub_C53130(&qword_4F920A0);
  __cxa_atexit(sub_984900, &qword_4F920A0, &qword_4A427C0);
  qword_4F91FC0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F92010 = 0x100000000LL;
  dword_4F91FCC &= 0x8000u;
  word_4F91FD0 = 0;
  qword_4F92008 = (__int64)&unk_4F92018;
  qword_4F91FD8 = 0;
  dword_4F91FC8 = v11;
  qword_4F91FE0 = 0;
  qword_4F91FE8 = 0;
  qword_4F91FF0 = 0;
  qword_4F91FF8 = 0;
  qword_4F92000 = 0;
  qword_4F92020 = 0;
  qword_4F92028 = (__int64)&unk_4F92040;
  qword_4F92030 = 1;
  dword_4F92038 = 0;
  byte_4F9203C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4F92010;
  if ( (unsigned __int64)(unsigned int)qword_4F92010 + 1 > HIDWORD(qword_4F92010) )
  {
    v23 = v12;
    sub_C8D5F0((char *)&unk_4F92018 - 16, &unk_4F92018, (unsigned int)qword_4F92010 + 1LL, 8);
    v13 = (unsigned int)qword_4F92010;
    v12 = v23;
  }
  *(_QWORD *)(qword_4F92008 + 8 * v13) = v12;
  qword_4F92050 = (__int64)&unk_49D9748;
  qword_4F91FC0 = (__int64)&unk_49DC090;
  qword_4F92060 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F92010) = qword_4F92010 + 1;
  qword_4F92080 = (__int64)nullsub_23;
  qword_4F92048 = 0;
  qword_4F92078 = (__int64)sub_984030;
  qword_4F92058 = 0;
  sub_C53080(&qword_4F91FC0, "cfg-hide-deoptimize-paths", 25);
  LOBYTE(qword_4F92048) = 0;
  LOWORD(qword_4F92058) = 256;
  sub_C53130(&qword_4F91FC0);
  __cxa_atexit(sub_984900, &qword_4F91FC0, &qword_4A427C0);
  qword_4F91EE0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F91F5C = 1;
  word_4F91EF0 = 0;
  qword_4F91F30 = 0x100000000LL;
  dword_4F91EEC &= 0x8000u;
  qword_4F91F28 = (__int64)&unk_4F91F38;
  qword_4F91EF8 = 0;
  dword_4F91EE8 = v14;
  qword_4F91F00 = 0;
  qword_4F91F08 = 0;
  qword_4F91F10 = 0;
  qword_4F91F18 = 0;
  qword_4F91F20 = 0;
  qword_4F91F40 = 0;
  qword_4F91F48 = (__int64)&unk_4F91F60;
  qword_4F91F50 = 1;
  dword_4F91F58 = 0;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4F91F30;
  if ( (unsigned __int64)(unsigned int)qword_4F91F30 + 1 > HIDWORD(qword_4F91F30) )
  {
    v24 = v15;
    sub_C8D5F0((char *)&unk_4F91F38 - 16, &unk_4F91F38, (unsigned int)qword_4F91F30 + 1LL, 8);
    v16 = (unsigned int)qword_4F91F30;
    v15 = v24;
  }
  *(_QWORD *)(qword_4F91F28 + 8 * v16) = v15;
  LODWORD(qword_4F91F30) = qword_4F91F30 + 1;
  byte_4F91F80 = 0;
  qword_4F91F70 = (__int64)&unk_49DE5F0;
  qword_4F91F68 = 0;
  qword_4F91F78 = 0;
  qword_4F91EE0 = (__int64)&unk_49DE610;
  qword_4F91F88 = (__int64)&unk_49DC2F0;
  qword_4F91FA8 = (__int64)nullsub_190;
  qword_4F91FA0 = (__int64)sub_D83E80;
  sub_C53080(&qword_4F91EE0, "cfg-hide-cold-paths", 19);
  qword_4F91F68 = 0;
  qword_4F91F08 = (__int64)"Hide blocks with relative frequency below the given value";
  byte_4F91F80 = 1;
  qword_4F91F78 = 0;
  qword_4F91F10 = 57;
  sub_C53130(&qword_4F91EE0);
  __cxa_atexit(sub_D84280, &qword_4F91EE0, &qword_4A427C0);
  v29 = 23;
  v27 = &v25;
  v28 = "Show heat colors in CFG";
  v26 = 1;
  v25 = 1;
  sub_11F3BE0(&unk_4F91E00, "cfg-heat-colors", &v27);
  __cxa_atexit(sub_984900, &unk_4F91E00, &qword_4A427C0);
  v28 = "Use raw weights for labels. Use percentages as default.";
  v27 = &v25;
  v29 = 55;
  v26 = 1;
  v25 = 0;
  sub_11F3BE0(&unk_4F91D20, "cfg-raw-weights", &v27);
  __cxa_atexit(sub_984900, &unk_4F91D20, &qword_4A427C0);
  qword_4F91C40 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F91CBC = 1;
  qword_4F91C90 = 0x100000000LL;
  dword_4F91C4C &= 0x8000u;
  qword_4F91C88 = (__int64)&unk_4F91C98;
  qword_4F91C58 = 0;
  qword_4F91C60 = 0;
  dword_4F91C48 = v17;
  word_4F91C50 = 0;
  qword_4F91C68 = 0;
  qword_4F91C70 = 0;
  qword_4F91C78 = 0;
  qword_4F91C80 = 0;
  qword_4F91CA0 = 0;
  qword_4F91CA8 = (__int64)&unk_4F91CC0;
  qword_4F91CB0 = 1;
  dword_4F91CB8 = 0;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_4F91C90;
  v20 = (unsigned int)qword_4F91C90 + 1LL;
  if ( v20 > HIDWORD(qword_4F91C90) )
  {
    sub_C8D5F0((char *)&unk_4F91C98 - 16, &unk_4F91C98, v20, 8);
    v19 = (unsigned int)qword_4F91C90;
  }
  *(_QWORD *)(qword_4F91C88 + 8 * v19) = v18;
  qword_4F91CD0 = (__int64)&unk_49D9748;
  qword_4F91C40 = (__int64)&unk_49DC090;
  qword_4F91CE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F91C90) = qword_4F91C90 + 1;
  qword_4F91D00 = (__int64)nullsub_23;
  qword_4F91CC8 = 0;
  qword_4F91CF8 = (__int64)sub_984030;
  qword_4F91CD8 = 0;
  sub_C53080(&qword_4F91C40, "cfg-weights", 11);
  LOBYTE(qword_4F91CC8) = 0;
  LOWORD(qword_4F91CD8) = 256;
  qword_4F91C70 = 31;
  LOBYTE(dword_4F91C4C) = dword_4F91C4C & 0x9F | 0x20;
  qword_4F91C68 = (__int64)"Show edges labeled with weights";
  sub_C53130(&qword_4F91C40);
  return __cxa_atexit(sub_984900, &qword_4F91C40, &qword_4A427C0);
}
