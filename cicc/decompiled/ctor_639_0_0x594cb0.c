// Function: ctor_639_0
// Address: 0x594cb0
//
int __fastcall ctor_639_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v25; // [rsp+8h] [rbp-108h]
  __int64 v26; // [rsp+8h] [rbp-108h]
  int v27; // [rsp+10h] [rbp-100h] BYREF
  int v28; // [rsp+14h] [rbp-FCh] BYREF
  int *v29; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v30[2]; // [rsp+20h] [rbp-F0h] BYREF
  const char *v31; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v32; // [rsp+38h] [rbp-D8h]
  _QWORD v33[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v34; // [rsp+50h] [rbp-C0h]
  const char *v35; // [rsp+58h] [rbp-B8h]
  __int64 v36; // [rsp+60h] [rbp-B0h]
  const char *v37; // [rsp+68h] [rbp-A8h]
  __int64 v38; // [rsp+70h] [rbp-A0h]
  int v39; // [rsp+78h] [rbp-98h]
  const char *v40; // [rsp+80h] [rbp-90h]
  __int64 v41; // [rsp+88h] [rbp-88h]
  char *v42; // [rsp+90h] [rbp-80h]
  __int64 v43; // [rsp+98h] [rbp-78h]
  int v44; // [rsp+A0h] [rbp-70h]
  const char *v45; // [rsp+A8h] [rbp-68h]
  __int64 v46; // [rsp+B0h] [rbp-60h]

  qword_5035560 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_50355B0 = 0x100000000LL;
  dword_503556C &= 0x8000u;
  word_5035570 = 0;
  qword_5035578 = 0;
  qword_5035580 = 0;
  dword_5035568 = v4;
  qword_5035588 = 0;
  qword_5035590 = 0;
  qword_5035598 = 0;
  qword_50355A0 = 0;
  qword_50355A8 = (__int64)&unk_50355B8;
  qword_50355C0 = 0;
  qword_50355C8 = (__int64)&unk_50355E0;
  qword_50355D0 = 1;
  dword_50355D8 = 0;
  byte_50355DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50355B0;
  v7 = (unsigned int)qword_50355B0 + 1LL;
  if ( v7 > HIDWORD(qword_50355B0) )
  {
    sub_C8D5F0((char *)&unk_50355B8 - 16, &unk_50355B8, v7, 8);
    v6 = (unsigned int)qword_50355B0;
  }
  *(_QWORD *)(qword_50355A8 + 8 * v6) = v5;
  LODWORD(qword_50355B0) = qword_50355B0 + 1;
  qword_50355E8 = 0;
  qword_50355F0 = (__int64)&unk_49D9748;
  qword_50355F8 = 0;
  qword_5035560 = (__int64)&unk_49DC090;
  qword_5035600 = (__int64)&unk_49DC1D0;
  qword_5035620 = (__int64)nullsub_23;
  qword_5035618 = (__int64)sub_984030;
  sub_C53080(&qword_5035560, "enable-if-conversion", 20);
  LOWORD(qword_50355F8) = 257;
  LOBYTE(qword_50355E8) = 1;
  qword_5035590 = 42;
  LOBYTE(dword_503556C) = dword_503556C & 0x9F | 0x20;
  qword_5035588 = (__int64)"Enable if-conversion during vectorization.";
  sub_C53130(&qword_5035560);
  __cxa_atexit(sub_984900, &qword_5035560, &qword_4A427C0);
  v32 = 71;
  v31 = "Enable recognition of non-constant strided pointer induction variables.";
  LODWORD(v29) = 1;
  LOBYTE(v28) = 0;
  v30[0] = &v28;
  sub_23A18A0(&unk_5035480, "lv-strided-pointer-ivs", v30, &v29, &v31);
  __cxa_atexit(sub_984900, &unk_5035480, &qword_4A427C0);
  v32 = 72;
  v31 = "Allow enabling loop hints to reorder FP operations during vectorization.";
  LODWORD(v29) = 1;
  LOBYTE(v28) = 1;
  v30[0] = &v28;
  sub_23A18A0(&unk_50353A0, "hints-allow-reordering", v30, &v29, &v31);
  __cxa_atexit(sub_984900, &unk_50353A0, &qword_4A427C0);
  qword_50352C0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_50353A0, v8, v9), 1u);
  qword_5035310 = 0x100000000LL;
  dword_50352CC &= 0x8000u;
  qword_5035308 = (__int64)&unk_5035318;
  word_50352D0 = 0;
  qword_50352D8 = 0;
  dword_50352C8 = v10;
  qword_50352E0 = 0;
  qword_50352E8 = 0;
  qword_50352F0 = 0;
  qword_50352F8 = 0;
  qword_5035300 = 0;
  qword_5035320 = 0;
  qword_5035328 = (__int64)&unk_5035340;
  qword_5035330 = 1;
  dword_5035338 = 0;
  byte_503533C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5035310;
  if ( (unsigned __int64)(unsigned int)qword_5035310 + 1 > HIDWORD(qword_5035310) )
  {
    v25 = v11;
    sub_C8D5F0((char *)&unk_5035318 - 16, &unk_5035318, (unsigned int)qword_5035310 + 1LL, 8);
    v12 = (unsigned int)qword_5035310;
    v11 = v25;
  }
  *(_QWORD *)(qword_5035308 + 8 * v12) = v11;
  LODWORD(qword_5035310) = qword_5035310 + 1;
  qword_5035348 = 0;
  qword_5035350 = (__int64)&unk_49D9728;
  qword_5035358 = 0;
  qword_50352C0 = (__int64)&unk_49DBF10;
  qword_5035360 = (__int64)&unk_49DC290;
  qword_5035380 = (__int64)nullsub_24;
  qword_5035378 = (__int64)sub_984050;
  sub_C53080(&qword_50352C0, "vectorize-scev-check-threshold", 30);
  LODWORD(qword_5035348) = 16;
  BYTE4(qword_5035358) = 1;
  LODWORD(qword_5035358) = 16;
  qword_50352F0 = 42;
  LOBYTE(dword_50352CC) = dword_50352CC & 0x9F | 0x20;
  qword_50352E8 = (__int64)"The maximum number of SCEV checks allowed.";
  sub_C53130(&qword_50352C0);
  __cxa_atexit(sub_984970, &qword_50352C0, &qword_4A427C0);
  qword_50351E0 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_50352C0, v13, v14), 1u);
  qword_5035230 = 0x100000000LL;
  dword_50351EC &= 0x8000u;
  word_50351F0 = 0;
  qword_5035228 = (__int64)&unk_5035238;
  qword_50351F8 = 0;
  dword_50351E8 = v15;
  qword_5035200 = 0;
  qword_5035208 = 0;
  qword_5035210 = 0;
  qword_5035218 = 0;
  qword_5035220 = 0;
  qword_5035240 = 0;
  qword_5035248 = (__int64)&unk_5035260;
  qword_5035250 = 1;
  dword_5035258 = 0;
  byte_503525C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5035230;
  if ( (unsigned __int64)(unsigned int)qword_5035230 + 1 > HIDWORD(qword_5035230) )
  {
    v26 = v16;
    sub_C8D5F0((char *)&unk_5035238 - 16, &unk_5035238, (unsigned int)qword_5035230 + 1LL, 8);
    v17 = (unsigned int)qword_5035230;
    v16 = v26;
  }
  *(_QWORD *)(qword_5035228 + 8 * v17) = v16;
  LODWORD(qword_5035230) = qword_5035230 + 1;
  qword_5035268 = 0;
  qword_5035270 = (__int64)&unk_49D9728;
  qword_5035278 = 0;
  qword_50351E0 = (__int64)&unk_49DBF10;
  qword_5035280 = (__int64)&unk_49DC290;
  qword_50352A0 = (__int64)nullsub_24;
  qword_5035298 = (__int64)sub_984050;
  sub_C53080(&qword_50351E0, "pragma-vectorize-scev-check-threshold", 37);
  LODWORD(qword_5035268) = 128;
  BYTE4(qword_5035278) = 1;
  LODWORD(qword_5035278) = 128;
  qword_5035210 = 73;
  LOBYTE(dword_50351EC) = dword_50351EC & 0x9F | 0x20;
  qword_5035208 = (__int64)"The maximum number of SCEV checks allowed with a vectorize(enable) pragma";
  sub_C53130(&qword_50351E0);
  __cxa_atexit(sub_984970, &qword_50351E0, &qword_4A427C0);
  v33[1] = 3;
  v31 = (const char *)v33;
  v33[0] = "off";
  v35 = "Scalable vectorization is disabled.";
  v37 = "preferred";
  v40 = "Scalable vectorization is available and favored when the cost is inconclusive.";
  v42 = "on";
  v45 = "Scalable vectorization is available and favored when the cost is inconclusive.";
  v32 = 0x400000003LL;
  v29 = &v27;
  v30[0] = "Control whether the compiler can use scalable vectors to vectorize a loop";
  v34 = 0;
  v36 = 35;
  v38 = 9;
  v39 = 1;
  v41 = 78;
  v43 = 2;
  v44 = 1;
  v46 = 78;
  v30[1] = 73;
  v28 = 1;
  v27 = -1;
  sub_31AF440(&unk_5034F80, "scalable-vectorization", &v29, &v28, v30, &v31);
  if ( v31 != (const char *)v33 )
    _libc_free(v31, "scalable-vectorization");
  __cxa_atexit(sub_31A40D0, &unk_5034F80, &qword_4A427C0);
  qword_5034EA0 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_31A40D0, &unk_5034F80, v18, v19), 1u);
  byte_5034F1C = 1;
  qword_5034EF0 = 0x100000000LL;
  dword_5034EAC &= 0x8000u;
  qword_5034EB8 = 0;
  qword_5034EC0 = 0;
  qword_5034EC8 = 0;
  dword_5034EA8 = v20;
  word_5034EB0 = 0;
  qword_5034ED0 = 0;
  qword_5034ED8 = 0;
  qword_5034EE0 = 0;
  qword_5034EE8 = (__int64)&unk_5034EF8;
  qword_5034F00 = 0;
  qword_5034F08 = (__int64)&unk_5034F20;
  qword_5034F10 = 1;
  dword_5034F18 = 0;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5034EF0;
  v23 = (unsigned int)qword_5034EF0 + 1LL;
  if ( v23 > HIDWORD(qword_5034EF0) )
  {
    sub_C8D5F0((char *)&unk_5034EF8 - 16, &unk_5034EF8, v23, 8);
    v22 = (unsigned int)qword_5034EF0;
  }
  *(_QWORD *)(qword_5034EE8 + 8 * v22) = v21;
  LODWORD(qword_5034EF0) = qword_5034EF0 + 1;
  qword_5034F28 = 0;
  qword_5034F30 = (__int64)&unk_49D9748;
  qword_5034F38 = 0;
  qword_5034EA0 = (__int64)&unk_49DC090;
  qword_5034F40 = (__int64)&unk_49DC1D0;
  qword_5034F60 = (__int64)nullsub_23;
  qword_5034F58 = (__int64)sub_984030;
  sub_C53080(&qword_5034EA0, "enable-histogram-loop-vectorization", 35);
  LOBYTE(qword_5034F28) = 0;
  LOWORD(qword_5034F38) = 256;
  qword_5034ED0 = 61;
  LOBYTE(dword_5034EAC) = dword_5034EAC & 0x9F | 0x20;
  qword_5034EC8 = (__int64)"Enables autovectorization of some loops containing histograms";
  sub_C53130(&qword_5034EA0);
  return __cxa_atexit(sub_984900, &qword_5034EA0, &qword_4A427C0);
}
