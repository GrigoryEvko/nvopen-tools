// Function: ctor_514_0
// Address: 0x55e1b0
//
int ctor_514_0()
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
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v18; // [rsp+8h] [rbp-108h]
  __int64 v19; // [rsp+8h] [rbp-108h]
  __int64 v20; // [rsp+8h] [rbp-108h]
  int v21; // [rsp+10h] [rbp-100h] BYREF
  int v22; // [rsp+14h] [rbp-FCh] BYREF
  int *v23; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD v26[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v27; // [rsp+50h] [rbp-C0h]
  const char *v28; // [rsp+58h] [rbp-B8h]
  __int64 v29; // [rsp+60h] [rbp-B0h]
  const char *v30; // [rsp+68h] [rbp-A8h]
  __int64 v31; // [rsp+70h] [rbp-A0h]
  int v32; // [rsp+78h] [rbp-98h]
  const char *v33; // [rsp+80h] [rbp-90h]
  __int64 v34; // [rsp+88h] [rbp-88h]

  qword_500CC40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500CC90 = 0x100000000LL;
  dword_500CC4C &= 0x8000u;
  word_500CC50 = 0;
  qword_500CC58 = 0;
  qword_500CC60 = 0;
  dword_500CC48 = v0;
  qword_500CC68 = 0;
  qword_500CC70 = 0;
  qword_500CC78 = 0;
  qword_500CC80 = 0;
  qword_500CC88 = (__int64)&unk_500CC98;
  qword_500CCA0 = 0;
  qword_500CCA8 = (__int64)&unk_500CCC0;
  qword_500CCB0 = 1;
  dword_500CCB8 = 0;
  byte_500CCBC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500CC90;
  v3 = (unsigned int)qword_500CC90 + 1LL;
  if ( v3 > HIDWORD(qword_500CC90) )
  {
    sub_C8D5F0((char *)&unk_500CC98 - 16, &unk_500CC98, v3, 8);
    v2 = (unsigned int)qword_500CC90;
  }
  *(_QWORD *)(qword_500CC88 + 8 * v2) = v1;
  qword_500CCD0 = (__int64)&unk_49D9748;
  qword_500CC40 = (__int64)&unk_49DC090;
  qword_500CCE0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500CC90) = qword_500CC90 + 1;
  qword_500CD00 = (__int64)nullsub_23;
  qword_500CCC8 = 0;
  qword_500CCF8 = (__int64)sub_984030;
  qword_500CCD8 = 0;
  sub_C53080(&qword_500CC40, "disable-loop-idiom-vectorize-all", 32);
  LOWORD(qword_500CCD8) = 256;
  LOBYTE(qword_500CCC8) = 0;
  qword_500CC70 = 34;
  LOBYTE(dword_500CC4C) = dword_500CC4C & 0x9F | 0x20;
  qword_500CC68 = (__int64)"Disable Loop Idiom Vectorize Pass.";
  sub_C53130(&qword_500CC40);
  __cxa_atexit(sub_984900, &qword_500CC40, &qword_4A427C0);
  v22 = 0;
  v23 = &v22;
  v25[0] = v26;
  v26[0] = "masked";
  v28 = "Use masked vector intrinsics";
  v30 = "predicated";
  v33 = "Use VP intrinsics";
  v25[1] = 0x400000002LL;
  v24[0] = "The vectorization style for loop idiom transform.";
  v26[1] = 6;
  v27 = 0;
  v29 = 28;
  v31 = 10;
  v32 = 1;
  v34 = 17;
  v24[1] = 49;
  v21 = 1;
  sub_2AA7120(&unk_500C9E0, "loop-idiom-vectorize-style", &v21, v24, v25, &v23);
  if ( (_QWORD *)v25[0] != v26 )
    _libc_free(v25[0], "loop-idiom-vectorize-style");
  __cxa_atexit(sub_2A9D150, &unk_500C9E0, &qword_4A427C0);
  qword_500C900 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500C950 = 0x100000000LL;
  dword_500C90C &= 0x8000u;
  qword_500C948 = (__int64)&unk_500C958;
  word_500C910 = 0;
  qword_500C918 = 0;
  dword_500C908 = v4;
  qword_500C920 = 0;
  qword_500C928 = 0;
  qword_500C930 = 0;
  qword_500C938 = 0;
  qword_500C940 = 0;
  qword_500C960 = 0;
  qword_500C968 = (__int64)&unk_500C980;
  qword_500C970 = 1;
  dword_500C978 = 0;
  byte_500C97C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500C950;
  if ( (unsigned __int64)(unsigned int)qword_500C950 + 1 > HIDWORD(qword_500C950) )
  {
    v18 = v5;
    sub_C8D5F0((char *)&unk_500C958 - 16, &unk_500C958, (unsigned int)qword_500C950 + 1LL, 8);
    v6 = (unsigned int)qword_500C950;
    v5 = v18;
  }
  *(_QWORD *)(qword_500C948 + 8 * v6) = v5;
  qword_500C990 = (__int64)&unk_49D9748;
  qword_500C900 = (__int64)&unk_49DC090;
  qword_500C9A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500C950) = qword_500C950 + 1;
  qword_500C9C0 = (__int64)nullsub_23;
  qword_500C988 = 0;
  qword_500C9B8 = (__int64)sub_984030;
  qword_500C998 = 0;
  sub_C53080(&qword_500C900, "disable-loop-idiom-vectorize-bytecmp", 36);
  LOBYTE(qword_500C988) = 0;
  LOWORD(qword_500C998) = 256;
  qword_500C930 = 80;
  LOBYTE(dword_500C90C) = dword_500C90C & 0x9F | 0x20;
  qword_500C928 = (__int64)"Proceed with Loop Idiom Vectorize Pass, but do not convert byte-compare loop(s).";
  sub_C53130(&qword_500C900);
  __cxa_atexit(sub_984900, &qword_500C900, &qword_4A427C0);
  qword_500C820 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500C870 = 0x100000000LL;
  dword_500C82C &= 0x8000u;
  word_500C830 = 0;
  qword_500C868 = (__int64)&unk_500C878;
  qword_500C838 = 0;
  dword_500C828 = v7;
  qword_500C840 = 0;
  qword_500C848 = 0;
  qword_500C850 = 0;
  qword_500C858 = 0;
  qword_500C860 = 0;
  qword_500C880 = 0;
  qword_500C888 = (__int64)&unk_500C8A0;
  qword_500C890 = 1;
  dword_500C898 = 0;
  byte_500C89C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_500C870;
  if ( (unsigned __int64)(unsigned int)qword_500C870 + 1 > HIDWORD(qword_500C870) )
  {
    v19 = v8;
    sub_C8D5F0((char *)&unk_500C878 - 16, &unk_500C878, (unsigned int)qword_500C870 + 1LL, 8);
    v9 = (unsigned int)qword_500C870;
    v8 = v19;
  }
  *(_QWORD *)(qword_500C868 + 8 * v9) = v8;
  LODWORD(qword_500C870) = qword_500C870 + 1;
  qword_500C8A8 = 0;
  qword_500C8B0 = (__int64)&unk_49D9728;
  qword_500C8B8 = 0;
  qword_500C820 = (__int64)&unk_49DBF10;
  qword_500C8C0 = (__int64)&unk_49DC290;
  qword_500C8E0 = (__int64)nullsub_24;
  qword_500C8D8 = (__int64)sub_984050;
  sub_C53080(&qword_500C820, "loop-idiom-vectorize-bytecmp-vf", 31);
  qword_500C850 = 51;
  LODWORD(qword_500C8A8) = 16;
  BYTE4(qword_500C8B8) = 1;
  LODWORD(qword_500C8B8) = 16;
  LOBYTE(dword_500C82C) = dword_500C82C & 0x9F | 0x20;
  qword_500C848 = (__int64)"The vectorization factor for byte-compare patterns.";
  sub_C53130(&qword_500C820);
  __cxa_atexit(sub_984970, &qword_500C820, &qword_4A427C0);
  qword_500C740 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500C7BC = 1;
  word_500C750 = 0;
  qword_500C790 = 0x100000000LL;
  dword_500C74C &= 0x8000u;
  qword_500C788 = (__int64)&unk_500C798;
  qword_500C758 = 0;
  dword_500C748 = v10;
  qword_500C760 = 0;
  qword_500C768 = 0;
  qword_500C770 = 0;
  qword_500C778 = 0;
  qword_500C780 = 0;
  qword_500C7A0 = 0;
  qword_500C7A8 = (__int64)&unk_500C7C0;
  qword_500C7B0 = 1;
  dword_500C7B8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_500C790;
  if ( (unsigned __int64)(unsigned int)qword_500C790 + 1 > HIDWORD(qword_500C790) )
  {
    v20 = v11;
    sub_C8D5F0((char *)&unk_500C798 - 16, &unk_500C798, (unsigned int)qword_500C790 + 1LL, 8);
    v12 = (unsigned int)qword_500C790;
    v11 = v20;
  }
  *(_QWORD *)(qword_500C788 + 8 * v12) = v11;
  qword_500C7D0 = (__int64)&unk_49D9748;
  qword_500C740 = (__int64)&unk_49DC090;
  qword_500C7E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500C790) = qword_500C790 + 1;
  qword_500C800 = (__int64)nullsub_23;
  qword_500C7C8 = 0;
  qword_500C7F8 = (__int64)sub_984030;
  qword_500C7D8 = 0;
  sub_C53080(&qword_500C740, "disable-loop-idiom-vectorize-find-first-byte", 44);
  LOWORD(qword_500C7D8) = 256;
  LOBYTE(qword_500C7C8) = 0;
  qword_500C770 = 39;
  LOBYTE(dword_500C74C) = dword_500C74C & 0x9F | 0x20;
  qword_500C768 = (__int64)"Do not convert find-first-byte loop(s).";
  sub_C53130(&qword_500C740);
  __cxa_atexit(sub_984900, &qword_500C740, &qword_4A427C0);
  qword_500C660 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500C6B0 = 0x100000000LL;
  dword_500C66C &= 0x8000u;
  word_500C670 = 0;
  qword_500C6A8 = (__int64)&unk_500C6B8;
  qword_500C678 = 0;
  dword_500C668 = v13;
  qword_500C680 = 0;
  qword_500C688 = 0;
  qword_500C690 = 0;
  qword_500C698 = 0;
  qword_500C6A0 = 0;
  qword_500C6C0 = 0;
  qword_500C6C8 = (__int64)&unk_500C6E0;
  qword_500C6D0 = 1;
  dword_500C6D8 = 0;
  byte_500C6DC = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_500C6B0;
  v16 = (unsigned int)qword_500C6B0 + 1LL;
  if ( v16 > HIDWORD(qword_500C6B0) )
  {
    sub_C8D5F0((char *)&unk_500C6B8 - 16, &unk_500C6B8, v16, 8);
    v15 = (unsigned int)qword_500C6B0;
  }
  *(_QWORD *)(qword_500C6A8 + 8 * v15) = v14;
  qword_500C6F0 = (__int64)&unk_49D9748;
  qword_500C660 = (__int64)&unk_49DC090;
  qword_500C700 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500C6B0) = qword_500C6B0 + 1;
  qword_500C720 = (__int64)nullsub_23;
  qword_500C6E8 = 0;
  qword_500C718 = (__int64)sub_984030;
  qword_500C6F8 = 0;
  sub_C53080(&qword_500C660, "loop-idiom-vectorize-verify", 27);
  LOBYTE(qword_500C6E8) = 0;
  qword_500C690 = 49;
  LOBYTE(dword_500C66C) = dword_500C66C & 0x9F | 0x20;
  LOWORD(qword_500C6F8) = 256;
  qword_500C688 = (__int64)"Verify loops generated Loop Idiom Vectorize Pass.";
  sub_C53130(&qword_500C660);
  return __cxa_atexit(sub_984900, &qword_500C660, &qword_4A427C0);
}
