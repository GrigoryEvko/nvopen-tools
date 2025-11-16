// Function: ctor_664
// Address: 0x59dce0
//
int __fastcall ctor_664(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v32; // [rsp+8h] [rbp-38h]

  qword_503AC80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503ACD0 = 0x100000000LL;
  dword_503AC8C &= 0x8000u;
  word_503AC90 = 0;
  qword_503AC98 = 0;
  qword_503ACA0 = 0;
  dword_503AC88 = v4;
  qword_503ACA8 = 0;
  qword_503ACB0 = 0;
  qword_503ACB8 = 0;
  qword_503ACC0 = 0;
  qword_503ACC8 = (__int64)&unk_503ACD8;
  qword_503ACE0 = 0;
  qword_503ACE8 = (__int64)&unk_503AD00;
  qword_503ACF0 = 1;
  dword_503ACF8 = 0;
  byte_503ACFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503ACD0;
  v7 = (unsigned int)qword_503ACD0 + 1LL;
  if ( v7 > HIDWORD(qword_503ACD0) )
  {
    sub_C8D5F0((char *)&unk_503ACD8 - 16, &unk_503ACD8, v7, 8);
    v6 = (unsigned int)qword_503ACD0;
  }
  *(_QWORD *)(qword_503ACC8 + 8 * v6) = v5;
  LODWORD(qword_503ACD0) = qword_503ACD0 + 1;
  qword_503AD08 = 0;
  qword_503AD10 = (__int64)&unk_49DC110;
  qword_503AD18 = 0;
  qword_503AC80 = (__int64)&unk_49D97F0;
  qword_503AD20 = (__int64)&unk_49DC200;
  qword_503AD40 = (__int64)nullsub_26;
  qword_503AD38 = (__int64)sub_9C26D0;
  sub_C53080(&qword_503AC80, "enable-tail-merge", 17);
  LODWORD(qword_503AD08) = 0;
  BYTE4(qword_503AD18) = 1;
  LODWORD(qword_503AD18) = 0;
  LOBYTE(dword_503AC8C) = dword_503AC8C & 0x9F | 0x20;
  sub_C53130(&qword_503AC80);
  __cxa_atexit(sub_9C44F0, &qword_503AC80, &qword_4A427C0);
  qword_503ABA0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_9C44F0, &qword_503AC80, v8, v9), 1u);
  qword_503ABF0 = 0x100000000LL;
  dword_503ABAC &= 0x8000u;
  word_503ABB0 = 0;
  qword_503ABB8 = 0;
  qword_503ABC0 = 0;
  dword_503ABA8 = v10;
  qword_503ABC8 = 0;
  qword_503ABD0 = 0;
  qword_503ABD8 = 0;
  qword_503ABE0 = 0;
  qword_503ABE8 = (__int64)&unk_503ABF8;
  qword_503AC00 = 0;
  qword_503AC08 = (__int64)&unk_503AC20;
  qword_503AC10 = 1;
  dword_503AC18 = 0;
  byte_503AC1C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503ABF0;
  v13 = (unsigned int)qword_503ABF0 + 1LL;
  if ( v13 > HIDWORD(qword_503ABF0) )
  {
    sub_C8D5F0((char *)&unk_503ABF8 - 16, &unk_503ABF8, v13, 8);
    v12 = (unsigned int)qword_503ABF0;
  }
  *(_QWORD *)(qword_503ABE8 + 8 * v12) = v11;
  qword_503AC30 = (__int64)&unk_49D9728;
  qword_503ABA0 = (__int64)&unk_49DBF10;
  LODWORD(qword_503ABF0) = qword_503ABF0 + 1;
  qword_503AC28 = 0;
  qword_503AC40 = (__int64)&unk_49DC290;
  qword_503AC38 = 0;
  qword_503AC60 = (__int64)nullsub_24;
  qword_503AC58 = (__int64)sub_984050;
  sub_C53080(&qword_503ABA0, "tail-merge-threshold", 20);
  qword_503ABD0 = 51;
  qword_503ABC8 = (__int64)"Max number of predecessors to consider tail merging";
  LODWORD(qword_503AC28) = 150;
  BYTE4(qword_503AC38) = 1;
  LODWORD(qword_503AC38) = 150;
  LOBYTE(dword_503ABAC) = dword_503ABAC & 0x9F | 0x20;
  sub_C53130(&qword_503ABA0);
  __cxa_atexit(sub_984970, &qword_503ABA0, &qword_4A427C0);
  qword_503AAC0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503ABA0, v14, v15), 1u);
  qword_503AB10 = 0x100000000LL;
  dword_503AACC &= 0x8000u;
  word_503AAD0 = 0;
  qword_503AAD8 = 0;
  qword_503AAE0 = 0;
  dword_503AAC8 = v16;
  qword_503AAE8 = 0;
  qword_503AAF0 = 0;
  qword_503AAF8 = 0;
  qword_503AB00 = 0;
  qword_503AB08 = (__int64)&unk_503AB18;
  qword_503AB20 = 0;
  qword_503AB28 = (__int64)&unk_503AB40;
  qword_503AB30 = 1;
  dword_503AB38 = 0;
  byte_503AB3C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_503AB10;
  if ( (unsigned __int64)(unsigned int)qword_503AB10 + 1 > HIDWORD(qword_503AB10) )
  {
    v32 = v17;
    sub_C8D5F0((char *)&unk_503AB18 - 16, &unk_503AB18, (unsigned int)qword_503AB10 + 1LL, 8);
    v18 = (unsigned int)qword_503AB10;
    v17 = v32;
  }
  *(_QWORD *)(qword_503AB08 + 8 * v18) = v17;
  qword_503AB50 = (__int64)&unk_49D9728;
  qword_503AAC0 = (__int64)&unk_49DBF10;
  LODWORD(qword_503AB10) = qword_503AB10 + 1;
  qword_503AB48 = 0;
  qword_503AB60 = (__int64)&unk_49DC290;
  qword_503AB58 = 0;
  qword_503AB80 = (__int64)nullsub_24;
  qword_503AB78 = (__int64)sub_984050;
  sub_C53080(&qword_503AAC0, "tail-merge-size", 15);
  qword_503AAF0 = 51;
  qword_503AAE8 = (__int64)"Min number of instructions to consider tail merging";
  LODWORD(qword_503AB48) = 3;
  BYTE4(qword_503AB58) = 1;
  LODWORD(qword_503AB58) = 3;
  LOBYTE(dword_503AACC) = dword_503AACC & 0x9F | 0x20;
  sub_C53130(&qword_503AAC0);
  __cxa_atexit(sub_984970, &qword_503AAC0, &qword_4A427C0);
  qword_503A9E0 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503AAC0, v19, v20), 1u);
  qword_503AA30 = 0x100000000LL;
  dword_503A9EC &= 0x8000u;
  word_503A9F0 = 0;
  qword_503A9F8 = 0;
  qword_503AA00 = 0;
  dword_503A9E8 = v21;
  qword_503AA08 = 0;
  qword_503AA10 = 0;
  qword_503AA18 = 0;
  qword_503AA20 = 0;
  qword_503AA28 = (__int64)&unk_503AA38;
  qword_503AA40 = 0;
  qword_503AA48 = (__int64)&unk_503AA60;
  qword_503AA50 = 1;
  dword_503AA58 = 0;
  byte_503AA5C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_503AA30;
  v24 = (unsigned int)qword_503AA30 + 1LL;
  if ( v24 > HIDWORD(qword_503AA30) )
  {
    sub_C8D5F0((char *)&unk_503AA38 - 16, &unk_503AA38, v24, 8);
    v23 = (unsigned int)qword_503AA30;
  }
  *(_QWORD *)(qword_503AA28 + 8 * v23) = v22;
  qword_503AA70 = (__int64)&unk_49D9748;
  qword_503A9E0 = (__int64)&unk_49DC090;
  LODWORD(qword_503AA30) = qword_503AA30 + 1;
  qword_503AA68 = 0;
  qword_503AA80 = (__int64)&unk_49DC1D0;
  qword_503AA78 = 0;
  qword_503AAA0 = (__int64)nullsub_23;
  qword_503AA98 = (__int64)sub_984030;
  sub_C53080(&qword_503A9E0, "eliminate-redundant-movs", 24);
  LOWORD(qword_503AA78) = 257;
  LOBYTE(qword_503AA68) = 1;
  LOBYTE(dword_503A9EC) = dword_503A9EC & 0x9F | 0x20;
  sub_C53130(&qword_503A9E0);
  __cxa_atexit(sub_984900, &qword_503A9E0, &qword_4A427C0);
  qword_503A900 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503A9E0, v25, v26), 1u);
  qword_503A950 = 0x100000000LL;
  word_503A910 = 0;
  dword_503A90C &= 0x8000u;
  qword_503A918 = 0;
  qword_503A920 = 0;
  dword_503A908 = v27;
  qword_503A928 = 0;
  qword_503A930 = 0;
  qword_503A938 = 0;
  qword_503A940 = 0;
  qword_503A948 = (__int64)&unk_503A958;
  qword_503A960 = 0;
  qword_503A968 = (__int64)&unk_503A980;
  qword_503A970 = 1;
  dword_503A978 = 0;
  byte_503A97C = 1;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_503A950;
  v30 = (unsigned int)qword_503A950 + 1LL;
  if ( v30 > HIDWORD(qword_503A950) )
  {
    sub_C8D5F0((char *)&unk_503A958 - 16, &unk_503A958, v30, 8);
    v29 = (unsigned int)qword_503A950;
  }
  *(_QWORD *)(qword_503A948 + 8 * v29) = v28;
  qword_503A990 = (__int64)&unk_49D9748;
  qword_503A900 = (__int64)&unk_49DC090;
  LODWORD(qword_503A950) = qword_503A950 + 1;
  qword_503A988 = 0;
  qword_503A9A0 = (__int64)&unk_49DC1D0;
  qword_503A998 = 0;
  qword_503A9C0 = (__int64)nullsub_23;
  qword_503A9B8 = (__int64)sub_984030;
  sub_C53080(&qword_503A900, "set-dbg-loc-new-branch", 22);
  LOBYTE(qword_503A988) = 0;
  LOWORD(qword_503A998) = 256;
  LOBYTE(dword_503A90C) = dword_503A90C & 0x9F | 0x20;
  sub_C53130(&qword_503A900);
  return __cxa_atexit(sub_984900, &qword_503A900, &qword_4A427C0);
}
