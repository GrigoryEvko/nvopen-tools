// Function: ctor_597
// Address: 0x57dd10
//
int __fastcall ctor_597(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  int v27; // [rsp+24h] [rbp-4Ch] BYREF
  int *v28; // [rsp+28h] [rbp-48h]
  const char *v29; // [rsp+30h] [rbp-40h] BYREF
  __int64 v30; // [rsp+38h] [rbp-38h]

  qword_5026880 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_50268D0 = 0x100000000LL;
  dword_502688C &= 0x8000u;
  word_5026890 = 0;
  qword_5026898 = 0;
  qword_50268A0 = 0;
  dword_5026888 = v4;
  qword_50268A8 = 0;
  qword_50268B0 = 0;
  qword_50268B8 = 0;
  qword_50268C0 = 0;
  qword_50268C8 = (__int64)&unk_50268D8;
  qword_50268E0 = 0;
  qword_50268E8 = (__int64)&unk_5026900;
  qword_50268F0 = 1;
  dword_50268F8 = 0;
  byte_50268FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50268D0;
  v7 = (unsigned int)qword_50268D0 + 1LL;
  if ( v7 > HIDWORD(qword_50268D0) )
  {
    sub_C8D5F0((char *)&unk_50268D8 - 16, &unk_50268D8, v7, 8);
    v6 = (unsigned int)qword_50268D0;
  }
  *(_QWORD *)(qword_50268C8 + 8 * v6) = v5;
  qword_5026910 = (__int64)&unk_49D9728;
  qword_5026880 = (__int64)&unk_49DBF10;
  qword_5026920 = (__int64)&unk_49DC290;
  LODWORD(qword_50268D0) = qword_50268D0 + 1;
  qword_5026940 = (__int64)nullsub_24;
  qword_5026908 = 0;
  qword_5026938 = (__int64)sub_984050;
  qword_5026918 = 0;
  sub_C53080(&qword_5026880, "tail-dup-size", 13);
  qword_50268B0 = 49;
  qword_50268A8 = (__int64)"Maximum instructions to consider tail duplicating";
  LODWORD(qword_5026908) = 2;
  BYTE4(qword_5026918) = 1;
  LODWORD(qword_5026918) = 2;
  LOBYTE(dword_502688C) = dword_502688C & 0x9F | 0x20;
  sub_C53130(&qword_5026880);
  __cxa_atexit(sub_984970, &qword_5026880, &qword_4A427C0);
  qword_50267A0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5026880, v8, v9), 1u);
  byte_502681C = 1;
  word_50267B0 = 0;
  qword_50267F0 = 0x100000000LL;
  dword_50267AC &= 0x8000u;
  qword_50267E8 = (__int64)&unk_50267F8;
  qword_50267B8 = 0;
  dword_50267A8 = v10;
  qword_50267C0 = 0;
  qword_50267C8 = 0;
  qword_50267D0 = 0;
  qword_50267D8 = 0;
  qword_50267E0 = 0;
  qword_5026800 = 0;
  qword_5026808 = (__int64)&unk_5026820;
  qword_5026810 = 1;
  dword_5026818 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50267F0;
  if ( (unsigned __int64)(unsigned int)qword_50267F0 + 1 > HIDWORD(qword_50267F0) )
  {
    v25 = v11;
    sub_C8D5F0((char *)&unk_50267F8 - 16, &unk_50267F8, (unsigned int)qword_50267F0 + 1LL, 8);
    v12 = (unsigned int)qword_50267F0;
    v11 = v25;
  }
  *(_QWORD *)(qword_50267E8 + 8 * v12) = v11;
  qword_5026830 = (__int64)&unk_49D9728;
  qword_50267A0 = (__int64)&unk_49DBF10;
  qword_5026840 = (__int64)&unk_49DC290;
  LODWORD(qword_50267F0) = qword_50267F0 + 1;
  qword_5026860 = (__int64)nullsub_24;
  qword_5026828 = 0;
  qword_5026858 = (__int64)sub_984050;
  qword_5026838 = 0;
  sub_C53080(&qword_50267A0, "tail-dup-indirect-size", 22);
  qword_50267D0 = 89;
  qword_50267C8 = (__int64)"Maximum instructions to consider tail duplicating blocks that end with indirect branches.";
  LODWORD(qword_5026828) = 20;
  BYTE4(qword_5026838) = 1;
  LODWORD(qword_5026838) = 20;
  LOBYTE(dword_50267AC) = dword_50267AC & 0x9F | 0x20;
  sub_C53130(&qword_50267A0);
  __cxa_atexit(sub_984970, &qword_50267A0, &qword_4A427C0);
  v28 = &v27;
  v29 = "Maximum predecessors (maximum successors at the same time) to consider tail duplicating blocks.";
  v27 = 16;
  v30 = 95;
  sub_2FD7790(&unk_50266C0, "tail-dup-pred-size", &v29);
  __cxa_atexit(sub_984970, &unk_50266C0, &qword_4A427C0);
  v29 = "Maximum successors (maximum predecessors at the same time) to consider tail duplicating blocks.";
  v28 = &v27;
  v27 = 16;
  v30 = 95;
  sub_2FD7790(&unk_50265E0, "tail-dup-succ-size", &v29);
  __cxa_atexit(sub_984970, &unk_50265E0, &qword_4A427C0);
  qword_5026500 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &unk_50265E0, v13, v14), 1u);
  qword_5026550 = 0x100000000LL;
  dword_502650C &= 0x8000u;
  word_5026510 = 0;
  qword_5026548 = (__int64)&unk_5026558;
  qword_5026518 = 0;
  dword_5026508 = v15;
  qword_5026520 = 0;
  qword_5026528 = 0;
  qword_5026530 = 0;
  qword_5026538 = 0;
  qword_5026540 = 0;
  qword_5026560 = 0;
  qword_5026568 = (__int64)&unk_5026580;
  qword_5026570 = 1;
  dword_5026578 = 0;
  byte_502657C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5026550;
  if ( (unsigned __int64)(unsigned int)qword_5026550 + 1 > HIDWORD(qword_5026550) )
  {
    v26 = v16;
    sub_C8D5F0((char *)&unk_5026558 - 16, &unk_5026558, (unsigned int)qword_5026550 + 1LL, 8);
    v17 = (unsigned int)qword_5026550;
    v16 = v26;
  }
  *(_QWORD *)(qword_5026548 + 8 * v17) = v16;
  LODWORD(qword_5026550) = qword_5026550 + 1;
  qword_5026588 = 0;
  qword_5026590 = (__int64)&unk_49D9748;
  qword_5026598 = 0;
  qword_5026500 = (__int64)&unk_49DC090;
  qword_50265A0 = (__int64)&unk_49DC1D0;
  qword_50265C0 = (__int64)nullsub_23;
  qword_50265B8 = (__int64)sub_984030;
  sub_C53080(&qword_5026500, "tail-dup-verify", 15);
  qword_5026530 = 48;
  qword_5026528 = (__int64)"Verify sanity of PHI instructions during taildup";
  LOWORD(qword_5026598) = 256;
  LOBYTE(qword_5026588) = 0;
  LOBYTE(dword_502650C) = dword_502650C & 0x9F | 0x20;
  sub_C53130(&qword_5026500);
  __cxa_atexit(sub_984900, &qword_5026500, &qword_4A427C0);
  qword_5026420 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5026500, v18, v19), 1u);
  byte_502649C = 1;
  qword_5026470 = 0x100000000LL;
  dword_502642C &= 0x8000u;
  qword_5026468 = (__int64)&unk_5026478;
  qword_5026438 = 0;
  qword_5026440 = 0;
  dword_5026428 = v20;
  word_5026430 = 0;
  qword_5026448 = 0;
  qword_5026450 = 0;
  qword_5026458 = 0;
  qword_5026460 = 0;
  qword_5026480 = 0;
  qword_5026488 = (__int64)&unk_50264A0;
  qword_5026490 = 1;
  dword_5026498 = 0;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5026470;
  v23 = (unsigned int)qword_5026470 + 1LL;
  if ( v23 > HIDWORD(qword_5026470) )
  {
    sub_C8D5F0((char *)&unk_5026478 - 16, &unk_5026478, v23, 8);
    v22 = (unsigned int)qword_5026470;
  }
  *(_QWORD *)(qword_5026468 + 8 * v22) = v21;
  qword_50264B0 = (__int64)&unk_49D9728;
  qword_5026420 = (__int64)&unk_49DBF10;
  qword_50264C0 = (__int64)&unk_49DC290;
  LODWORD(qword_5026470) = qword_5026470 + 1;
  qword_50264E0 = (__int64)nullsub_24;
  qword_50264A8 = 0;
  qword_50264D8 = (__int64)sub_984050;
  qword_50264B8 = 0;
  sub_C53080(&qword_5026420, "tail-dup-limit", 14);
  LODWORD(qword_50264A8) = -1;
  BYTE4(qword_50264B8) = 1;
  LODWORD(qword_50264B8) = -1;
  LOBYTE(dword_502642C) = dword_502642C & 0x9F | 0x20;
  sub_C53130(&qword_5026420);
  return __cxa_atexit(sub_984970, &qword_5026420, &qword_4A427C0);
}
