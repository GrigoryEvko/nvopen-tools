// Function: ctor_633_0
// Address: 0x592410
//
int ctor_633_0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+8h] [rbp-68h]
  int v25; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v27; // [rsp+28h] [rbp-48h] BYREF
  const char *v28; // [rsp+30h] [rbp-40h] BYREF
  __int64 v29; // [rsp+38h] [rbp-38h]

  v28 = "Minimum number of similar functions with the same hash required for merging.";
  LODWORD(v26) = 1;
  v25 = 2;
  v27 = (__int64 *)&v25;
  v29 = 76;
  sub_311AAE0(&unk_5032840, "global-merging-min-merges", &v28, &v27, &v26);
  __cxa_atexit(sub_984970, &unk_5032840, &qword_4A427C0);
  LODWORD(v26) = 1;
  v28 = "The minimum instruction count required when merging functions.";
  v25 = 1;
  v27 = (__int64 *)&v25;
  v29 = 62;
  sub_311AAE0(&unk_5032760, "global-merging-min-instrs", &v28, &v27, &v26);
  __cxa_atexit(sub_984970, &unk_5032760, &qword_4A427C0);
  qword_5032680 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &unk_5032760, v0, v1), 1u);
  qword_50326D0 = 0x100000000LL;
  dword_503268C &= 0x8000u;
  word_5032690 = 0;
  qword_5032698 = 0;
  qword_50326A0 = 0;
  dword_5032688 = v2;
  qword_50326A8 = 0;
  qword_50326B0 = 0;
  qword_50326B8 = 0;
  qword_50326C0 = 0;
  qword_50326C8 = (__int64)&unk_50326D8;
  qword_50326E0 = 0;
  qword_50326E8 = (__int64)&unk_5032700;
  qword_50326F0 = 1;
  dword_50326F8 = 0;
  byte_50326FC = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_50326D0;
  if ( (unsigned __int64)(unsigned int)qword_50326D0 + 1 > HIDWORD(qword_50326D0) )
  {
    v23 = v3;
    sub_C8D5F0((char *)&unk_50326D8 - 16, &unk_50326D8, (unsigned int)qword_50326D0 + 1LL, 8);
    v4 = (unsigned int)qword_50326D0;
    v3 = v23;
  }
  *(_QWORD *)(qword_50326C8 + 8 * v4) = v3;
  LODWORD(qword_50326D0) = qword_50326D0 + 1;
  qword_5032708 = 0;
  qword_5032710 = (__int64)&unk_49D9728;
  qword_5032718 = 0;
  qword_5032680 = (__int64)&unk_49DBF10;
  qword_5032720 = (__int64)&unk_49DC290;
  qword_5032740 = (__int64)nullsub_24;
  qword_5032738 = (__int64)sub_984050;
  sub_C53080(&qword_5032680, "global-merging-max-params", 25);
  qword_50326B0 = 64;
  qword_50326A8 = (__int64)"The maximum number of parameters allowed when merging functions.";
  LODWORD(qword_5032708) = -1;
  BYTE4(qword_5032718) = 1;
  LODWORD(qword_5032718) = -1;
  LOBYTE(dword_503268C) = dword_503268C & 0x9F | 0x20;
  sub_C53130(&qword_5032680);
  __cxa_atexit(sub_984970, &qword_5032680, &qword_4A427C0);
  qword_50325A0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5032680, v5, v6), 1u);
  byte_503261C = 1;
  word_50325B0 = 0;
  qword_50325F0 = 0x100000000LL;
  dword_50325AC &= 0x8000u;
  qword_50325E8 = (__int64)&unk_50325F8;
  qword_50325B8 = 0;
  dword_50325A8 = v7;
  qword_50325C0 = 0;
  qword_50325C8 = 0;
  qword_50325D0 = 0;
  qword_50325D8 = 0;
  qword_50325E0 = 0;
  qword_5032600 = 0;
  qword_5032608 = (__int64)&unk_5032620;
  qword_5032610 = 1;
  dword_5032618 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_50325F0;
  v10 = (unsigned int)qword_50325F0 + 1LL;
  if ( v10 > HIDWORD(qword_50325F0) )
  {
    sub_C8D5F0((char *)&unk_50325F8 - 16, &unk_50325F8, v10, 8);
    v9 = (unsigned int)qword_50325F0;
  }
  *(_QWORD *)(qword_50325E8 + 8 * v9) = v8;
  LODWORD(qword_50325F0) = qword_50325F0 + 1;
  qword_5032628 = 0;
  qword_5032630 = (__int64)&unk_49D9748;
  qword_5032638 = 0;
  qword_50325A0 = (__int64)&unk_49DC090;
  qword_5032640 = (__int64)&unk_49DC1D0;
  qword_5032660 = (__int64)nullsub_23;
  qword_5032658 = (__int64)sub_984030;
  sub_C53080(&qword_50325A0, "global-merging-skip-no-params", 29);
  qword_50325C8 = (__int64)"Skip merging functions with no parameters.";
  LOWORD(qword_5032638) = 257;
  LOBYTE(qword_5032628) = 1;
  qword_50325D0 = 42;
  LOBYTE(dword_50325AC) = dword_50325AC & 0x9F | 0x20;
  sub_C53130(&qword_50325A0);
  __cxa_atexit(sub_984900, &qword_50325A0, &qword_4A427C0);
  v27 = &v26;
  v25 = 1;
  v26 = 0x3FF3333333333333LL;
  v28 = "The overhead cost associated with each instruction when lowering to machine instruction.";
  v29 = 88;
  sub_311AD00(&unk_50324C0, "global-merging-inst-overhead", &v28, &v27, &v25);
  __cxa_atexit(sub_D84280, &unk_50324C0, &qword_4A427C0);
  qword_50323E0 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_D84280, &unk_50324C0, v11, v12), 1u);
  qword_5032430 = 0x100000000LL;
  dword_50323EC &= 0x8000u;
  word_50323F0 = 0;
  qword_5032428 = (__int64)&unk_5032438;
  qword_50323F8 = 0;
  dword_50323E8 = v13;
  qword_5032400 = 0;
  qword_5032408 = 0;
  qword_5032410 = 0;
  qword_5032418 = 0;
  qword_5032420 = 0;
  qword_5032440 = 0;
  qword_5032448 = (__int64)&unk_5032460;
  qword_5032450 = 1;
  dword_5032458 = 0;
  byte_503245C = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_5032430;
  if ( (unsigned __int64)(unsigned int)qword_5032430 + 1 > HIDWORD(qword_5032430) )
  {
    v24 = v14;
    sub_C8D5F0((char *)&unk_5032438 - 16, &unk_5032438, (unsigned int)qword_5032430 + 1LL, 8);
    v15 = (unsigned int)qword_5032430;
    v14 = v24;
  }
  *(_QWORD *)(qword_5032428 + 8 * v15) = v14;
  LODWORD(qword_5032430) = qword_5032430 + 1;
  byte_5032480 = 0;
  qword_5032470 = (__int64)&unk_49DE5F0;
  qword_5032468 = 0;
  qword_5032478 = 0;
  qword_50323E0 = (__int64)&unk_49DE610;
  qword_5032488 = (__int64)&unk_49DC2F0;
  qword_50324A8 = (__int64)nullsub_190;
  qword_50324A0 = (__int64)sub_D83E80;
  sub_C53080(&qword_50323E0, "global-merging-param-overhead", 29);
  qword_5032408 = (__int64)"The overhead cost associated with each parameter when merging functions.";
  qword_5032468 = 0x4000000000000000LL;
  qword_5032478 = 0x4000000000000000LL;
  byte_5032480 = 1;
  LOBYTE(dword_50323EC) = dword_50323EC & 0x9F | 0x20;
  qword_5032410 = 72;
  sub_C53130(&qword_50323E0);
  __cxa_atexit(sub_D84280, &qword_50323E0, &qword_4A427C0);
  v27 = &v26;
  v25 = 1;
  v26 = 0x3FF0000000000000LL;
  v28 = "The overhead cost associated with each function call when merging functions.";
  v29 = 76;
  sub_311AD00(&unk_5032300, "global-merging-call-overhead", &v28, &v27, &v25);
  __cxa_atexit(sub_D84280, &unk_5032300, &qword_4A427C0);
  qword_5032220 = (__int64)&unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_D84280, &unk_5032300, v16, v17), 1u);
  dword_503222C &= 0x8000u;
  word_5032230 = 0;
  qword_5032270 = 0x100000000LL;
  qword_5032238 = 0;
  qword_5032240 = 0;
  qword_5032248 = 0;
  dword_5032228 = v18;
  qword_5032250 = 0;
  qword_5032258 = 0;
  qword_5032260 = 0;
  qword_5032268 = (__int64)&unk_5032278;
  qword_5032280 = 0;
  qword_5032288 = (__int64)&unk_50322A0;
  qword_5032290 = 1;
  dword_5032298 = 0;
  byte_503229C = 1;
  v19 = sub_C57470();
  v20 = (unsigned int)qword_5032270;
  v21 = (unsigned int)qword_5032270 + 1LL;
  if ( v21 > HIDWORD(qword_5032270) )
  {
    sub_C8D5F0((char *)&unk_5032278 - 16, &unk_5032278, v21, 8);
    v20 = (unsigned int)qword_5032270;
  }
  *(_QWORD *)(qword_5032268 + 8 * v20) = v19;
  LODWORD(qword_5032270) = qword_5032270 + 1;
  byte_50322C0 = 0;
  qword_50322B0 = (__int64)&unk_49DE5F0;
  qword_50322A8 = 0;
  qword_50322B8 = 0;
  qword_5032220 = (__int64)&unk_49DE610;
  qword_50322C8 = (__int64)&unk_49DC2F0;
  qword_50322E8 = (__int64)nullsub_190;
  qword_50322E0 = (__int64)sub_D83E80;
  sub_C53080(&qword_5032220, "global-merging-extra-threshold", 30);
  qword_5032250 = 91;
  qword_5032248 = (__int64)"An additional cost threshold that must be exceeded for merging to be considered beneficial.";
  qword_50322A8 = 0;
  byte_50322C0 = 1;
  qword_50322B8 = 0;
  LOBYTE(dword_503222C) = dword_503222C & 0x9F | 0x20;
  sub_C53130(&qword_5032220);
  return __cxa_atexit(sub_D84280, &qword_5032220, &qword_4A427C0);
}
