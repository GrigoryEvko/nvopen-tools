// Function: ctor_671_0
// Address: 0x5a0470
//
int ctor_671_0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // r14
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // edx
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  char v13; // [rsp+23h] [rbp-4Dh] BYREF
  int v14; // [rsp+24h] [rbp-4Ch] BYREF
  char *v15; // [rsp+28h] [rbp-48h] BYREF
  const char *v16; // [rsp+30h] [rbp-40h] BYREF
  __int64 v17; // [rsp+38h] [rbp-38h]

  sub_D95050(&qword_503CE80, 0, 0);
  qword_503CF08 = 0;
  qword_503CF40 = (__int64)nullsub_24;
  qword_503CF10 = (__int64)&unk_49D9728;
  qword_503CF18 = 0;
  qword_503CE80 = (__int64)&unk_49DBF10;
  qword_503CF20 = (__int64)&unk_49DC290;
  qword_503CF38 = (__int64)sub_984050;
  sub_C53080(&qword_503CE80, "align-all-blocks", 16);
  qword_503CEB0 = 103;
  qword_503CEA8 = (__int64)"Force the alignment of all blocks in the function in log2 format (e.g 4 means align on 16B boundaries).";
  LODWORD(qword_503CF08) = 0;
  BYTE4(qword_503CF18) = 1;
  LODWORD(qword_503CF18) = 0;
  byte_503CE8C = byte_503CE8C & 0x9F | 0x20;
  sub_C53130(&qword_503CE80);
  __cxa_atexit(sub_984970, &qword_503CE80, &qword_4A427C0);
  sub_D95050(&qword_503CDA0, 0, 0);
  qword_503CE30 = (__int64)&unk_49D9728;
  qword_503CDA0 = (__int64)&unk_49DBF10;
  qword_503CE40 = (__int64)&unk_49DC290;
  qword_503CE60 = (__int64)nullsub_24;
  qword_503CE58 = (__int64)sub_984050;
  qword_503CE28 = 0;
  qword_503CE38 = 0;
  sub_C53080(&qword_503CDA0, "align-all-nofallthru-blocks", 27);
  qword_503CDD0 = 167;
  qword_503CDC8 = (__int64)"Force the alignment of all blocks that have no fall-through predecessors (i.e. don't add nops"
                           " that are executed). In log2 format (e.g 4 means align on 16B boundaries).";
  LODWORD(qword_503CE28) = 0;
  BYTE4(qword_503CE38) = 1;
  LODWORD(qword_503CE38) = 0;
  byte_503CDAC = byte_503CDAC & 0x9F | 0x20;
  sub_C53130(&qword_503CDA0);
  __cxa_atexit(sub_984970, &qword_503CDA0, &qword_4A427C0);
  sub_D95050(&qword_503CCC0, 0, 0);
  qword_503CD50 = (__int64)&unk_49D9728;
  qword_503CCC0 = (__int64)&unk_49DBF10;
  qword_503CD60 = (__int64)&unk_49DC290;
  qword_503CD80 = (__int64)nullsub_24;
  qword_503CD78 = (__int64)sub_984050;
  qword_503CD48 = 0;
  qword_503CD58 = 0;
  sub_C53080(&qword_503CCC0, "max-bytes-for-alignment", 23);
  qword_503CCF0 = 73;
  qword_503CCE8 = (__int64)"Forces the maximum bytes allowed to be emitted when padding for alignment";
  LODWORD(qword_503CD48) = 0;
  BYTE4(qword_503CD58) = 1;
  LODWORD(qword_503CD58) = 0;
  byte_503CCCC = byte_503CCCC & 0x9F | 0x20;
  sub_C53130(&qword_503CCC0);
  __cxa_atexit(sub_984970, &qword_503CCC0, &qword_4A427C0);
  sub_D95050(&qword_503CBE0, 0, 0);
  qword_503CC70 = (__int64)&unk_49D9728;
  qword_503CBE0 = (__int64)&unk_49DBF10;
  qword_503CC80 = (__int64)&unk_49DC290;
  qword_503CCA0 = (__int64)nullsub_24;
  qword_503CC98 = (__int64)sub_984050;
  qword_503CC68 = 0;
  qword_503CC78 = 0;
  sub_C53080(&qword_503CBE0, "block-placement-exit-block-bias", 31);
  qword_503CC10 = 104;
  qword_503CC08 = (__int64)"Block frequency percentage a loop exit block needs over the original exit to be considered the new exit.";
  LODWORD(qword_503CC68) = 0;
  BYTE4(qword_503CC78) = 1;
  LODWORD(qword_503CC78) = 0;
  byte_503CBEC = byte_503CBEC & 0x9F | 0x20;
  sub_C53130(&qword_503CBE0);
  __cxa_atexit(sub_984970, &qword_503CBE0, &qword_4A427C0);
  sub_D95050(&qword_503CB00, 0, 0);
  qword_503CB90 = (__int64)&unk_49D9728;
  qword_503CB00 = (__int64)&unk_49DBF10;
  qword_503CBA0 = (__int64)&unk_49DC290;
  qword_503CBC0 = (__int64)nullsub_24;
  qword_503CBB8 = (__int64)sub_984050;
  qword_503CB88 = 0;
  qword_503CB98 = 0;
  sub_C53080(&qword_503CB00, "loop-to-cold-block-ratio", 24);
  qword_503CB30 = 108;
  qword_503CB28 = (__int64)"Outline loop blocks from loop chain if (frequency of loop) / (frequency of block) is greater than this ratio";
  LODWORD(qword_503CB88) = 5;
  BYTE4(qword_503CB98) = 1;
  LODWORD(qword_503CB98) = 5;
  byte_503CB0C = byte_503CB0C & 0x9F | 0x20;
  sub_C53130(&qword_503CB00);
  __cxa_atexit(sub_984970, &qword_503CB00, &qword_4A427C0);
  v14 = 1;
  v16 = "Force outlining cold blocks from loops.";
  v15 = &v13;
  v13 = 0;
  v17 = 39;
  sub_35177C0(&unk_503CA20, "force-loop-cold-block", &v16, &v15, &v14);
  __cxa_atexit(sub_984900, &unk_503CA20, &qword_4A427C0);
  v16 = "Model the cost of loop rotation more precisely by using profile data.";
  v15 = &v13;
  v14 = 1;
  v13 = 0;
  v17 = 69;
  sub_35177C0(&unk_503C940, "precise-rotation-cost", &v16, &v15, &v14);
  __cxa_atexit(sub_984900, &unk_503C940, &qword_4A427C0);
  v16 = "Force the use of precise cost loop rotation strategy.";
  v14 = 1;
  v15 = &v13;
  v13 = 0;
  v17 = 53;
  sub_35179D0(&unk_503C860, "force-precise-rotation-cost", &v16, &v15, &v14);
  __cxa_atexit(sub_984900, &unk_503C860, &qword_4A427C0);
  sub_D95050(&qword_503C780, 0, 0);
  qword_503C810 = (__int64)&unk_49D9728;
  qword_503C838 = (__int64)sub_984050;
  qword_503C780 = (__int64)&unk_49DBF10;
  qword_503C820 = (__int64)&unk_49DC290;
  qword_503C840 = (__int64)nullsub_24;
  qword_503C808 = 0;
  qword_503C818 = 0;
  sub_C53080(&qword_503C780, "misfetch-cost", 13);
  qword_503C7B0 = 130;
  qword_503C7A8 = (__int64)"Cost that models the probabilistic risk of an instruction misfetch due to a jump comparing to"
                           " falling through, whose cost is zero.";
  LODWORD(qword_503C808) = 1;
  BYTE4(qword_503C818) = 1;
  LODWORD(qword_503C818) = 1;
  byte_503C78C = byte_503C78C & 0x9F | 0x20;
  sub_C53130(&qword_503C780);
  __cxa_atexit(sub_984970, &qword_503C780, &qword_4A427C0);
  sub_D95050(&qword_503C6A0, 0, 0);
  qword_503C730 = (__int64)&unk_49D9728;
  qword_503C758 = (__int64)sub_984050;
  qword_503C6A0 = (__int64)&unk_49DBF10;
  qword_503C740 = (__int64)&unk_49DC290;
  qword_503C760 = (__int64)nullsub_24;
  qword_503C728 = 0;
  qword_503C738 = 0;
  sub_C53080(&qword_503C6A0, "jump-inst-cost", 14);
  qword_503C6D0 = 26;
  qword_503C6C8 = (__int64)"Cost of jump instructions.";
  LODWORD(qword_503C728) = 1;
  BYTE4(qword_503C738) = 1;
  LODWORD(qword_503C738) = 1;
  byte_503C6AC = byte_503C6AC & 0x9F | 0x20;
  sub_C53130(&qword_503C6A0);
  __cxa_atexit(sub_984970, &qword_503C6A0, &qword_4A427C0);
  sub_D95050(&qword_503C5C0, 0, 0);
  qword_503C648 = 0;
  qword_503C658 = 0;
  qword_503C650 = (__int64)&unk_49D9748;
  qword_503C5C0 = (__int64)&unk_49DC090;
  qword_503C660 = (__int64)&unk_49DC1D0;
  qword_503C680 = (__int64)nullsub_23;
  qword_503C678 = (__int64)sub_984030;
  sub_C53080(&qword_503C5C0, "tail-dup-placement", 18);
  qword_503C5F0 = 102;
  qword_503C5E8 = (__int64)"Perform tail duplication during placement. Creates more fallthrough opportunities in outline branches.";
  LOWORD(qword_503C658) = 257;
  LOBYTE(qword_503C648) = 1;
  byte_503C5CC = byte_503C5CC & 0x9F | 0x20;
  sub_C53130(&qword_503C5C0);
  __cxa_atexit(sub_984900, &qword_503C5C0, &qword_4A427C0);
  v13 = 1;
  v15 = &v13;
  v16 = "Perform branch folding during placement. Reduces code size.";
  v14 = 1;
  v17 = 59;
  sub_35177C0(&unk_503C4E0, "branch-fold-placement", &v16, &v15, &v14);
  __cxa_atexit(sub_984900, &unk_503C4E0, &qword_4A427C0);
  qword_503C400 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_503C4E0, v0, v1), 1u);
  qword_503C450 = 0x100000000LL;
  dword_503C40C &= 0x8000u;
  word_503C410 = 0;
  qword_503C418 = 0;
  qword_503C420 = 0;
  dword_503C408 = v2;
  qword_503C428 = 0;
  qword_503C430 = 0;
  qword_503C438 = 0;
  qword_503C440 = 0;
  qword_503C448 = (__int64)&unk_503C458;
  qword_503C460 = 0;
  qword_503C468 = (__int64)&unk_503C480;
  qword_503C470 = 1;
  dword_503C478 = 0;
  byte_503C47C = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_503C450;
  v5 = (unsigned int)qword_503C450 + 1LL;
  if ( v5 > HIDWORD(qword_503C450) )
  {
    sub_C8D5F0((char *)&unk_503C458 - 16, &unk_503C458, v5, 8);
    v4 = (unsigned int)qword_503C450;
  }
  *(_QWORD *)(qword_503C448 + 8 * v4) = v3;
  LODWORD(qword_503C450) = qword_503C450 + 1;
  qword_503C4C0 = (__int64)nullsub_24;
  qword_503C490 = (__int64)&unk_49D9728;
  qword_503C488 = 0;
  qword_503C498 = 0;
  qword_503C400 = (__int64)&unk_49DBF10;
  qword_503C4A0 = (__int64)&unk_49DC290;
  qword_503C4B8 = (__int64)sub_984050;
  sub_C53080(&qword_503C400, "tail-dup-placement-threshold", 28);
  qword_503C430 = 132;
  qword_503C428 = (__int64)"Instruction cutoff for tail duplication during layout. Tail merging during layout is forced t"
                           "o have a threshold that won't conflict.";
  LODWORD(qword_503C488) = 2;
  BYTE4(qword_503C498) = 1;
  LODWORD(qword_503C498) = 2;
  LOBYTE(dword_503C40C) = dword_503C40C & 0x9F | 0x20;
  sub_C53130(&qword_503C400);
  __cxa_atexit(sub_984970, &qword_503C400, &qword_4A427C0);
  qword_503C320 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503C400, v6, v7), 1u);
  byte_503C39C = 1;
  qword_503C370 = 0x100000000LL;
  dword_503C32C &= 0x8000u;
  qword_503C338 = 0;
  qword_503C340 = 0;
  qword_503C348 = 0;
  dword_503C328 = v8;
  word_503C330 = 0;
  qword_503C350 = 0;
  qword_503C358 = 0;
  qword_503C360 = 0;
  qword_503C368 = (__int64)&unk_503C378;
  qword_503C380 = 0;
  qword_503C388 = (__int64)&unk_503C3A0;
  qword_503C390 = 1;
  dword_503C398 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_503C370;
  v11 = (unsigned int)qword_503C370 + 1LL;
  if ( v11 > HIDWORD(qword_503C370) )
  {
    sub_C8D5F0((char *)&unk_503C378 - 16, &unk_503C378, v11, 8);
    v10 = (unsigned int)qword_503C370;
  }
  *(_QWORD *)(qword_503C368 + 8 * v10) = v9;
  LODWORD(qword_503C370) = qword_503C370 + 1;
  qword_503C3E0 = (__int64)nullsub_24;
  qword_503C3B0 = (__int64)&unk_49D9728;
  qword_503C3A8 = 0;
  qword_503C320 = (__int64)&unk_49DBF10;
  qword_503C3B8 = 0;
  qword_503C3C0 = (__int64)&unk_49DC290;
  qword_503C3D8 = (__int64)sub_984050;
  sub_C53080(&qword_503C320, "tail-dup-placement-aggressive-threshold", 39);
  qword_503C350 = 156;
  qword_503C348 = (__int64)"Instruction cutoff for aggressive tail duplication during layout. Used at -O3. Tail merging d"
                           "uring layout is forced to have a threshold that won't conflict.";
  LODWORD(qword_503C3A8) = 4;
  BYTE4(qword_503C3B8) = 1;
  LODWORD(qword_503C3B8) = 4;
  LOBYTE(dword_503C32C) = dword_503C32C & 0x9F | 0x20;
  sub_C53130(&qword_503C320);
  __cxa_atexit(sub_984970, &qword_503C320, &qword_4A427C0);
  sub_D95050(&qword_503C240, 0, 0);
  qword_503C2D0 = (__int64)&unk_49D9728;
  qword_503C240 = (__int64)&unk_49DBF10;
  qword_503C2E0 = (__int64)&unk_49DC290;
  qword_503C300 = (__int64)nullsub_24;
  qword_503C2F8 = (__int64)sub_984050;
  qword_503C2C8 = 0;
  qword_503C2D8 = 0;
  sub_C53080(&qword_503C240, "tail-dup-placement-penalty", 26);
  qword_503C270 = 213;
  qword_503C268 = (__int64)"Cost penalty for blocks that can avoid breaking CFG by copying. Copying can increase fallthro"
                           "ugh, but it also increases icache pressure. This parameter controls the penalty to account fo"
                           "r that. Percent as integer.";
  LODWORD(qword_503C2C8) = 2;
  BYTE4(qword_503C2D8) = 1;
  LODWORD(qword_503C2D8) = 2;
  byte_503C24C = byte_503C24C & 0x9F | 0x20;
  sub_C53130(&qword_503C240);
  __cxa_atexit(sub_984970, &qword_503C240, &qword_4A427C0);
  sub_D95050(&qword_503C160, 0, 0);
  qword_503C1F0 = (__int64)&unk_49D9728;
  qword_503C160 = (__int64)&unk_49DBF10;
  qword_503C200 = (__int64)&unk_49DC290;
  qword_503C220 = (__int64)nullsub_24;
  qword_503C218 = (__int64)sub_984050;
  qword_503C1E8 = 0;
  qword_503C1F8 = 0;
  sub_C53080(&qword_503C160, "tail-dup-profile-percent-threshold", 34);
  qword_503C190 = 167;
  qword_503C188 = (__int64)"If profile count information is used in tail duplication cost model, the gained fall through "
                           "number from tail duplication should be at least this percent of hot count.";
  LODWORD(qword_503C1E8) = 50;
  BYTE4(qword_503C1F8) = 1;
  LODWORD(qword_503C1F8) = 50;
  byte_503C16C = byte_503C16C & 0x9F | 0x20;
  sub_C53130(&qword_503C160);
  __cxa_atexit(sub_984970, &qword_503C160, &qword_4A427C0);
  sub_D95050(&qword_503C080, 0, 0);
  qword_503C110 = (__int64)&unk_49D9728;
  qword_503C080 = (__int64)&unk_49DBF10;
  qword_503C120 = (__int64)&unk_49DC290;
  qword_503C140 = (__int64)nullsub_24;
  qword_503C138 = (__int64)sub_984050;
  qword_503C108 = 0;
  qword_503C118 = 0;
  sub_C53080(&qword_503C080, "triangle-chain-count", 20);
  qword_503C0B0 = 126;
  qword_503C0A8 = (__int64)"Number of triangle-shaped-CFG's that need to be in a row for the triangle tail duplication he"
                           "uristic to kick in. 0 to disable.";
  LODWORD(qword_503C108) = 2;
  BYTE4(qword_503C118) = 1;
  LODWORD(qword_503C118) = 2;
  byte_503C08C = byte_503C08C & 0x9F | 0x20;
  sub_C53130(&qword_503C080);
  __cxa_atexit(sub_984970, &qword_503C080, &qword_4A427C0);
  v13 = 0;
  v15 = &v13;
  v16 = "If true, basic blocks are re-numbered before MBP layout is printed into a dot graph. Only used when a function i"
        "s being printed.";
  v14 = 1;
  v17 = 128;
  sub_35179D0(&unk_503BFA0, "renumber-blocks-before-view", &v16, &v15, &v14);
  __cxa_atexit(sub_984900, &unk_503BFA0, &qword_4A427C0);
  sub_D95050(&qword_503BEC0, 0, 0);
  qword_503BF50 = (__int64)&unk_49D9728;
  qword_503BEC0 = (__int64)&unk_49DBF10;
  qword_503BF60 = (__int64)&unk_49DC290;
  qword_503BF80 = (__int64)nullsub_24;
  qword_503BF78 = (__int64)sub_984050;
  qword_503BF48 = 0;
  qword_503BF58 = 0;
  sub_C53080(&qword_503BEC0, "ext-tsp-block-placement-max-blocks", 34);
  qword_503BEF0 = 76;
  qword_503BEE8 = (__int64)"Maximum number of basic blocks in a function to run ext-TSP block placement.";
  LODWORD(qword_503BF48) = -1;
  BYTE4(qword_503BF58) = 1;
  LODWORD(qword_503BF58) = -1;
  byte_503BECC = byte_503BECC & 0x9F | 0x20;
  sub_C53130(&qword_503BEC0);
  __cxa_atexit(sub_984970, &qword_503BEC0, &qword_4A427C0);
  sub_D95050(&qword_503BDE0, 0, 0);
  qword_503BE68 = 0;
  qword_503BE78 = 0;
  qword_503BE70 = (__int64)&unk_49D9748;
  qword_503BDE0 = (__int64)&unk_49DC090;
  qword_503BE80 = (__int64)&unk_49DC1D0;
  qword_503BEA0 = (__int64)nullsub_23;
  qword_503BE98 = (__int64)sub_984030;
  sub_C53080(&qword_503BDE0, "apply-ext-tsp-for-size", 22);
  LOBYTE(qword_503BE68) = 0;
  LOWORD(qword_503BE78) = 256;
  qword_503BE10 = 43;
  byte_503BDEC = byte_503BDEC & 0x9F | 0x20;
  qword_503BE08 = (__int64)"Use ext-tsp for size-aware block placement.";
  sub_C53130(&qword_503BDE0);
  return __cxa_atexit(sub_984900, &qword_503BDE0, &qword_4A427C0);
}
