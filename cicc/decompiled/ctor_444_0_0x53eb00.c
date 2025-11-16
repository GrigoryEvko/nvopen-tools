// Function: ctor_444_0
// Address: 0x53eb00
//
int ctor_444_0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // [rsp+8h] [rbp-88h]
  int v7; // [rsp+18h] [rbp-78h] BYREF
  int v8; // [rsp+1Ch] [rbp-74h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-60h] BYREF
  const char *v11; // [rsp+40h] [rbp-50h] BYREF
  __int64 v12; // [rsp+48h] [rbp-48h]
  _QWORD v13[8]; // [rsp+50h] [rbp-40h] BYREF

  v0 = sub_C60B10();
  v11 = (const char *)v13;
  v1 = v0;
  sub_2753A20(&v11, "Controls which MemoryDefs are eliminated.");
  v9[0] = v10;
  sub_2753A20(v9, "dse-memoryssa");
  sub_CF9810(v1, v9, &v11);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  if ( v11 != (const char *)v13 )
    j_j___libc_free_0(v11, v13[0] + 1LL);
  sub_D95050(&qword_4FFAAE0, 0, 0);
  qword_4FFAB68 = 0;
  qword_4FFAB78 = 0;
  qword_4FFAB70 = (__int64)&unk_49D9748;
  qword_4FFAAE0 = (__int64)&unk_49DC090;
  qword_4FFAB80 = (__int64)&unk_49DC1D0;
  qword_4FFABA0 = (__int64)nullsub_23;
  qword_4FFAB98 = (__int64)sub_984030;
  sub_C53080(&qword_4FFAAE0, "enable-dse-partial-overwrite-tracking", 37);
  LOWORD(qword_4FFAB78) = 257;
  LOBYTE(qword_4FFAB68) = 1;
  qword_4FFAB10 = 40;
  byte_4FFAAEC = byte_4FFAAEC & 0x9F | 0x20;
  qword_4FFAB08 = (__int64)"Enable partial-overwrite tracking in DSE";
  sub_C53130(&qword_4FFAAE0);
  __cxa_atexit(sub_984900, &qword_4FFAAE0, &qword_4A427C0);
  sub_D95050(&qword_4FFAA00, 0, 0);
  qword_4FFAA00 = (__int64)&unk_49DC090;
  qword_4FFAAC0 = (__int64)nullsub_23;
  qword_4FFAA90 = (__int64)&unk_49D9748;
  qword_4FFAAA0 = (__int64)&unk_49DC1D0;
  qword_4FFAAB8 = (__int64)sub_984030;
  qword_4FFAA88 = 0;
  qword_4FFAA98 = 0;
  sub_C53080(&qword_4FFAA00, "enable-dse-partial-store-merging", 32);
  LOWORD(qword_4FFAA98) = 257;
  LOBYTE(qword_4FFAA88) = 1;
  qword_4FFAA30 = 35;
  byte_4FFAA0C = byte_4FFAA0C & 0x9F | 0x20;
  qword_4FFAA28 = (__int64)"Enable partial store merging in DSE";
  sub_C53130(&qword_4FFAA00);
  __cxa_atexit(sub_984900, &qword_4FFAA00, &qword_4A427C0);
  v9[0] = &v7;
  v11 = "The number of memory instructions to scan for dead store elimination (default = 150)";
  v12 = 84;
  v8 = 1;
  v7 = 150;
  sub_27576A0(&unk_4FFA920, "dse-memoryssa-scanlimit", v9, &v8, &v11);
  __cxa_atexit(sub_984970, &unk_4FFA920, &qword_4A427C0);
  v12 = 102;
  v11 = "The maximum number of steps while walking upwards to find MemoryDefs that may be killed (default = 90)";
  v9[0] = &v7;
  v8 = 1;
  v7 = 90;
  sub_27576A0(&unk_4FFA840, "dse-memoryssa-walklimit", v9, &v8, &v11);
  __cxa_atexit(sub_984970, &unk_4FFA840, &qword_4A427C0);
  sub_D95050(&qword_4FFA760, 0, 0);
  qword_4FFA7E8 = 0;
  qword_4FFA820 = (__int64)nullsub_24;
  qword_4FFA7F0 = (__int64)&unk_49D9728;
  qword_4FFA760 = (__int64)&unk_49DBF10;
  qword_4FFA800 = (__int64)&unk_49DC290;
  qword_4FFA818 = (__int64)sub_984050;
  qword_4FFA7F8 = 0;
  sub_C53080(&qword_4FFA760, "dse-memoryssa-partial-store-limit", 33);
  LODWORD(qword_4FFA7E8) = 5;
  BYTE4(qword_4FFA7F8) = 1;
  LODWORD(qword_4FFA7F8) = 5;
  qword_4FFA790 = 107;
  byte_4FFA76C = byte_4FFA76C & 0x9F | 0x20;
  qword_4FFA788 = (__int64)"The maximum number candidates that only partially overwrite the killing MemoryDef to consider (default = 5)";
  sub_C53130(&qword_4FFA760);
  __cxa_atexit(sub_984970, &qword_4FFA760, &qword_4A427C0);
  sub_D95050(&qword_4FFA680, 0, 0);
  qword_4FFA720 = (__int64)&unk_49DC290;
  qword_4FFA710 = (__int64)&unk_49D9728;
  qword_4FFA680 = (__int64)&unk_49DBF10;
  qword_4FFA740 = (__int64)nullsub_24;
  qword_4FFA738 = (__int64)sub_984050;
  qword_4FFA708 = 0;
  qword_4FFA718 = 0;
  sub_C53080(&qword_4FFA680, "dse-memoryssa-defs-per-block-limit", 34);
  LODWORD(qword_4FFA708) = 5000;
  BYTE4(qword_4FFA718) = 1;
  LODWORD(qword_4FFA718) = 5000;
  qword_4FFA6B0 = 110;
  byte_4FFA68C = byte_4FFA68C & 0x9F | 0x20;
  qword_4FFA6A8 = (__int64)"The number of MemoryDefs we consider as candidates to eliminated other stores per basic block"
                           " (default = 5000)";
  sub_C53130(&qword_4FFA680);
  __cxa_atexit(sub_984970, &qword_4FFA680, &qword_4A427C0);
  qword_4FFA5A0 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFA5F0 = 0x100000000LL;
  dword_4FFA5AC &= 0x8000u;
  word_4FFA5B0 = 0;
  qword_4FFA5B8 = 0;
  qword_4FFA5C0 = 0;
  dword_4FFA5A8 = v2;
  qword_4FFA5C8 = 0;
  qword_4FFA5D0 = 0;
  qword_4FFA5D8 = 0;
  qword_4FFA5E0 = 0;
  qword_4FFA5E8 = (__int64)&unk_4FFA5F8;
  qword_4FFA600 = 0;
  qword_4FFA608 = (__int64)&unk_4FFA620;
  qword_4FFA610 = 1;
  dword_4FFA618 = 0;
  byte_4FFA61C = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_4FFA5F0;
  if ( (unsigned __int64)(unsigned int)qword_4FFA5F0 + 1 > HIDWORD(qword_4FFA5F0) )
  {
    v6 = v3;
    sub_C8D5F0((char *)&unk_4FFA5F8 - 16, &unk_4FFA5F8, (unsigned int)qword_4FFA5F0 + 1LL, 8);
    v4 = (unsigned int)qword_4FFA5F0;
    v3 = v6;
  }
  *(_QWORD *)(qword_4FFA5E8 + 8 * v4) = v3;
  LODWORD(qword_4FFA5F0) = qword_4FFA5F0 + 1;
  qword_4FFA660 = (__int64)nullsub_24;
  qword_4FFA630 = (__int64)&unk_49D9728;
  qword_4FFA5A0 = (__int64)&unk_49DBF10;
  qword_4FFA640 = (__int64)&unk_49DC290;
  qword_4FFA658 = (__int64)sub_984050;
  qword_4FFA628 = 0;
  qword_4FFA638 = 0;
  sub_C53080(&qword_4FFA5A0, "dse-memoryssa-samebb-cost", 25);
  LODWORD(qword_4FFA628) = 1;
  BYTE4(qword_4FFA638) = 1;
  LODWORD(qword_4FFA638) = 1;
  qword_4FFA5D0 = 80;
  LOBYTE(dword_4FFA5AC) = dword_4FFA5AC & 0x9F | 0x20;
  qword_4FFA5C8 = (__int64)"The cost of a step in the same basic block as the killing MemoryDef(default = 1)";
  sub_C53130(&qword_4FFA5A0);
  __cxa_atexit(sub_984970, &qword_4FFA5A0, &qword_4A427C0);
  sub_D95050(&qword_4FFA4C0, 0, 0);
  qword_4FFA560 = (__int64)&unk_49DC290;
  qword_4FFA550 = (__int64)&unk_49D9728;
  qword_4FFA4C0 = (__int64)&unk_49DBF10;
  qword_4FFA580 = (__int64)nullsub_24;
  qword_4FFA578 = (__int64)sub_984050;
  qword_4FFA548 = 0;
  qword_4FFA558 = 0;
  sub_C53080(&qword_4FFA4C0, "dse-memoryssa-otherbb-cost", 26);
  LODWORD(qword_4FFA548) = 5;
  BYTE4(qword_4FFA558) = 1;
  LODWORD(qword_4FFA558) = 5;
  qword_4FFA4F0 = 85;
  byte_4FFA4CC = byte_4FFA4CC & 0x9F | 0x20;
  qword_4FFA4E8 = (__int64)"The cost of a step in a different basic block than the killing MemoryDef(default = 5)";
  sub_C53130(&qword_4FFA4C0);
  __cxa_atexit(sub_984970, &qword_4FFA4C0, &qword_4A427C0);
  sub_D95050(&qword_4FFA3E0, 0, 0);
  qword_4FFA480 = (__int64)&unk_49DC290;
  qword_4FFA470 = (__int64)&unk_49D9728;
  qword_4FFA3E0 = (__int64)&unk_49DBF10;
  qword_4FFA4A0 = (__int64)nullsub_24;
  qword_4FFA498 = (__int64)sub_984050;
  qword_4FFA468 = 0;
  qword_4FFA478 = 0;
  sub_C53080(&qword_4FFA3E0, "dse-memoryssa-path-check-limit", 30);
  LODWORD(qword_4FFA468) = 50;
  BYTE4(qword_4FFA478) = 1;
  LODWORD(qword_4FFA478) = 50;
  qword_4FFA410 = 126;
  byte_4FFA3EC = byte_4FFA3EC & 0x9F | 0x20;
  qword_4FFA408 = (__int64)"The maximum number of blocks to check when trying to prove that all paths to an exit go throu"
                           "gh a killing block (default = 50)";
  sub_C53130(&qword_4FFA3E0);
  __cxa_atexit(sub_984970, &qword_4FFA3E0, &qword_4A427C0);
  sub_D95050(&qword_4FFA300, 0, 0);
  qword_4FFA388 = 0;
  qword_4FFA398 = 0;
  qword_4FFA390 = (__int64)&unk_49D9748;
  qword_4FFA300 = (__int64)&unk_49DC090;
  qword_4FFA3A0 = (__int64)&unk_49DC1D0;
  qword_4FFA3C0 = (__int64)nullsub_23;
  qword_4FFA3B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FFA300, "dse-optimize-memoryssa", 22);
  LOBYTE(qword_4FFA388) = 1;
  LOWORD(qword_4FFA398) = 257;
  qword_4FFA330 = 38;
  byte_4FFA30C = byte_4FFA30C & 0x9F | 0x20;
  qword_4FFA328 = (__int64)"Allow DSE to optimize memory accesses.";
  sub_C53130(&qword_4FFA300);
  __cxa_atexit(sub_984900, &qword_4FFA300, &qword_4A427C0);
  sub_D95050(&qword_4FFA220, 0, 0);
  qword_4FFA2B0 = (__int64)&unk_49D9748;
  qword_4FFA2E0 = (__int64)nullsub_23;
  qword_4FFA220 = (__int64)&unk_49DC090;
  qword_4FFA2C0 = (__int64)&unk_49DC1D0;
  qword_4FFA2D8 = (__int64)sub_984030;
  qword_4FFA2A8 = 0;
  qword_4FFA2B8 = 0;
  sub_C53080(&qword_4FFA220, "enable-dse-initializes-attr-improvement", 39);
  LOWORD(qword_4FFA2B8) = 257;
  LOBYTE(qword_4FFA2A8) = 1;
  qword_4FFA250 = 46;
  byte_4FFA22C = byte_4FFA22C & 0x9F | 0x20;
  qword_4FFA248 = (__int64)"Enable the initializes attr improvement in DSE";
  sub_C53130(&qword_4FFA220);
  return __cxa_atexit(sub_984900, &qword_4FFA220, &qword_4A427C0);
}
