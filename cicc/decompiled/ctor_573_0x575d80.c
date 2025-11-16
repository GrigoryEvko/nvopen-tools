// Function: ctor_573
// Address: 0x575d80
//
int ctor_573()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+18h] [rbp-58h]
  int v21; // [rsp+24h] [rbp-4Ch] BYREF
  int *v22; // [rsp+28h] [rbp-48h]
  const char *v23; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24; // [rsp+38h] [rbp-38h]

  qword_5022280 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50222D0 = 0x100000000LL;
  dword_502228C &= 0x8000u;
  word_5022290 = 0;
  qword_5022298 = 0;
  qword_50222A0 = 0;
  dword_5022288 = v0;
  qword_50222A8 = 0;
  qword_50222B0 = 0;
  qword_50222B8 = 0;
  qword_50222C0 = 0;
  qword_50222C8 = (__int64)&unk_50222D8;
  qword_50222E0 = 0;
  qword_50222E8 = (__int64)&unk_5022300;
  qword_50222F0 = 1;
  dword_50222F8 = 0;
  byte_50222FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50222D0;
  v3 = (unsigned int)qword_50222D0 + 1LL;
  if ( v3 > HIDWORD(qword_50222D0) )
  {
    sub_C8D5F0((char *)&unk_50222D8 - 16, &unk_50222D8, v3, 8);
    v2 = (unsigned int)qword_50222D0;
  }
  *(_QWORD *)(qword_50222C8 + 8 * v2) = v1;
  qword_5022310 = (__int64)&unk_49D9748;
  qword_5022280 = (__int64)&unk_49DC090;
  LODWORD(qword_50222D0) = qword_50222D0 + 1;
  qword_5022308 = 0;
  qword_5022320 = (__int64)&unk_49DC1D0;
  qword_5022318 = 0;
  qword_5022340 = (__int64)nullsub_23;
  qword_5022338 = (__int64)sub_984030;
  sub_C53080(&qword_5022280, "machine-sink-split", 18);
  qword_50222A8 = (__int64)"Split critical edges during machine sinking";
  LOWORD(qword_5022318) = 257;
  LOBYTE(qword_5022308) = 1;
  qword_50222B0 = 43;
  LOBYTE(dword_502228C) = dword_502228C & 0x9F | 0x20;
  sub_C53130(&qword_5022280);
  __cxa_atexit(sub_984900, &qword_5022280, &qword_4A427C0);
  qword_50221A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50221F0 = 0x100000000LL;
  dword_50221AC &= 0x8000u;
  qword_50221E8 = (__int64)&unk_50221F8;
  word_50221B0 = 0;
  qword_50221B8 = 0;
  dword_50221A8 = v4;
  qword_50221C0 = 0;
  qword_50221C8 = 0;
  qword_50221D0 = 0;
  qword_50221D8 = 0;
  qword_50221E0 = 0;
  qword_5022200 = 0;
  qword_5022208 = (__int64)&unk_5022220;
  qword_5022210 = 1;
  dword_5022218 = 0;
  byte_502221C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50221F0;
  v7 = (unsigned int)qword_50221F0 + 1LL;
  if ( v7 > HIDWORD(qword_50221F0) )
  {
    sub_C8D5F0((char *)&unk_50221F8 - 16, &unk_50221F8, v7, 8);
    v6 = (unsigned int)qword_50221F0;
  }
  *(_QWORD *)(qword_50221E8 + 8 * v6) = v5;
  qword_5022230 = (__int64)&unk_49D9748;
  qword_50221A0 = (__int64)&unk_49DC090;
  LODWORD(qword_50221F0) = qword_50221F0 + 1;
  qword_5022228 = 0;
  qword_5022240 = (__int64)&unk_49DC1D0;
  qword_5022238 = 0;
  qword_5022260 = (__int64)nullsub_23;
  qword_5022258 = (__int64)sub_984030;
  sub_C53080(&qword_50221A0, "machine-sink-bfi", 16);
  qword_50221C8 = (__int64)"Use block frequency info to find successors to sink";
  LOWORD(qword_5022238) = 257;
  LOBYTE(qword_5022228) = 1;
  qword_50221D0 = 51;
  LOBYTE(dword_50221AC) = dword_50221AC & 0x9F | 0x20;
  sub_C53130(&qword_50221A0);
  __cxa_atexit(sub_984900, &qword_50221A0, &qword_4A427C0);
  qword_50220C0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022110 = 0x100000000LL;
  dword_50220CC &= 0x8000u;
  qword_5022108 = (__int64)&unk_5022118;
  word_50220D0 = 0;
  qword_50220D8 = 0;
  dword_50220C8 = v8;
  qword_50220E0 = 0;
  qword_50220E8 = 0;
  qword_50220F0 = 0;
  qword_50220F8 = 0;
  qword_5022100 = 0;
  qword_5022120 = 0;
  qword_5022128 = (__int64)&unk_5022140;
  qword_5022130 = 1;
  dword_5022138 = 0;
  byte_502213C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5022110;
  v11 = (unsigned int)qword_5022110 + 1LL;
  if ( v11 > HIDWORD(qword_5022110) )
  {
    sub_C8D5F0((char *)&unk_5022118 - 16, &unk_5022118, v11, 8);
    v10 = (unsigned int)qword_5022110;
  }
  *(_QWORD *)(qword_5022108 + 8 * v10) = v9;
  LODWORD(qword_5022110) = qword_5022110 + 1;
  qword_5022148 = 0;
  qword_5022150 = (__int64)&unk_49D9728;
  qword_5022158 = 0;
  qword_50220C0 = (__int64)&unk_49DBF10;
  qword_5022160 = (__int64)&unk_49DC290;
  qword_5022180 = (__int64)nullsub_24;
  qword_5022178 = (__int64)sub_984050;
  sub_C53080(&qword_50220C0, "machine-sink-split-probability-threshold", 40);
  qword_50220F0 = 222;
  qword_50220E8 = (__int64)"Percentage threshold for splitting single-instruction critical edge. If the branch threshold "
                           "is higher than this threshold, we allow speculative execution of up to 1 instruction to avoid"
                           " branching to splitted critical edge";
  LODWORD(qword_5022148) = 40;
  BYTE4(qword_5022158) = 1;
  LODWORD(qword_5022158) = 40;
  LOBYTE(dword_50220CC) = dword_50220CC & 0x9F | 0x20;
  sub_C53130(&qword_50220C0);
  __cxa_atexit(sub_984970, &qword_50220C0, &qword_4A427C0);
  v22 = &v21;
  v23 = "Do not try to find alias store for a load if there is a in-path block whose instruction number is higher than this threshold.";
  v21 = 2000;
  v24 = 125;
  sub_2ED5480(&unk_5021FE0, "machine-sink-load-instrs-threshold", &v23);
  __cxa_atexit(sub_984970, &unk_5021FE0, &qword_4A427C0);
  v23 = "Do not try to find alias store for a load if the block number in the straight line is higher than this threshold.";
  v22 = &v21;
  v21 = 20;
  v24 = 113;
  sub_2ED5480(&unk_5021F00, "machine-sink-load-blocks-threshold", &v23);
  __cxa_atexit(sub_984970, &unk_5021F00, &qword_4A427C0);
  qword_5021E20 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5021E70 = 0x100000000LL;
  dword_5021E2C &= 0x8000u;
  word_5021E30 = 0;
  qword_5021E68 = (__int64)&unk_5021E78;
  qword_5021E38 = 0;
  dword_5021E28 = v12;
  qword_5021E40 = 0;
  qword_5021E48 = 0;
  qword_5021E50 = 0;
  qword_5021E58 = 0;
  qword_5021E60 = 0;
  qword_5021E80 = 0;
  qword_5021E88 = (__int64)&unk_5021EA0;
  qword_5021E90 = 1;
  dword_5021E98 = 0;
  byte_5021E9C = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_5021E70;
  if ( (unsigned __int64)(unsigned int)qword_5021E70 + 1 > HIDWORD(qword_5021E70) )
  {
    v20 = v13;
    sub_C8D5F0((char *)&unk_5021E78 - 16, &unk_5021E78, (unsigned int)qword_5021E70 + 1LL, 8);
    v14 = (unsigned int)qword_5021E70;
    v13 = v20;
  }
  *(_QWORD *)(qword_5021E68 + 8 * v14) = v13;
  qword_5021EB0 = (__int64)&unk_49D9748;
  qword_5021E20 = (__int64)&unk_49DC090;
  LODWORD(qword_5021E70) = qword_5021E70 + 1;
  qword_5021EA8 = 0;
  qword_5021EC0 = (__int64)&unk_49DC1D0;
  qword_5021EB8 = 0;
  qword_5021EE0 = (__int64)nullsub_23;
  qword_5021ED8 = (__int64)sub_984030;
  sub_C53080(&qword_5021E20, "sink-insts-to-avoid-spills", 26);
  qword_5021E50 = 54;
  qword_5021E48 = (__int64)"Sink instructions into cycles to avoid register spills";
  LOWORD(qword_5021EB8) = 256;
  LOBYTE(qword_5021EA8) = 0;
  LOBYTE(dword_5021E2C) = dword_5021E2C & 0x9F | 0x20;
  sub_C53130(&qword_5021E20);
  __cxa_atexit(sub_984900, &qword_5021E20, &qword_4A427C0);
  qword_5021D40 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5021DBC = 1;
  qword_5021D90 = 0x100000000LL;
  dword_5021D4C &= 0x8000u;
  qword_5021D58 = 0;
  qword_5021D60 = 0;
  qword_5021D68 = 0;
  dword_5021D48 = v15;
  word_5021D50 = 0;
  qword_5021D70 = 0;
  qword_5021D78 = 0;
  qword_5021D80 = 0;
  qword_5021D88 = (__int64)&unk_5021D98;
  qword_5021DA0 = 0;
  qword_5021DA8 = (__int64)&unk_5021DC0;
  qword_5021DB0 = 1;
  dword_5021DB8 = 0;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5021D90;
  v18 = (unsigned int)qword_5021D90 + 1LL;
  if ( v18 > HIDWORD(qword_5021D90) )
  {
    sub_C8D5F0((char *)&unk_5021D98 - 16, &unk_5021D98, v18, 8);
    v17 = (unsigned int)qword_5021D90;
  }
  *(_QWORD *)(qword_5021D88 + 8 * v17) = v16;
  LODWORD(qword_5021D90) = qword_5021D90 + 1;
  qword_5021DC8 = 0;
  qword_5021DD0 = (__int64)&unk_49D9728;
  qword_5021DD8 = 0;
  qword_5021D40 = (__int64)&unk_49DBF10;
  qword_5021DE0 = (__int64)&unk_49DC290;
  qword_5021E00 = (__int64)nullsub_24;
  qword_5021DF8 = (__int64)sub_984050;
  sub_C53080(&qword_5021D40, "machine-sink-cycle-limit", 24);
  qword_5021D70 = 64;
  qword_5021D68 = (__int64)"The maximum number of instructions considered for cycle sinking.";
  LODWORD(qword_5021DC8) = 50;
  BYTE4(qword_5021DD8) = 1;
  LODWORD(qword_5021DD8) = 50;
  LOBYTE(dword_5021D4C) = dword_5021D4C & 0x9F | 0x20;
  sub_C53130(&qword_5021D40);
  return __cxa_atexit(sub_984970, &qword_5021D40, &qword_4A427C0);
}
