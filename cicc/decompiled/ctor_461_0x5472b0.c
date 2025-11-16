// Function: ctor_461
// Address: 0x5472b0
//
int ctor_461()
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
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_4FFF260 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFF2B0 = 0x100000000LL;
  dword_4FFF26C &= 0x8000u;
  word_4FFF270 = 0;
  qword_4FFF278 = 0;
  qword_4FFF280 = 0;
  dword_4FFF268 = v0;
  qword_4FFF288 = 0;
  qword_4FFF290 = 0;
  qword_4FFF298 = 0;
  qword_4FFF2A0 = 0;
  qword_4FFF2A8 = (__int64)&unk_4FFF2B8;
  qword_4FFF2C0 = 0;
  qword_4FFF2C8 = (__int64)&unk_4FFF2E0;
  qword_4FFF2D0 = 1;
  dword_4FFF2D8 = 0;
  byte_4FFF2DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFF2B0;
  v3 = (unsigned int)qword_4FFF2B0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFF2B0) )
  {
    sub_C8D5F0((char *)&unk_4FFF2B8 - 16, &unk_4FFF2B8, v3, 8);
    v2 = (unsigned int)qword_4FFF2B0;
  }
  *(_QWORD *)(qword_4FFF2A8 + 8 * v2) = v1;
  LODWORD(qword_4FFF2B0) = qword_4FFF2B0 + 1;
  qword_4FFF2E8 = 0;
  qword_4FFF2F0 = (__int64)&unk_49D9728;
  qword_4FFF2F8 = 0;
  qword_4FFF260 = (__int64)&unk_49DBF10;
  qword_4FFF300 = (__int64)&unk_49DC290;
  qword_4FFF320 = (__int64)nullsub_24;
  qword_4FFF318 = (__int64)sub_984050;
  sub_C53080(&qword_4FFF260, "loop-flatten-cost-threshold", 27);
  LODWORD(qword_4FFF2E8) = 2;
  BYTE4(qword_4FFF2F8) = 1;
  LODWORD(qword_4FFF2F8) = 2;
  qword_4FFF290 = 77;
  LOBYTE(dword_4FFF26C) = dword_4FFF26C & 0x9F | 0x20;
  qword_4FFF288 = (__int64)"Limit on the cost of instructions that can be repeated due to loop flattening";
  sub_C53130(&qword_4FFF260);
  __cxa_atexit(sub_984970, &qword_4FFF260, &qword_4A427C0);
  qword_4FFF180 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFF1D0 = 0x100000000LL;
  dword_4FFF18C &= 0x8000u;
  word_4FFF190 = 0;
  qword_4FFF198 = 0;
  qword_4FFF1A0 = 0;
  dword_4FFF188 = v4;
  qword_4FFF1A8 = 0;
  qword_4FFF1B0 = 0;
  qword_4FFF1B8 = 0;
  qword_4FFF1C0 = 0;
  qword_4FFF1C8 = (__int64)&unk_4FFF1D8;
  qword_4FFF1E0 = 0;
  qword_4FFF1E8 = (__int64)&unk_4FFF200;
  qword_4FFF1F0 = 1;
  dword_4FFF1F8 = 0;
  byte_4FFF1FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFF1D0;
  v7 = (unsigned int)qword_4FFF1D0 + 1LL;
  if ( v7 > HIDWORD(qword_4FFF1D0) )
  {
    sub_C8D5F0((char *)&unk_4FFF1D8 - 16, &unk_4FFF1D8, v7, 8);
    v6 = (unsigned int)qword_4FFF1D0;
  }
  *(_QWORD *)(qword_4FFF1C8 + 8 * v6) = v5;
  qword_4FFF210 = (__int64)&unk_49D9748;
  qword_4FFF180 = (__int64)&unk_49DC090;
  qword_4FFF220 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFF1D0) = qword_4FFF1D0 + 1;
  qword_4FFF240 = (__int64)nullsub_23;
  qword_4FFF208 = 0;
  qword_4FFF238 = (__int64)sub_984030;
  qword_4FFF218 = 0;
  sub_C53080(&qword_4FFF180, "loop-flatten-assume-no-overflow", 31);
  LOWORD(qword_4FFF218) = 256;
  LOBYTE(qword_4FFF208) = 0;
  qword_4FFF1B0 = 76;
  LOBYTE(dword_4FFF18C) = dword_4FFF18C & 0x9F | 0x20;
  qword_4FFF1A8 = (__int64)"Assume that the product of the two iteration trip counts will never overflow";
  sub_C53130(&qword_4FFF180);
  __cxa_atexit(sub_984900, &qword_4FFF180, &qword_4A427C0);
  qword_4FFF0A0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFF0F0 = 0x100000000LL;
  dword_4FFF0AC &= 0x8000u;
  qword_4FFF0E8 = (__int64)&unk_4FFF0F8;
  word_4FFF0B0 = 0;
  qword_4FFF0B8 = 0;
  dword_4FFF0A8 = v8;
  qword_4FFF0C0 = 0;
  qword_4FFF0C8 = 0;
  qword_4FFF0D0 = 0;
  qword_4FFF0D8 = 0;
  qword_4FFF0E0 = 0;
  qword_4FFF100 = 0;
  qword_4FFF108 = (__int64)&unk_4FFF120;
  qword_4FFF110 = 1;
  dword_4FFF118 = 0;
  byte_4FFF11C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FFF0F0;
  if ( (unsigned __int64)(unsigned int)qword_4FFF0F0 + 1 > HIDWORD(qword_4FFF0F0) )
  {
    v16 = v9;
    sub_C8D5F0((char *)&unk_4FFF0F8 - 16, &unk_4FFF0F8, (unsigned int)qword_4FFF0F0 + 1LL, 8);
    v10 = (unsigned int)qword_4FFF0F0;
    v9 = v16;
  }
  *(_QWORD *)(qword_4FFF0E8 + 8 * v10) = v9;
  qword_4FFF130 = (__int64)&unk_49D9748;
  qword_4FFF0A0 = (__int64)&unk_49DC090;
  qword_4FFF140 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFF0F0) = qword_4FFF0F0 + 1;
  qword_4FFF160 = (__int64)nullsub_23;
  qword_4FFF128 = 0;
  qword_4FFF158 = (__int64)sub_984030;
  qword_4FFF138 = 0;
  sub_C53080(&qword_4FFF0A0, "loop-flatten-widen-iv", 21);
  LOWORD(qword_4FFF138) = 257;
  LOBYTE(qword_4FFF128) = 1;
  qword_4FFF0D0 = 91;
  LOBYTE(dword_4FFF0AC) = dword_4FFF0AC & 0x9F | 0x20;
  qword_4FFF0C8 = (__int64)"Widen the loop induction variables, if possible, so overflow checks won't reject flattening";
  sub_C53130(&qword_4FFF0A0);
  __cxa_atexit(sub_984900, &qword_4FFF0A0, &qword_4A427C0);
  qword_4FFEFC0 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFF010 = 0x100000000LL;
  dword_4FFEFCC &= 0x8000u;
  word_4FFEFD0 = 0;
  qword_4FFF008 = (__int64)&unk_4FFF018;
  qword_4FFEFD8 = 0;
  dword_4FFEFC8 = v11;
  qword_4FFEFE0 = 0;
  qword_4FFEFE8 = 0;
  qword_4FFEFF0 = 0;
  qword_4FFEFF8 = 0;
  qword_4FFF000 = 0;
  qword_4FFF020 = 0;
  qword_4FFF028 = (__int64)&unk_4FFF040;
  qword_4FFF030 = 1;
  dword_4FFF038 = 0;
  byte_4FFF03C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FFF010;
  v14 = (unsigned int)qword_4FFF010 + 1LL;
  if ( v14 > HIDWORD(qword_4FFF010) )
  {
    sub_C8D5F0((char *)&unk_4FFF018 - 16, &unk_4FFF018, v14, 8);
    v13 = (unsigned int)qword_4FFF010;
  }
  *(_QWORD *)(qword_4FFF008 + 8 * v13) = v12;
  qword_4FFF050 = (__int64)&unk_49D9748;
  qword_4FFEFC0 = (__int64)&unk_49DC090;
  qword_4FFF060 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFF010) = qword_4FFF010 + 1;
  qword_4FFF080 = (__int64)nullsub_23;
  qword_4FFF048 = 0;
  qword_4FFF078 = (__int64)sub_984030;
  qword_4FFF058 = 0;
  sub_C53080(&qword_4FFEFC0, "loop-flatten-version-loops", 26);
  LOBYTE(qword_4FFF048) = 1;
  qword_4FFEFF0 = 46;
  LOBYTE(dword_4FFEFCC) = dword_4FFEFCC & 0x9F | 0x20;
  LOWORD(qword_4FFF058) = 257;
  qword_4FFEFE8 = (__int64)"Version loops if flattened loop could overflow";
  sub_C53130(&qword_4FFEFC0);
  return __cxa_atexit(sub_984900, &qword_4FFEFC0, &qword_4A427C0);
}
