// Function: ctor_019
// Address: 0x455190
//
int ctor_019()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_4F81280 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F812D0 = 0x100000000LL;
  dword_4F8128C &= 0x8000u;
  word_4F81290 = 0;
  qword_4F81298 = 0;
  qword_4F812A0 = 0;
  dword_4F81288 = v0;
  qword_4F812A8 = 0;
  qword_4F812B0 = 0;
  qword_4F812B8 = 0;
  qword_4F812C0 = 0;
  qword_4F812C8 = (__int64)&unk_4F812D8;
  qword_4F812E0 = 0;
  qword_4F812E8 = (__int64)&unk_4F81300;
  qword_4F812F0 = 1;
  dword_4F812F8 = 0;
  byte_4F812FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F812D0;
  v3 = (unsigned int)qword_4F812D0 + 1LL;
  if ( v3 > HIDWORD(qword_4F812D0) )
  {
    sub_C8D5F0((char *)&unk_4F812D8 - 16, &unk_4F812D8, v3, 8);
    v2 = (unsigned int)qword_4F812D0;
  }
  *(_QWORD *)(qword_4F812C8 + 8 * v2) = v1;
  qword_4F81310 = (__int64)&unk_49D9748;
  qword_4F81280 = (__int64)&unk_49DC090;
  qword_4F81320 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F812D0) = qword_4F812D0 + 1;
  qword_4F81340 = (__int64)nullsub_23;
  qword_4F81308 = 0;
  qword_4F81338 = (__int64)sub_984030;
  qword_4F81318 = 0;
  sub_C53080(&qword_4F81280, "use-constant-int-for-fixed-length-splat", 39);
  LOWORD(qword_4F81318) = 256;
  LOBYTE(qword_4F81308) = 0;
  qword_4F812B0 = 59;
  LOBYTE(dword_4F8128C) = dword_4F8128C & 0x9F | 0x20;
  qword_4F812A8 = (__int64)"Use ConstantInt's native fixed-length vector splat support.";
  sub_C53130(&qword_4F81280);
  __cxa_atexit(sub_984900, &qword_4F81280, &qword_4A427C0);
  qword_4F811A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F811F0 = 0x100000000LL;
  dword_4F811AC &= 0x8000u;
  qword_4F811E8 = (__int64)&unk_4F811F8;
  word_4F811B0 = 0;
  qword_4F811B8 = 0;
  dword_4F811A8 = v4;
  qword_4F811C0 = 0;
  qword_4F811C8 = 0;
  qword_4F811D0 = 0;
  qword_4F811D8 = 0;
  qword_4F811E0 = 0;
  qword_4F81200 = 0;
  qword_4F81208 = (__int64)&unk_4F81220;
  qword_4F81210 = 1;
  dword_4F81218 = 0;
  byte_4F8121C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F811F0;
  if ( (unsigned __int64)(unsigned int)qword_4F811F0 + 1 > HIDWORD(qword_4F811F0) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_4F811F8 - 16, &unk_4F811F8, (unsigned int)qword_4F811F0 + 1LL, 8);
    v6 = (unsigned int)qword_4F811F0;
    v5 = v15;
  }
  *(_QWORD *)(qword_4F811E8 + 8 * v6) = v5;
  qword_4F81230 = (__int64)&unk_49D9748;
  qword_4F811A0 = (__int64)&unk_49DC090;
  qword_4F81240 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F811F0) = qword_4F811F0 + 1;
  qword_4F81260 = (__int64)nullsub_23;
  qword_4F81228 = 0;
  qword_4F81258 = (__int64)sub_984030;
  qword_4F81238 = 0;
  sub_C53080(&qword_4F811A0, "use-constant-fp-for-fixed-length-splat", 38);
  LOWORD(qword_4F81238) = 256;
  LOBYTE(qword_4F81228) = 0;
  qword_4F811D0 = 58;
  LOBYTE(dword_4F811AC) = dword_4F811AC & 0x9F | 0x20;
  qword_4F811C8 = (__int64)"Use ConstantFP's native fixed-length vector splat support.";
  sub_C53130(&qword_4F811A0);
  __cxa_atexit(sub_984900, &qword_4F811A0, &qword_4A427C0);
  qword_4F810C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F81110 = 0x100000000LL;
  dword_4F810CC &= 0x8000u;
  qword_4F81108 = (__int64)&unk_4F81118;
  word_4F810D0 = 0;
  qword_4F810D8 = 0;
  dword_4F810C8 = v7;
  qword_4F810E0 = 0;
  qword_4F810E8 = 0;
  qword_4F810F0 = 0;
  qword_4F810F8 = 0;
  qword_4F81100 = 0;
  qword_4F81120 = 0;
  qword_4F81128 = (__int64)&unk_4F81140;
  qword_4F81130 = 1;
  dword_4F81138 = 0;
  byte_4F8113C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F81110;
  if ( (unsigned __int64)(unsigned int)qword_4F81110 + 1 > HIDWORD(qword_4F81110) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_4F81118 - 16, &unk_4F81118, (unsigned int)qword_4F81110 + 1LL, 8);
    v9 = (unsigned int)qword_4F81110;
    v8 = v16;
  }
  *(_QWORD *)(qword_4F81108 + 8 * v9) = v8;
  qword_4F81150 = (__int64)&unk_49D9748;
  qword_4F810C0 = (__int64)&unk_49DC090;
  qword_4F81160 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F81110) = qword_4F81110 + 1;
  qword_4F81180 = (__int64)nullsub_23;
  qword_4F81148 = 0;
  qword_4F81178 = (__int64)sub_984030;
  qword_4F81158 = 0;
  sub_C53080(&qword_4F810C0, "use-constant-int-for-scalable-splat", 35);
  LOWORD(qword_4F81158) = 256;
  LOBYTE(qword_4F81148) = 0;
  qword_4F810F0 = 55;
  LOBYTE(dword_4F810CC) = dword_4F810CC & 0x9F | 0x20;
  qword_4F810E8 = (__int64)"Use ConstantInt's native scalable vector splat support.";
  sub_C53130(&qword_4F810C0);
  __cxa_atexit(sub_984900, &qword_4F810C0, &qword_4A427C0);
  qword_4F80FE0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F81030 = 0x100000000LL;
  dword_4F80FEC &= 0x8000u;
  word_4F80FF0 = 0;
  qword_4F81028 = (__int64)&unk_4F81038;
  qword_4F80FF8 = 0;
  dword_4F80FE8 = v10;
  qword_4F81000 = 0;
  qword_4F81008 = 0;
  qword_4F81010 = 0;
  qword_4F81018 = 0;
  qword_4F81020 = 0;
  qword_4F81040 = 0;
  qword_4F81048 = (__int64)&unk_4F81060;
  qword_4F81050 = 1;
  dword_4F81058 = 0;
  byte_4F8105C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4F81030;
  v13 = (unsigned int)qword_4F81030 + 1LL;
  if ( v13 > HIDWORD(qword_4F81030) )
  {
    sub_C8D5F0((char *)&unk_4F81038 - 16, &unk_4F81038, v13, 8);
    v12 = (unsigned int)qword_4F81030;
  }
  *(_QWORD *)(qword_4F81028 + 8 * v12) = v11;
  qword_4F81070 = (__int64)&unk_49D9748;
  qword_4F80FE0 = (__int64)&unk_49DC090;
  qword_4F81080 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F81030) = qword_4F81030 + 1;
  qword_4F810A0 = (__int64)nullsub_23;
  qword_4F81068 = 0;
  qword_4F81098 = (__int64)sub_984030;
  qword_4F81078 = 0;
  sub_C53080(&qword_4F80FE0, "use-constant-fp-for-scalable-splat", 34);
  LOBYTE(qword_4F81068) = 0;
  LOWORD(qword_4F81078) = 256;
  qword_4F81010 = 54;
  LOBYTE(dword_4F80FEC) = dword_4F80FEC & 0x9F | 0x20;
  qword_4F81008 = (__int64)"Use ConstantFP's native scalable vector splat support.";
  sub_C53130(&qword_4F80FE0);
  return __cxa_atexit(sub_984900, &qword_4F80FE0, &qword_4A427C0);
}
