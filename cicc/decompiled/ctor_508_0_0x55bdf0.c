// Function: ctor_508_0
// Address: 0x55bdf0
//
int ctor_508_0()
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
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+10h] [rbp-50h] BYREF
  int v22; // [rsp+14h] [rbp-4Ch] BYREF
  int *v23; // [rsp+18h] [rbp-48h] BYREF
  const char *v24; // [rsp+20h] [rbp-40h] BYREF
  __int64 v25; // [rsp+28h] [rbp-38h]

  qword_500B900 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500B950 = 0x100000000LL;
  dword_500B90C &= 0x8000u;
  word_500B910 = 0;
  qword_500B918 = 0;
  qword_500B920 = 0;
  dword_500B908 = v0;
  qword_500B928 = 0;
  qword_500B930 = 0;
  qword_500B938 = 0;
  qword_500B940 = 0;
  qword_500B948 = (__int64)&unk_500B958;
  qword_500B960 = 0;
  qword_500B968 = (__int64)&unk_500B980;
  qword_500B970 = 1;
  dword_500B978 = 0;
  byte_500B97C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500B950;
  v3 = (unsigned int)qword_500B950 + 1LL;
  if ( v3 > HIDWORD(qword_500B950) )
  {
    sub_C8D5F0((char *)&unk_500B958 - 16, &unk_500B958, v3, 8);
    v2 = (unsigned int)qword_500B950;
  }
  *(_QWORD *)(qword_500B948 + 8 * v2) = v1;
  qword_500B990 = (__int64)&unk_49D9748;
  qword_500B900 = (__int64)&unk_49DC090;
  qword_500B9A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500B950) = qword_500B950 + 1;
  qword_500B9C0 = (__int64)nullsub_23;
  qword_500B9B8 = (__int64)sub_984030;
  qword_500B988 = 0;
  qword_500B998 = 0;
  sub_C53080(&qword_500B900, "sample-profile-even-flow-distribution", 37);
  LOWORD(qword_500B998) = 257;
  LOBYTE(qword_500B988) = 1;
  qword_500B930 = 77;
  LOBYTE(dword_500B90C) = dword_500B90C & 0x9F | 0x20;
  qword_500B928 = (__int64)"Try to evenly distribute flow when there are multiple equally likely options.";
  sub_C53130(&qword_500B900);
  __cxa_atexit(sub_984900, &qword_500B900, &qword_4A427C0);
  qword_500B820 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500B870 = 0x100000000LL;
  dword_500B82C &= 0x8000u;
  qword_500B868 = (__int64)&unk_500B878;
  word_500B830 = 0;
  qword_500B838 = 0;
  dword_500B828 = v4;
  qword_500B840 = 0;
  qword_500B848 = 0;
  qword_500B850 = 0;
  qword_500B858 = 0;
  qword_500B860 = 0;
  qword_500B880 = 0;
  qword_500B888 = (__int64)&unk_500B8A0;
  qword_500B890 = 1;
  dword_500B898 = 0;
  byte_500B89C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500B870;
  if ( (unsigned __int64)(unsigned int)qword_500B870 + 1 > HIDWORD(qword_500B870) )
  {
    v19 = v5;
    sub_C8D5F0((char *)&unk_500B878 - 16, &unk_500B878, (unsigned int)qword_500B870 + 1LL, 8);
    v6 = (unsigned int)qword_500B870;
    v5 = v19;
  }
  *(_QWORD *)(qword_500B868 + 8 * v6) = v5;
  qword_500B8B0 = (__int64)&unk_49D9748;
  qword_500B820 = (__int64)&unk_49DC090;
  qword_500B8C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500B870) = qword_500B870 + 1;
  qword_500B8E0 = (__int64)nullsub_23;
  qword_500B8D8 = (__int64)sub_984030;
  qword_500B8A8 = 0;
  qword_500B8B8 = 0;
  sub_C53080(&qword_500B820, "sample-profile-rebalance-unknown", 32);
  LOWORD(qword_500B8B8) = 257;
  LOBYTE(qword_500B8A8) = 1;
  qword_500B850 = 50;
  LOBYTE(dword_500B82C) = dword_500B82C & 0x9F | 0x20;
  qword_500B848 = (__int64)"Evenly re-distribute flow among unknown subgraphs.";
  sub_C53130(&qword_500B820);
  __cxa_atexit(sub_984900, &qword_500B820, &qword_4A427C0);
  qword_500B740 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500B790 = 0x100000000LL;
  dword_500B74C &= 0x8000u;
  qword_500B788 = (__int64)&unk_500B798;
  word_500B750 = 0;
  qword_500B758 = 0;
  dword_500B748 = v7;
  qword_500B760 = 0;
  qword_500B768 = 0;
  qword_500B770 = 0;
  qword_500B778 = 0;
  qword_500B780 = 0;
  qword_500B7A0 = 0;
  qword_500B7A8 = (__int64)&unk_500B7C0;
  qword_500B7B0 = 1;
  dword_500B7B8 = 0;
  byte_500B7BC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_500B790;
  if ( (unsigned __int64)(unsigned int)qword_500B790 + 1 > HIDWORD(qword_500B790) )
  {
    v20 = v8;
    sub_C8D5F0((char *)&unk_500B798 - 16, &unk_500B798, (unsigned int)qword_500B790 + 1LL, 8);
    v9 = (unsigned int)qword_500B790;
    v8 = v20;
  }
  *(_QWORD *)(qword_500B788 + 8 * v9) = v8;
  qword_500B7D0 = (__int64)&unk_49D9748;
  qword_500B740 = (__int64)&unk_49DC090;
  qword_500B7E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500B790) = qword_500B790 + 1;
  qword_500B800 = (__int64)nullsub_23;
  qword_500B7F8 = (__int64)sub_984030;
  qword_500B7C8 = 0;
  qword_500B7D8 = 0;
  sub_C53080(&qword_500B740, "sample-profile-join-islands", 27);
  LOWORD(qword_500B7D8) = 257;
  LOBYTE(qword_500B7C8) = 1;
  qword_500B770 = 46;
  LOBYTE(dword_500B74C) = dword_500B74C & 0x9F | 0x20;
  qword_500B768 = (__int64)"Join isolated components having positive flow.";
  sub_C53130(&qword_500B740);
  __cxa_atexit(sub_984900, &qword_500B740, &qword_4A427C0);
  v25 = 46;
  v24 = "The cost of increasing a block's count by one.";
  v22 = 1;
  v21 = 10;
  v23 = &v21;
  sub_2A5BB30(&unk_500B660, "sample-profile-profi-cost-block-inc", &v23, &v22, &v24);
  __cxa_atexit(sub_984970, &unk_500B660, &qword_4A427C0);
  v25 = 46;
  v24 = "The cost of decreasing a block's count by one.";
  v22 = 1;
  v21 = 20;
  v23 = &v21;
  sub_2A5BB30(&unk_500B580, "sample-profile-profi-cost-block-dec", &v23, &v22, &v24);
  __cxa_atexit(sub_984970, &unk_500B580, &qword_4A427C0);
  v25 = 54;
  v24 = "The cost of increasing the entry block's count by one.";
  v22 = 1;
  v21 = 40;
  v23 = &v21;
  sub_2A5BD50(&unk_500B4A0, "sample-profile-profi-cost-block-entry-inc", &v23, &v22, &v24);
  __cxa_atexit(sub_984970, &unk_500B4A0, &qword_4A427C0);
  v25 = 54;
  v24 = "The cost of decreasing the entry block's count by one.";
  v22 = 1;
  v21 = 10;
  v23 = &v21;
  sub_2A5BD50(&unk_500B3C0, "sample-profile-profi-cost-block-entry-dec", &v23, &v22, &v24);
  __cxa_atexit(sub_984970, &unk_500B3C0, &qword_4A427C0);
  qword_500B2E0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500B330 = 0x100000000LL;
  word_500B2F0 = 0;
  dword_500B2EC &= 0x8000u;
  qword_500B2F8 = 0;
  qword_500B300 = 0;
  dword_500B2E8 = v10;
  qword_500B308 = 0;
  qword_500B310 = 0;
  qword_500B318 = 0;
  qword_500B320 = 0;
  qword_500B328 = (__int64)&unk_500B338;
  qword_500B340 = 0;
  qword_500B348 = (__int64)&unk_500B360;
  qword_500B350 = 1;
  dword_500B358 = 0;
  byte_500B35C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_500B330;
  v13 = (unsigned int)qword_500B330 + 1LL;
  if ( v13 > HIDWORD(qword_500B330) )
  {
    sub_C8D5F0((char *)&unk_500B338 - 16, &unk_500B338, v13, 8);
    v12 = (unsigned int)qword_500B330;
  }
  *(_QWORD *)(qword_500B328 + 8 * v12) = v11;
  qword_500B370 = (__int64)&unk_49D9728;
  qword_500B2E0 = (__int64)&unk_49DBF10;
  LODWORD(qword_500B330) = qword_500B330 + 1;
  qword_500B368 = 0;
  qword_500B380 = (__int64)&unk_49DC290;
  qword_500B378 = 0;
  qword_500B3A0 = (__int64)nullsub_24;
  qword_500B398 = (__int64)sub_984050;
  sub_C53080(&qword_500B2E0, "sample-profile-profi-cost-block-zero-inc", 40);
  LODWORD(qword_500B368) = 11;
  BYTE4(qword_500B378) = 1;
  LODWORD(qword_500B378) = 11;
  qword_500B310 = 59;
  LOBYTE(dword_500B2EC) = dword_500B2EC & 0x9F | 0x20;
  qword_500B308 = (__int64)"The cost of increasing a count of zero-weight block by one.";
  sub_C53130(&qword_500B2E0);
  __cxa_atexit(sub_984970, &qword_500B2E0, &qword_4A427C0);
  qword_500B200 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500B20C &= 0x8000u;
  word_500B210 = 0;
  qword_500B250 = 0x100000000LL;
  qword_500B218 = 0;
  qword_500B220 = 0;
  qword_500B228 = 0;
  dword_500B208 = v14;
  qword_500B230 = 0;
  qword_500B238 = 0;
  qword_500B240 = 0;
  qword_500B248 = (__int64)&unk_500B258;
  qword_500B260 = 0;
  qword_500B268 = (__int64)&unk_500B280;
  qword_500B270 = 1;
  dword_500B278 = 0;
  byte_500B27C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_500B250;
  v17 = (unsigned int)qword_500B250 + 1LL;
  if ( v17 > HIDWORD(qword_500B250) )
  {
    sub_C8D5F0((char *)&unk_500B258 - 16, &unk_500B258, v17, 8);
    v16 = (unsigned int)qword_500B250;
  }
  *(_QWORD *)(qword_500B248 + 8 * v16) = v15;
  qword_500B290 = (__int64)&unk_49D9728;
  qword_500B200 = (__int64)&unk_49DBF10;
  LODWORD(qword_500B250) = qword_500B250 + 1;
  qword_500B288 = 0;
  qword_500B2A0 = (__int64)&unk_49DC290;
  qword_500B298 = 0;
  qword_500B2C0 = (__int64)nullsub_24;
  qword_500B2B8 = (__int64)sub_984050;
  sub_C53080(&qword_500B200, "sample-profile-profi-cost-block-unknown-inc", 43);
  LODWORD(qword_500B288) = 0;
  BYTE4(qword_500B298) = 1;
  LODWORD(qword_500B298) = 0;
  qword_500B230 = 55;
  LOBYTE(dword_500B20C) = dword_500B20C & 0x9F | 0x20;
  qword_500B228 = (__int64)"The cost of increasing an unknown block's count by one.";
  sub_C53130(&qword_500B200);
  return __cxa_atexit(sub_984970, &qword_500B200, &qword_4A427C0);
}
