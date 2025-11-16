// Function: ctor_479
// Address: 0x54ed10
//
int ctor_479()
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
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  qword_5004AE0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5004AEC &= 0x8000u;
  word_5004AF0 = 0;
  qword_5004B30 = 0x100000000LL;
  qword_5004AF8 = 0;
  qword_5004B00 = 0;
  qword_5004B08 = 0;
  dword_5004AE8 = v0;
  qword_5004B10 = 0;
  qword_5004B18 = 0;
  qword_5004B20 = 0;
  qword_5004B28 = (__int64)&unk_5004B38;
  qword_5004B40 = 0;
  qword_5004B48 = (__int64)&unk_5004B60;
  qword_5004B50 = 1;
  dword_5004B58 = 0;
  byte_5004B5C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5004B30;
  v3 = (unsigned int)qword_5004B30 + 1LL;
  if ( v3 > HIDWORD(qword_5004B30) )
  {
    sub_C8D5F0((char *)&unk_5004B38 - 16, &unk_5004B38, v3, 8);
    v2 = (unsigned int)qword_5004B30;
  }
  *(_QWORD *)(qword_5004B28 + 8 * v2) = v1;
  qword_5004B70 = (__int64)&unk_49D9748;
  qword_5004AE0 = (__int64)&unk_49DC090;
  qword_5004B80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5004B30) = qword_5004B30 + 1;
  qword_5004BA0 = (__int64)nullsub_23;
  qword_5004B68 = 0;
  qword_5004B98 = (__int64)sub_984030;
  qword_5004B78 = 0;
  sub_C53080(&qword_5004AE0, "spp-all-backedges", 17);
  LOWORD(qword_5004B78) = 256;
  LOBYTE(qword_5004B68) = 0;
  LOBYTE(dword_5004AEC) = dword_5004AEC & 0x9F | 0x20;
  sub_C53130(&qword_5004AE0);
  __cxa_atexit(sub_984900, &qword_5004AE0, &qword_4A427C0);
  qword_5004A00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5004A0C &= 0x8000u;
  word_5004A10 = 0;
  qword_5004A50 = 0x100000000LL;
  qword_5004A48 = (__int64)&unk_5004A58;
  qword_5004A18 = 0;
  qword_5004A20 = 0;
  dword_5004A08 = v4;
  qword_5004A28 = 0;
  qword_5004A30 = 0;
  qword_5004A38 = 0;
  qword_5004A40 = 0;
  qword_5004A60 = 0;
  qword_5004A68 = (__int64)&unk_5004A80;
  qword_5004A70 = 1;
  dword_5004A78 = 0;
  byte_5004A7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5004A50;
  if ( (unsigned __int64)(unsigned int)qword_5004A50 + 1 > HIDWORD(qword_5004A50) )
  {
    v21 = v5;
    sub_C8D5F0((char *)&unk_5004A58 - 16, &unk_5004A58, (unsigned int)qword_5004A50 + 1LL, 8);
    v6 = (unsigned int)qword_5004A50;
    v5 = v21;
  }
  *(_QWORD *)(qword_5004A48 + 8 * v6) = v5;
  LODWORD(qword_5004A50) = qword_5004A50 + 1;
  qword_5004A88 = 0;
  qword_5004A90 = (__int64)&unk_49DA090;
  qword_5004A98 = 0;
  qword_5004A00 = (__int64)&unk_49DBF90;
  qword_5004AA0 = (__int64)&unk_49DC230;
  qword_5004AC0 = (__int64)nullsub_58;
  qword_5004AB8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5004A00, "spp-counted-loop-trip-width", 27);
  LODWORD(qword_5004A88) = 32;
  BYTE4(qword_5004A98) = 1;
  LODWORD(qword_5004A98) = 32;
  LOBYTE(dword_5004A0C) = dword_5004A0C & 0x9F | 0x20;
  sub_C53130(&qword_5004A00);
  __cxa_atexit(sub_B2B680, &qword_5004A00, &qword_4A427C0);
  qword_5004920 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5004970 = 0x100000000LL;
  dword_500492C &= 0x8000u;
  qword_5004968 = (__int64)&unk_5004978;
  word_5004930 = 0;
  qword_5004938 = 0;
  dword_5004928 = v7;
  qword_5004940 = 0;
  qword_5004948 = 0;
  qword_5004950 = 0;
  qword_5004958 = 0;
  qword_5004960 = 0;
  qword_5004980 = 0;
  qword_5004988 = (__int64)&unk_50049A0;
  qword_5004990 = 1;
  dword_5004998 = 0;
  byte_500499C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5004970;
  if ( (unsigned __int64)(unsigned int)qword_5004970 + 1 > HIDWORD(qword_5004970) )
  {
    v22 = v8;
    sub_C8D5F0((char *)&unk_5004978 - 16, &unk_5004978, (unsigned int)qword_5004970 + 1LL, 8);
    v9 = (unsigned int)qword_5004970;
    v8 = v22;
  }
  *(_QWORD *)(qword_5004968 + 8 * v9) = v8;
  qword_50049B0 = (__int64)&unk_49D9748;
  qword_5004920 = (__int64)&unk_49DC090;
  qword_50049C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5004970) = qword_5004970 + 1;
  qword_50049E0 = (__int64)nullsub_23;
  qword_50049A8 = 0;
  qword_50049D8 = (__int64)sub_984030;
  qword_50049B8 = 0;
  sub_C53080(&qword_5004920, "spp-split-backedge", 18);
  LOWORD(qword_50049B8) = 256;
  LOBYTE(qword_50049A8) = 0;
  LOBYTE(dword_500492C) = dword_500492C & 0x9F | 0x20;
  sub_C53130(&qword_5004920);
  __cxa_atexit(sub_984900, &qword_5004920, &qword_4A427C0);
  qword_5004840 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5004890 = 0x100000000LL;
  dword_500484C &= 0x8000u;
  qword_5004888 = (__int64)&unk_5004898;
  word_5004850 = 0;
  qword_5004858 = 0;
  dword_5004848 = v10;
  qword_5004860 = 0;
  qword_5004868 = 0;
  qword_5004870 = 0;
  qword_5004878 = 0;
  qword_5004880 = 0;
  qword_50048A0 = 0;
  qword_50048A8 = (__int64)&unk_50048C0;
  qword_50048B0 = 1;
  dword_50048B8 = 0;
  byte_50048BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5004890;
  if ( (unsigned __int64)(unsigned int)qword_5004890 + 1 > HIDWORD(qword_5004890) )
  {
    v23 = v11;
    sub_C8D5F0((char *)&unk_5004898 - 16, &unk_5004898, (unsigned int)qword_5004890 + 1LL, 8);
    v12 = (unsigned int)qword_5004890;
    v11 = v23;
  }
  *(_QWORD *)(qword_5004888 + 8 * v12) = v11;
  qword_50048D0 = (__int64)&unk_49D9748;
  qword_5004840 = (__int64)&unk_49DC090;
  qword_50048E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5004890) = qword_5004890 + 1;
  qword_5004900 = (__int64)nullsub_23;
  qword_50048C8 = 0;
  qword_50048F8 = (__int64)sub_984030;
  qword_50048D8 = 0;
  sub_C53080(&qword_5004840, "spp-no-entry", 12);
  LOWORD(qword_50048D8) = 256;
  LOBYTE(qword_50048C8) = 0;
  LOBYTE(dword_500484C) = dword_500484C & 0x9F | 0x20;
  sub_C53130(&qword_5004840);
  __cxa_atexit(sub_984900, &qword_5004840, &qword_4A427C0);
  qword_5004760 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50047B0 = 0x100000000LL;
  dword_500476C &= 0x8000u;
  qword_50047A8 = (__int64)&unk_50047B8;
  word_5004770 = 0;
  qword_5004778 = 0;
  dword_5004768 = v13;
  qword_5004780 = 0;
  qword_5004788 = 0;
  qword_5004790 = 0;
  qword_5004798 = 0;
  qword_50047A0 = 0;
  qword_50047C0 = 0;
  qword_50047C8 = (__int64)&unk_50047E0;
  qword_50047D0 = 1;
  dword_50047D8 = 0;
  byte_50047DC = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_50047B0;
  if ( (unsigned __int64)(unsigned int)qword_50047B0 + 1 > HIDWORD(qword_50047B0) )
  {
    v24 = v14;
    sub_C8D5F0((char *)&unk_50047B8 - 16, &unk_50047B8, (unsigned int)qword_50047B0 + 1LL, 8);
    v15 = (unsigned int)qword_50047B0;
    v14 = v24;
  }
  *(_QWORD *)(qword_50047A8 + 8 * v15) = v14;
  qword_50047F0 = (__int64)&unk_49D9748;
  qword_5004760 = (__int64)&unk_49DC090;
  qword_5004800 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50047B0) = qword_50047B0 + 1;
  qword_5004820 = (__int64)nullsub_23;
  qword_50047E8 = 0;
  qword_5004818 = (__int64)sub_984030;
  qword_50047F8 = 0;
  sub_C53080(&qword_5004760, "spp-no-call", 11);
  LOWORD(qword_50047F8) = 256;
  LOBYTE(qword_50047E8) = 0;
  LOBYTE(dword_500476C) = dword_500476C & 0x9F | 0x20;
  sub_C53130(&qword_5004760);
  __cxa_atexit(sub_984900, &qword_5004760, &qword_4A427C0);
  qword_5004680 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50046D0 = 0x100000000LL;
  dword_500468C &= 0x8000u;
  word_5004690 = 0;
  qword_50046C8 = (__int64)&unk_50046D8;
  qword_5004698 = 0;
  dword_5004688 = v16;
  qword_50046A0 = 0;
  qword_50046A8 = 0;
  qword_50046B0 = 0;
  qword_50046B8 = 0;
  qword_50046C0 = 0;
  qword_50046E0 = 0;
  qword_50046E8 = (__int64)&unk_5004700;
  qword_50046F0 = 1;
  dword_50046F8 = 0;
  byte_50046FC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_50046D0;
  v19 = (unsigned int)qword_50046D0 + 1LL;
  if ( v19 > HIDWORD(qword_50046D0) )
  {
    sub_C8D5F0((char *)&unk_50046D8 - 16, &unk_50046D8, v19, 8);
    v18 = (unsigned int)qword_50046D0;
  }
  *(_QWORD *)(qword_50046C8 + 8 * v18) = v17;
  qword_5004710 = (__int64)&unk_49D9748;
  qword_5004680 = (__int64)&unk_49DC090;
  qword_5004720 = (__int64)&unk_49DC1D0;
  LODWORD(qword_50046D0) = qword_50046D0 + 1;
  qword_5004740 = (__int64)nullsub_23;
  qword_5004708 = 0;
  qword_5004738 = (__int64)sub_984030;
  qword_5004718 = 0;
  sub_C53080(&qword_5004680, "spp-no-backedge", 15);
  LOBYTE(qword_5004708) = 0;
  LOBYTE(dword_500468C) = dword_500468C & 0x9F | 0x20;
  LOWORD(qword_5004718) = 256;
  sub_C53130(&qword_5004680);
  return __cxa_atexit(sub_984900, &qword_5004680, &qword_4A427C0);
}
