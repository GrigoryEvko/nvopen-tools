// Function: ctor_473
// Address: 0x54d740
//
int ctor_473()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_50038C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500393C = 1;
  qword_5003910 = 0x100000000LL;
  dword_50038CC &= 0x8000u;
  qword_50038D8 = 0;
  qword_50038E0 = 0;
  qword_50038E8 = 0;
  dword_50038C8 = v0;
  word_50038D0 = 0;
  qword_50038F0 = 0;
  qword_50038F8 = 0;
  qword_5003900 = 0;
  qword_5003908 = (__int64)&unk_5003918;
  qword_5003920 = 0;
  qword_5003928 = (__int64)&unk_5003940;
  qword_5003930 = 1;
  dword_5003938 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5003910;
  v3 = (unsigned int)qword_5003910 + 1LL;
  if ( v3 > HIDWORD(qword_5003910) )
  {
    sub_C8D5F0((char *)&unk_5003918 - 16, &unk_5003918, v3, 8);
    v2 = (unsigned int)qword_5003910;
  }
  *(_QWORD *)(qword_5003908 + 8 * v2) = v1;
  LODWORD(qword_5003910) = qword_5003910 + 1;
  qword_5003948 = 0;
  qword_5003950 = (__int64)&unk_49E5940;
  qword_5003958 = 0;
  qword_50038C0 = (__int64)&unk_49E5960;
  qword_5003960 = (__int64)&unk_49DC320;
  qword_5003980 = (__int64)nullsub_385;
  qword_5003978 = (__int64)sub_1038930;
  sub_C53080(&qword_50038C0, "licm-versioning-invariant-threshold", 35);
  qword_50038E8 = (__int64)"LoopVersioningLICM's minimum allowed percentage of possible invariant instructions per loop";
  LODWORD(qword_5003948) = 1103626240;
  LODWORD(qword_5003958) = 1103626240;
  BYTE4(qword_5003958) = 1;
  LOBYTE(dword_50038CC) = dword_50038CC & 0x9F | 0x20;
  qword_50038F0 = 91;
  sub_C53130(&qword_50038C0);
  __cxa_atexit(sub_1038DB0, &qword_50038C0, &qword_4A427C0);
  qword_50037E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50037EC &= 0x8000u;
  word_50037F0 = 0;
  qword_5003830 = 0x100000000LL;
  qword_50037F8 = 0;
  qword_5003800 = 0;
  qword_5003808 = 0;
  dword_50037E8 = v4;
  qword_5003810 = 0;
  qword_5003818 = 0;
  qword_5003820 = 0;
  qword_5003828 = (__int64)&unk_5003838;
  qword_5003840 = 0;
  qword_5003848 = (__int64)&unk_5003860;
  qword_5003850 = 1;
  dword_5003858 = 0;
  byte_500385C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5003830;
  v7 = (unsigned int)qword_5003830 + 1LL;
  if ( v7 > HIDWORD(qword_5003830) )
  {
    sub_C8D5F0((char *)&unk_5003838 - 16, &unk_5003838, v7, 8);
    v6 = (unsigned int)qword_5003830;
  }
  *(_QWORD *)(qword_5003828 + 8 * v6) = v5;
  LODWORD(qword_5003830) = qword_5003830 + 1;
  qword_5003868 = 0;
  qword_5003870 = (__int64)&unk_49D9728;
  qword_5003878 = 0;
  qword_50037E0 = (__int64)&unk_49DBF10;
  qword_5003880 = (__int64)&unk_49DC290;
  qword_50038A0 = (__int64)nullsub_24;
  qword_5003898 = (__int64)sub_984050;
  sub_C53080(&qword_50037E0, "licm-versioning-max-depth-threshold", 35);
  qword_5003810 = 66;
  qword_5003808 = (__int64)"LoopVersioningLICM's threshold for maximum allowed loop nest/depth";
  LODWORD(qword_5003868) = 2;
  BYTE4(qword_5003878) = 1;
  LODWORD(qword_5003878) = 2;
  LOBYTE(dword_50037EC) = dword_50037EC & 0x9F | 0x20;
  sub_C53130(&qword_50037E0);
  return __cxa_atexit(sub_984970, &qword_50037E0, &qword_4A427C0);
}
