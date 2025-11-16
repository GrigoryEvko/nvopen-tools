// Function: ctor_089
// Address: 0x4a0d60
//
int ctor_089()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]

  qword_4F90740 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F90790 = 0x100000000LL;
  dword_4F9074C &= 0x8000u;
  word_4F90750 = 0;
  qword_4F90758 = 0;
  qword_4F90760 = 0;
  dword_4F90748 = v0;
  qword_4F90768 = 0;
  qword_4F90770 = 0;
  qword_4F90778 = 0;
  qword_4F90780 = 0;
  qword_4F90788 = (__int64)&unk_4F90798;
  qword_4F907A0 = 0;
  qword_4F907A8 = (__int64)&unk_4F907C0;
  qword_4F907B0 = 1;
  dword_4F907B8 = 0;
  byte_4F907BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F90790;
  v3 = (unsigned int)qword_4F90790 + 1LL;
  if ( v3 > HIDWORD(qword_4F90790) )
  {
    sub_C8D5F0((char *)&unk_4F90798 - 16, &unk_4F90798, v3, 8);
    v2 = (unsigned int)qword_4F90790;
  }
  *(_QWORD *)(qword_4F90788 + 8 * v2) = v1;
  qword_4F907D0 = (__int64)&unk_49D9728;
  qword_4F90740 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F90790) = qword_4F90790 + 1;
  qword_4F907C8 = 0;
  qword_4F907E0 = (__int64)&unk_49DC290;
  qword_4F907D8 = 0;
  qword_4F90800 = (__int64)nullsub_24;
  qword_4F907F8 = (__int64)sub_984050;
  sub_C53080(&qword_4F90740, "max-aggr-lower-size", 19);
  LODWORD(qword_4F907C8) = 128;
  BYTE4(qword_4F907D8) = 1;
  LODWORD(qword_4F907D8) = 128;
  qword_4F90770 = 61;
  LOBYTE(dword_4F9074C) = dword_4F9074C & 0x9F | 0x20;
  qword_4F90768 = (__int64)"The threshold size below which it's okay to lower aggregates.";
  sub_C53130(&qword_4F90740);
  __cxa_atexit(sub_984970, &qword_4F90740, &qword_4A427C0);
  qword_4F90660 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F906B0 = 0x100000000LL;
  dword_4F9066C &= 0x8000u;
  qword_4F906A8 = (__int64)&unk_4F906B8;
  word_4F90670 = 0;
  qword_4F90678 = 0;
  dword_4F90668 = v4;
  qword_4F90680 = 0;
  qword_4F90688 = 0;
  qword_4F90690 = 0;
  qword_4F90698 = 0;
  qword_4F906A0 = 0;
  qword_4F906C0 = 0;
  qword_4F906C8 = (__int64)&unk_4F906E0;
  qword_4F906D0 = 1;
  dword_4F906D8 = 0;
  byte_4F906DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F906B0;
  v7 = (unsigned int)qword_4F906B0 + 1LL;
  if ( v7 > HIDWORD(qword_4F906B0) )
  {
    sub_C8D5F0((char *)&unk_4F906B8 - 16, &unk_4F906B8, v7, 8);
    v6 = (unsigned int)qword_4F906B0;
  }
  *(_QWORD *)(qword_4F906A8 + 8 * v6) = v5;
  qword_4F906F0 = (__int64)&unk_49D9728;
  qword_4F90660 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F906B0) = qword_4F906B0 + 1;
  qword_4F906E8 = 0;
  qword_4F90700 = (__int64)&unk_49DC290;
  qword_4F906F8 = 0;
  qword_4F90720 = (__int64)nullsub_24;
  qword_4F90718 = (__int64)sub_984050;
  sub_C53080(&qword_4F90660, "aggressive-max-aggr-lower-size", 30);
  LODWORD(qword_4F906E8) = 256;
  BYTE4(qword_4F906F8) = 1;
  LODWORD(qword_4F906F8) = 256;
  qword_4F90690 = 72;
  LOBYTE(dword_4F9066C) = dword_4F9066C & 0x9F | 0x20;
  qword_4F90688 = (__int64)"The aggressive threshold size below which it's okay to lower aggregates.";
  sub_C53130(&qword_4F90660);
  __cxa_atexit(sub_984970, &qword_4F90660, &qword_4A427C0);
  qword_4F90580 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F905D0 = 0x100000000LL;
  dword_4F9058C &= 0x8000u;
  qword_4F905C8 = (__int64)&unk_4F905D8;
  word_4F90590 = 0;
  qword_4F90598 = 0;
  dword_4F90588 = v8;
  qword_4F905A0 = 0;
  qword_4F905A8 = 0;
  qword_4F905B0 = 0;
  qword_4F905B8 = 0;
  qword_4F905C0 = 0;
  qword_4F905E0 = 0;
  qword_4F905E8 = (__int64)&unk_4F90600;
  qword_4F905F0 = 1;
  dword_4F905F8 = 0;
  byte_4F905FC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4F905D0;
  v11 = (unsigned int)qword_4F905D0 + 1LL;
  if ( v11 > HIDWORD(qword_4F905D0) )
  {
    sub_C8D5F0((char *)&unk_4F905D8 - 16, &unk_4F905D8, v11, 8);
    v10 = (unsigned int)qword_4F905D0;
  }
  *(_QWORD *)(qword_4F905C8 + 8 * v10) = v9;
  LODWORD(qword_4F905D0) = qword_4F905D0 + 1;
  qword_4F90608 = 0;
  qword_4F90610 = (__int64)&unk_49D9748;
  qword_4F90618 = 0;
  qword_4F90580 = (__int64)&unk_49DC090;
  qword_4F90620 = (__int64)&unk_49DC1D0;
  qword_4F90640 = (__int64)nullsub_23;
  qword_4F90638 = (__int64)sub_984030;
  sub_C53080(&qword_4F90580, "instcombine-merge-stores-from-aggr", 34);
  LOWORD(qword_4F90618) = 257;
  LOBYTE(qword_4F90608) = 1;
  qword_4F905B0 = 49;
  LOBYTE(dword_4F9058C) = dword_4F9058C & 0x9F | 0x20;
  qword_4F905A8 = (__int64)"Whether we merge stores when splitting aggregates";
  sub_C53130(&qword_4F90580);
  __cxa_atexit(sub_984900, &qword_4F90580, &qword_4A427C0);
  qword_4F904A0 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F9051C = 1;
  word_4F904B0 = 0;
  qword_4F904F0 = 0x100000000LL;
  dword_4F904AC &= 0x8000u;
  qword_4F904E8 = (__int64)&unk_4F904F8;
  qword_4F904B8 = 0;
  dword_4F904A8 = v12;
  qword_4F904C0 = 0;
  qword_4F904C8 = 0;
  qword_4F904D0 = 0;
  qword_4F904D8 = 0;
  qword_4F904E0 = 0;
  qword_4F90500 = 0;
  qword_4F90508 = (__int64)&unk_4F90520;
  qword_4F90510 = 1;
  dword_4F90518 = 0;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4F904F0;
  if ( (unsigned __int64)(unsigned int)qword_4F904F0 + 1 > HIDWORD(qword_4F904F0) )
  {
    v19 = v13;
    sub_C8D5F0((char *)&unk_4F904F8 - 16, &unk_4F904F8, (unsigned int)qword_4F904F0 + 1LL, 8);
    v14 = (unsigned int)qword_4F904F0;
    v13 = v19;
  }
  *(_QWORD *)(qword_4F904E8 + 8 * v14) = v13;
  qword_4F90530 = (__int64)&unk_49D9728;
  qword_4F904A0 = (__int64)&unk_49DBF10;
  LODWORD(qword_4F904F0) = qword_4F904F0 + 1;
  qword_4F90528 = 0;
  qword_4F90540 = (__int64)&unk_49DC290;
  qword_4F90538 = 0;
  qword_4F90560 = (__int64)nullsub_24;
  qword_4F90558 = (__int64)sub_984050;
  sub_C53080(&qword_4F904A0, "instcombine-max-copied-from-constant-users", 42);
  LODWORD(qword_4F90528) = 300;
  qword_4F904C8 = (__int64)"Maximum users to visit in copy from constant transform";
  BYTE4(qword_4F90538) = 1;
  LODWORD(qword_4F90538) = 300;
  qword_4F904D0 = 54;
  LOBYTE(dword_4F904AC) = dword_4F904AC & 0x9F | 0x20;
  sub_C53130(&qword_4F904A0);
  __cxa_atexit(sub_984970, &qword_4F904A0, &qword_4A427C0);
  qword_4F903C0 = &unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F903CC = word_4F903CC & 0x8000;
  unk_4F903C8 = v15;
  qword_4F90408[1] = 0x100000000LL;
  unk_4F903D0 = 0;
  unk_4F903D8 = 0;
  unk_4F903E0 = 0;
  unk_4F903E8 = 0;
  unk_4F903F0 = 0;
  unk_4F903F8 = 0;
  unk_4F90400 = 0;
  qword_4F90408[0] = &qword_4F90408[2];
  qword_4F90408[3] = 0;
  qword_4F90408[4] = &qword_4F90408[7];
  qword_4F90408[5] = 1;
  LODWORD(qword_4F90408[6]) = 0;
  BYTE4(qword_4F90408[6]) = 1;
  v16 = sub_C57470();
  v17 = LODWORD(qword_4F90408[1]);
  if ( (unsigned __int64)LODWORD(qword_4F90408[1]) + 1 > HIDWORD(qword_4F90408[1]) )
  {
    sub_C8D5F0(qword_4F90408, &qword_4F90408[2], LODWORD(qword_4F90408[1]) + 1LL, 8);
    v17 = LODWORD(qword_4F90408[1]);
  }
  *(_QWORD *)(qword_4F90408[0] + 8 * v17) = v16;
  ++LODWORD(qword_4F90408[1]);
  qword_4F90408[8] = 0;
  qword_4F90408[9] = &unk_49D9748;
  qword_4F90408[10] = 0;
  qword_4F903C0 = &unk_49DC090;
  qword_4F90408[11] = &unk_49DC1D0;
  qword_4F90408[15] = nullsub_23;
  qword_4F90408[14] = sub_984030;
  sub_C53080(&qword_4F903C0, "enable-infer-alignment-pass", 27);
  LOBYTE(qword_4F90408[8]) = 0;
  LOWORD(qword_4F90408[10]) = 256;
  unk_4F903F0 = 76;
  LOBYTE(word_4F903CC) = word_4F903CC & 0x98 | 0x21;
  unk_4F903E8 = "Enable the InferAlignment pass, disabling alignment inference in InstCombine";
  sub_C53130(&qword_4F903C0);
  return __cxa_atexit(sub_984900, &qword_4F903C0, &qword_4A427C0);
}
