// Function: ctor_380
// Address: 0x519c80
//
int ctor_380()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4FDB520 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB570 = 0x100000000LL;
  dword_4FDB52C &= 0x8000u;
  word_4FDB530 = 0;
  qword_4FDB538 = 0;
  qword_4FDB540 = 0;
  dword_4FDB528 = v0;
  qword_4FDB548 = 0;
  qword_4FDB550 = 0;
  qword_4FDB558 = 0;
  qword_4FDB560 = 0;
  qword_4FDB568 = (__int64)&unk_4FDB578;
  qword_4FDB580 = 0;
  qword_4FDB588 = (__int64)&unk_4FDB5A0;
  qword_4FDB590 = 1;
  dword_4FDB598 = 0;
  byte_4FDB59C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FDB570;
  v3 = (unsigned int)qword_4FDB570 + 1LL;
  if ( v3 > HIDWORD(qword_4FDB570) )
  {
    sub_C8D5F0((char *)&unk_4FDB578 - 16, &unk_4FDB578, v3, 8);
    v2 = (unsigned int)qword_4FDB570;
  }
  *(_QWORD *)(qword_4FDB568 + 8 * v2) = v1;
  qword_4FDB5B0 = (__int64)&unk_49D9748;
  qword_4FDB520 = (__int64)&unk_49DC090;
  LODWORD(qword_4FDB570) = qword_4FDB570 + 1;
  qword_4FDB5A8 = 0;
  qword_4FDB5C0 = (__int64)&unk_49DC1D0;
  qword_4FDB5B8 = 0;
  qword_4FDB5E0 = (__int64)nullsub_23;
  qword_4FDB5D8 = (__int64)sub_984030;
  sub_C53080(&qword_4FDB520, "da-delinearize", 14);
  LOWORD(qword_4FDB5B8) = 257;
  LOBYTE(qword_4FDB5A8) = 1;
  qword_4FDB550 = 36;
  LOBYTE(dword_4FDB52C) = dword_4FDB52C & 0x9F | 0x20;
  qword_4FDB548 = (__int64)"Try to delinearize array references.";
  sub_C53130(&qword_4FDB520);
  __cxa_atexit(sub_984900, &qword_4FDB520, &qword_4A427C0);
  qword_4FDB440 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB490 = 0x100000000LL;
  word_4FDB450 = 0;
  dword_4FDB44C &= 0x8000u;
  qword_4FDB458 = 0;
  qword_4FDB460 = 0;
  dword_4FDB448 = v4;
  qword_4FDB468 = 0;
  qword_4FDB470 = 0;
  qword_4FDB478 = 0;
  qword_4FDB480 = 0;
  qword_4FDB488 = (__int64)&unk_4FDB498;
  qword_4FDB4A0 = 0;
  qword_4FDB4A8 = (__int64)&unk_4FDB4C0;
  qword_4FDB4B0 = 1;
  dword_4FDB4B8 = 0;
  byte_4FDB4BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FDB490;
  if ( (unsigned __int64)(unsigned int)qword_4FDB490 + 1 > HIDWORD(qword_4FDB490) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FDB498 - 16, &unk_4FDB498, (unsigned int)qword_4FDB490 + 1LL, 8);
    v6 = (unsigned int)qword_4FDB490;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FDB488 + 8 * v6) = v5;
  qword_4FDB4D0 = (__int64)&unk_49D9748;
  qword_4FDB440 = (__int64)&unk_49DC090;
  LODWORD(qword_4FDB490) = qword_4FDB490 + 1;
  qword_4FDB4C8 = 0;
  qword_4FDB4E0 = (__int64)&unk_49DC1D0;
  qword_4FDB4D8 = 0;
  qword_4FDB500 = (__int64)nullsub_23;
  qword_4FDB4F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FDB440, "da-disable-delinearization-checks", 33);
  qword_4FDB470 = 250;
  LOBYTE(dword_4FDB44C) = dword_4FDB44C & 0x9F | 0x20;
  qword_4FDB468 = (__int64)"Disable checks that try to statically verify validity of delinearized subscripts. Enabling th"
                           "is option may result in incorrect dependence vectors for languages that allow the subscript o"
                           "f one dimension to underflow or overflow into another dimension.";
  sub_C53130(&qword_4FDB440);
  __cxa_atexit(sub_984900, &qword_4FDB440, &qword_4A427C0);
  qword_4FDB360 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FDB36C &= 0x8000u;
  word_4FDB370 = 0;
  qword_4FDB3B0 = 0x100000000LL;
  qword_4FDB378 = 0;
  qword_4FDB380 = 0;
  qword_4FDB388 = 0;
  dword_4FDB368 = v7;
  qword_4FDB390 = 0;
  qword_4FDB398 = 0;
  qword_4FDB3A0 = 0;
  qword_4FDB3A8 = (__int64)&unk_4FDB3B8;
  qword_4FDB3C0 = 0;
  qword_4FDB3C8 = (__int64)&unk_4FDB3E0;
  qword_4FDB3D0 = 1;
  dword_4FDB3D8 = 0;
  byte_4FDB3DC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FDB3B0;
  v10 = (unsigned int)qword_4FDB3B0 + 1LL;
  if ( v10 > HIDWORD(qword_4FDB3B0) )
  {
    sub_C8D5F0((char *)&unk_4FDB3B8 - 16, &unk_4FDB3B8, v10, 8);
    v9 = (unsigned int)qword_4FDB3B0;
  }
  *(_QWORD *)(qword_4FDB3A8 + 8 * v9) = v8;
  LODWORD(qword_4FDB3B0) = qword_4FDB3B0 + 1;
  qword_4FDB3E8 = 0;
  qword_4FDB3F0 = (__int64)&unk_49D9728;
  qword_4FDB3F8 = 0;
  qword_4FDB360 = (__int64)&unk_49DBF10;
  qword_4FDB400 = (__int64)&unk_49DC290;
  qword_4FDB420 = (__int64)nullsub_24;
  qword_4FDB418 = (__int64)sub_984050;
  sub_C53080(&qword_4FDB360, "da-miv-max-level-threshold", 26);
  LODWORD(qword_4FDB3E8) = 7;
  BYTE4(qword_4FDB3F8) = 1;
  LODWORD(qword_4FDB3F8) = 7;
  qword_4FDB390 = 88;
  LOBYTE(dword_4FDB36C) = dword_4FDB36C & 0x9F | 0x20;
  qword_4FDB388 = (__int64)"Maximum depth allowed for the recursive algorithm used to explore MIV direction vectors.";
  sub_C53130(&qword_4FDB360);
  return __cxa_atexit(sub_984970, &qword_4FDB360, &qword_4A427C0);
}
