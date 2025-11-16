// Function: ctor_067
// Address: 0x496500
//
int ctor_067()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // r13
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // rax
  int result; // eax
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]

  qword_4F8AE00 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8AE0C = word_4F8AE0C & 0x8000;
  qword_4F8AE48[1] = 0x100000000LL;
  unk_4F8AE08 = v0;
  unk_4F8AE10 = 0;
  unk_4F8AE18 = 0;
  unk_4F8AE20 = 0;
  unk_4F8AE28 = 0;
  unk_4F8AE30 = 0;
  unk_4F8AE38 = 0;
  unk_4F8AE40 = 0;
  qword_4F8AE48[0] = &qword_4F8AE48[2];
  qword_4F8AE48[3] = 0;
  qword_4F8AE48[4] = &qword_4F8AE48[7];
  qword_4F8AE48[5] = 1;
  LODWORD(qword_4F8AE48[6]) = 0;
  BYTE4(qword_4F8AE48[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F8AE48[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8AE48[1]) + 1 > HIDWORD(qword_4F8AE48[1]) )
  {
    sub_C8D5F0(qword_4F8AE48, &qword_4F8AE48[2], LODWORD(qword_4F8AE48[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F8AE48[1]);
  }
  *(_QWORD *)(qword_4F8AE48[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F8AE48[1]);
  qword_4F8AE48[8] = 0;
  qword_4F8AE48[9] = &unk_49D9748;
  qword_4F8AE48[10] = 0;
  qword_4F8AE00 = &unk_49DC090;
  qword_4F8AE48[11] = &unk_49DC1D0;
  qword_4F8AE48[15] = nullsub_23;
  qword_4F8AE48[14] = sub_984030;
  sub_C53080(&qword_4F8AE00, "profile-summary-contextless", 27);
  unk_4F8AE30 = 53;
  LOBYTE(word_4F8AE0C) = word_4F8AE0C & 0x9F | 0x20;
  unk_4F8AE28 = "Merge context profiles before calculating thresholds.";
  sub_C53130(&qword_4F8AE00);
  __cxa_atexit(sub_984900, &qword_4F8AE00, &qword_4A427C0);
  qword_4F8AD20 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8AD2C = word_4F8AD2C & 0x8000;
  qword_4F8AD68[1] = 0x100000000LL;
  unk_4F8AD28 = v3;
  unk_4F8AD30 = 0;
  unk_4F8AD38 = 0;
  unk_4F8AD40 = 0;
  unk_4F8AD48 = 0;
  unk_4F8AD50 = 0;
  unk_4F8AD58 = 0;
  unk_4F8AD60 = 0;
  qword_4F8AD68[0] = &qword_4F8AD68[2];
  qword_4F8AD68[3] = 0;
  qword_4F8AD68[4] = &qword_4F8AD68[7];
  qword_4F8AD68[5] = 1;
  LODWORD(qword_4F8AD68[6]) = 0;
  BYTE4(qword_4F8AD68[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F8AD68[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8AD68[1]) + 1 > HIDWORD(qword_4F8AD68[1]) )
  {
    sub_C8D5F0(qword_4F8AD68, &qword_4F8AD68[2], LODWORD(qword_4F8AD68[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F8AD68[1]);
  }
  *(_QWORD *)(qword_4F8AD68[0] + 8 * v5) = v4;
  qword_4F8AD68[9] = &unk_49DA090;
  ++LODWORD(qword_4F8AD68[1]);
  qword_4F8AD68[8] = 0;
  qword_4F8AD20 = &unk_49DBF90;
  qword_4F8AD68[10] = 0;
  qword_4F8AD68[11] = &unk_49DC230;
  qword_4F8AD68[15] = nullsub_58;
  qword_4F8AD68[14] = sub_B2B5F0;
  sub_C53080(&qword_4F8AD20, "profile-summary-cutoff-hot", 26);
  LODWORD(qword_4F8AD68[8]) = 990000;
  BYTE4(qword_4F8AD68[10]) = 1;
  LODWORD(qword_4F8AD68[10]) = 990000;
  unk_4F8AD50 = 88;
  LOBYTE(word_4F8AD2C) = word_4F8AD2C & 0x9F | 0x20;
  unk_4F8AD48 = "A count is hot if it exceeds the minimum count to reach this percentile of total counts.";
  sub_C53130(&qword_4F8AD20);
  __cxa_atexit(sub_B2B680, &qword_4F8AD20, &qword_4A427C0);
  qword_4F8AC40 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8AC4C = word_4F8AC4C & 0x8000;
  qword_4F8AC88[1] = 0x100000000LL;
  unk_4F8AC48 = v6;
  unk_4F8AC50 = 0;
  unk_4F8AC58 = 0;
  unk_4F8AC60 = 0;
  unk_4F8AC68 = 0;
  unk_4F8AC70 = 0;
  unk_4F8AC78 = 0;
  unk_4F8AC80 = 0;
  qword_4F8AC88[0] = &qword_4F8AC88[2];
  qword_4F8AC88[3] = 0;
  qword_4F8AC88[4] = &qword_4F8AC88[7];
  qword_4F8AC88[5] = 1;
  LODWORD(qword_4F8AC88[6]) = 0;
  BYTE4(qword_4F8AC88[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_4F8AC88[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8AC88[1]) + 1 > HIDWORD(qword_4F8AC88[1]) )
  {
    v22 = v7;
    sub_C8D5F0(qword_4F8AC88, &qword_4F8AC88[2], LODWORD(qword_4F8AC88[1]) + 1LL, 8);
    v8 = LODWORD(qword_4F8AC88[1]);
    v7 = v22;
  }
  *(_QWORD *)(qword_4F8AC88[0] + 8 * v8) = v7;
  qword_4F8AC88[9] = &unk_49DA090;
  ++LODWORD(qword_4F8AC88[1]);
  qword_4F8AC88[8] = 0;
  qword_4F8AC40 = &unk_49DBF90;
  qword_4F8AC88[10] = 0;
  qword_4F8AC88[11] = &unk_49DC230;
  qword_4F8AC88[15] = nullsub_58;
  qword_4F8AC88[14] = sub_B2B5F0;
  sub_C53080(&qword_4F8AC40, "profile-summary-cutoff-cold", 27);
  BYTE4(qword_4F8AC88[10]) = 1;
  LODWORD(qword_4F8AC88[8]) = 999999;
  unk_4F8AC70 = 90;
  LODWORD(qword_4F8AC88[10]) = 999999;
  LOBYTE(word_4F8AC4C) = word_4F8AC4C & 0x9F | 0x20;
  unk_4F8AC68 = "A count is cold if it is below the minimum count to reach this percentile of total counts.";
  sub_C53130(&qword_4F8AC40);
  __cxa_atexit(sub_B2B680, &qword_4F8AC40, &qword_4A427C0);
  qword_4F8AB60 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8AB6C = word_4F8AB6C & 0x8000;
  qword_4F8ABA8[1] = 0x100000000LL;
  unk_4F8AB68 = v9;
  unk_4F8AB70 = 0;
  unk_4F8AB78 = 0;
  unk_4F8AB80 = 0;
  unk_4F8AB88 = 0;
  unk_4F8AB90 = 0;
  unk_4F8AB98 = 0;
  unk_4F8ABA0 = 0;
  qword_4F8ABA8[0] = &qword_4F8ABA8[2];
  qword_4F8ABA8[3] = 0;
  qword_4F8ABA8[4] = &qword_4F8ABA8[7];
  qword_4F8ABA8[5] = 1;
  LODWORD(qword_4F8ABA8[6]) = 0;
  BYTE4(qword_4F8ABA8[6]) = 1;
  v10 = sub_C57470();
  v11 = LODWORD(qword_4F8ABA8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8ABA8[1]) + 1 > HIDWORD(qword_4F8ABA8[1]) )
  {
    sub_C8D5F0(qword_4F8ABA8, &qword_4F8ABA8[2], LODWORD(qword_4F8ABA8[1]) + 1LL, 8);
    v11 = LODWORD(qword_4F8ABA8[1]);
  }
  *(_QWORD *)(qword_4F8ABA8[0] + 8 * v11) = v10;
  qword_4F8ABA8[9] = &unk_49D9728;
  ++LODWORD(qword_4F8ABA8[1]);
  qword_4F8ABA8[8] = 0;
  qword_4F8AB60 = &unk_49DBF10;
  qword_4F8ABA8[10] = 0;
  qword_4F8ABA8[11] = &unk_49DC290;
  qword_4F8ABA8[15] = nullsub_24;
  qword_4F8ABA8[14] = sub_984050;
  sub_C53080(&qword_4F8AB60, "profile-summary-huge-working-set-size-threshold", 47);
  LODWORD(qword_4F8ABA8[8]) = 15000;
  BYTE4(qword_4F8ABA8[10]) = 1;
  LODWORD(qword_4F8ABA8[10]) = 15000;
  unk_4F8AB90 = 149;
  LOBYTE(word_4F8AB6C) = word_4F8AB6C & 0x9F | 0x20;
  unk_4F8AB88 = "The code working set size is considered huge if the number of blocks required to reach the -profile-summ"
                "ary-cutoff-hot percentile exceeds this count.";
  sub_C53130(&qword_4F8AB60);
  __cxa_atexit(sub_984970, &qword_4F8AB60, &qword_4A427C0);
  qword_4F8AA80 = &unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8AA8C = word_4F8AA8C & 0x8000;
  qword_4F8AAC8[1] = 0x100000000LL;
  unk_4F8AA88 = v12;
  unk_4F8AA90 = 0;
  unk_4F8AA98 = 0;
  unk_4F8AAA0 = 0;
  unk_4F8AAA8 = 0;
  unk_4F8AAB0 = 0;
  unk_4F8AAB8 = 0;
  unk_4F8AAC0 = 0;
  qword_4F8AAC8[0] = &qword_4F8AAC8[2];
  qword_4F8AAC8[3] = 0;
  qword_4F8AAC8[4] = &qword_4F8AAC8[7];
  qword_4F8AAC8[5] = 1;
  LODWORD(qword_4F8AAC8[6]) = 0;
  BYTE4(qword_4F8AAC8[6]) = 1;
  v13 = sub_C57470();
  v14 = LODWORD(qword_4F8AAC8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8AAC8[1]) + 1 > HIDWORD(qword_4F8AAC8[1]) )
  {
    v23 = v13;
    sub_C8D5F0(qword_4F8AAC8, &qword_4F8AAC8[2], LODWORD(qword_4F8AAC8[1]) + 1LL, 8);
    v14 = LODWORD(qword_4F8AAC8[1]);
    v13 = v23;
  }
  *(_QWORD *)(qword_4F8AAC8[0] + 8 * v14) = v13;
  qword_4F8AAC8[9] = &unk_49D9728;
  ++LODWORD(qword_4F8AAC8[1]);
  qword_4F8AAC8[8] = 0;
  qword_4F8AA80 = &unk_49DBF10;
  qword_4F8AAC8[10] = 0;
  qword_4F8AAC8[11] = &unk_49DC290;
  qword_4F8AAC8[15] = nullsub_24;
  qword_4F8AAC8[14] = sub_984050;
  sub_C53080(&qword_4F8AA80, "profile-summary-large-working-set-size-threshold", 48);
  BYTE4(qword_4F8AAC8[10]) = 1;
  LODWORD(qword_4F8AAC8[8]) = 12500;
  unk_4F8AAB0 = 150;
  LODWORD(qword_4F8AAC8[10]) = 12500;
  LOBYTE(word_4F8AA8C) = word_4F8AA8C & 0x9F | 0x20;
  unk_4F8AAA8 = "The code working set size is considered large if the number of blocks required to reach the -profile-sum"
                "mary-cutoff-hot percentile exceeds this count.";
  sub_C53130(&qword_4F8AA80);
  __cxa_atexit(sub_984970, &qword_4F8AA80, &qword_4A427C0);
  qword_4F8A9A0 = &unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8A9AC = word_4F8A9AC & 0x8000;
  unk_4F8A9A8 = v15;
  qword_4F8A9E8[1] = 0x100000000LL;
  unk_4F8A9B0 = 0;
  unk_4F8A9B8 = 0;
  unk_4F8A9C0 = 0;
  unk_4F8A9C8 = 0;
  unk_4F8A9D0 = 0;
  unk_4F8A9D8 = 0;
  unk_4F8A9E0 = 0;
  qword_4F8A9E8[0] = &qword_4F8A9E8[2];
  qword_4F8A9E8[3] = 0;
  qword_4F8A9E8[4] = &qword_4F8A9E8[7];
  qword_4F8A9E8[5] = 1;
  LODWORD(qword_4F8A9E8[6]) = 0;
  BYTE4(qword_4F8A9E8[6]) = 1;
  v16 = sub_C57470();
  v17 = LODWORD(qword_4F8A9E8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8A9E8[1]) + 1 > HIDWORD(qword_4F8A9E8[1]) )
  {
    sub_C8D5F0(qword_4F8A9E8, &qword_4F8A9E8[2], LODWORD(qword_4F8A9E8[1]) + 1LL, 8);
    v17 = LODWORD(qword_4F8A9E8[1]);
  }
  *(_QWORD *)(qword_4F8A9E8[0] + 8 * v17) = v16;
  qword_4F8A9E8[9] = &unk_49DB998;
  ++LODWORD(qword_4F8A9E8[1]);
  qword_4F8A9E8[8] = 0;
  qword_4F8A9A0 = &unk_49DB9B8;
  qword_4F8A9E8[10] = 0;
  LOBYTE(qword_4F8A9E8[11]) = 0;
  qword_4F8A9E8[12] = &unk_49DC2C0;
  qword_4F8A9E8[16] = nullsub_121;
  qword_4F8A9E8[15] = sub_C1A370;
  sub_C53080(&qword_4F8A9A0, "profile-summary-hot-count", 25);
  unk_4F8A9D0 = 82;
  LOBYTE(word_4F8A9AC) = word_4F8A9AC & 0x9F | 0x40;
  unk_4F8A9C8 = "A fixed hot count that overrides the count derived from profile-summary-cutoff-hot";
  sub_C53130(&qword_4F8A9A0);
  __cxa_atexit(sub_C1A610, &qword_4F8A9A0, &qword_4A427C0);
  qword_4F8A8C0 = &unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8A8CC = word_4F8A8CC & 0x8000;
  unk_4F8A8D0 = 0;
  qword_4F8A908[1] = 0x100000000LL;
  unk_4F8A8C8 = v18;
  unk_4F8A8D8 = 0;
  unk_4F8A8E0 = 0;
  unk_4F8A8E8 = 0;
  unk_4F8A8F0 = 0;
  unk_4F8A8F8 = 0;
  unk_4F8A900 = 0;
  qword_4F8A908[0] = &qword_4F8A908[2];
  qword_4F8A908[3] = 0;
  qword_4F8A908[4] = &qword_4F8A908[7];
  qword_4F8A908[5] = 1;
  LODWORD(qword_4F8A908[6]) = 0;
  BYTE4(qword_4F8A908[6]) = 1;
  v19 = sub_C57470();
  v20 = LODWORD(qword_4F8A908[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8A908[1]) + 1 > HIDWORD(qword_4F8A908[1]) )
  {
    sub_C8D5F0(qword_4F8A908, &qword_4F8A908[2], LODWORD(qword_4F8A908[1]) + 1LL, 8);
    v20 = LODWORD(qword_4F8A908[1]);
  }
  *(_QWORD *)(qword_4F8A908[0] + 8 * v20) = v19;
  qword_4F8A908[9] = &unk_49DB998;
  ++LODWORD(qword_4F8A908[1]);
  LOBYTE(qword_4F8A908[11]) = 0;
  qword_4F8A8C0 = &unk_49DB9B8;
  qword_4F8A908[8] = 0;
  qword_4F8A908[10] = 0;
  qword_4F8A908[12] = &unk_49DC2C0;
  qword_4F8A908[16] = nullsub_121;
  qword_4F8A908[15] = sub_C1A370;
  sub_C53080(&qword_4F8A8C0, "profile-summary-cold-count", 26);
  unk_4F8A8F0 = 84;
  LOBYTE(word_4F8A8CC) = word_4F8A8CC & 0x9F | 0x40;
  unk_4F8A8E8 = "A fixed cold count that overrides the count derived from profile-summary-cutoff-cold";
  sub_C53130(&qword_4F8A8C0);
  result = __cxa_atexit(sub_C1A610, &qword_4F8A8C0, &qword_4A427C0);
  qword_4F8A890 = 0;
  qword_4F8A898 = -1;
  qword_4F8A8A0 = 0;
  return result;
}
