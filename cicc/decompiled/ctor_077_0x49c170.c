// Function: ctor_077
// Address: 0x49c170
//
int ctor_077()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // r13
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  qword_4F8E4E0 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8E4EC = word_4F8E4EC & 0x8000;
  qword_4F8E528[1] = 0x100000000LL;
  unk_4F8E4E8 = v0;
  unk_4F8E4F0 = 0;
  unk_4F8E4F8 = 0;
  unk_4F8E500 = 0;
  unk_4F8E508 = 0;
  unk_4F8E510 = 0;
  unk_4F8E518 = 0;
  unk_4F8E520 = 0;
  qword_4F8E528[0] = &qword_4F8E528[2];
  qword_4F8E528[3] = 0;
  qword_4F8E528[4] = &qword_4F8E528[7];
  qword_4F8E528[5] = 1;
  LODWORD(qword_4F8E528[6]) = 0;
  BYTE4(qword_4F8E528[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F8E528[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8E528[1]) + 1 > HIDWORD(qword_4F8E528[1]) )
  {
    sub_C8D5F0(qword_4F8E528, &qword_4F8E528[2], LODWORD(qword_4F8E528[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F8E528[1]);
  }
  *(_QWORD *)(qword_4F8E528[0] + 8 * v2) = v1;
  qword_4F8E528[9] = &unk_49D9748;
  ++LODWORD(qword_4F8E528[1]);
  qword_4F8E528[8] = 0;
  qword_4F8E4E0 = &unk_49DC090;
  qword_4F8E528[10] = 0;
  qword_4F8E528[11] = &unk_49DC1D0;
  qword_4F8E528[15] = nullsub_23;
  qword_4F8E528[14] = sub_984030;
  sub_C53080(&qword_4F8E4E0, "check-bfi-unknown-block-queries", 31);
  LOWORD(qword_4F8E528[10]) = 256;
  LOBYTE(qword_4F8E528[8]) = 0;
  unk_4F8E510 = 89;
  LOBYTE(word_4F8E4EC) = word_4F8E4EC & 0x9F | 0x20;
  unk_4F8E508 = "Check if block frequency is queried for an unknown block for debugging missed BFI updates";
  sub_C53130(&qword_4F8E4E0);
  __cxa_atexit(sub_984900, &qword_4F8E4E0, &qword_4A427C0);
  qword_4F8E400 = &unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8E40C = word_4F8E40C & 0x8000;
  qword_4F8E448[1] = 0x100000000LL;
  unk_4F8E408 = v3;
  unk_4F8E410 = 0;
  unk_4F8E418 = 0;
  unk_4F8E420 = 0;
  unk_4F8E428 = 0;
  unk_4F8E430 = 0;
  unk_4F8E438 = 0;
  unk_4F8E440 = 0;
  qword_4F8E448[0] = &qword_4F8E448[2];
  qword_4F8E448[3] = 0;
  qword_4F8E448[4] = &qword_4F8E448[7];
  qword_4F8E448[5] = 1;
  LODWORD(qword_4F8E448[6]) = 0;
  BYTE4(qword_4F8E448[6]) = 1;
  v4 = sub_C57470();
  v5 = LODWORD(qword_4F8E448[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8E448[1]) + 1 > HIDWORD(qword_4F8E448[1]) )
  {
    v13 = v4;
    sub_C8D5F0(qword_4F8E448, &qword_4F8E448[2], LODWORD(qword_4F8E448[1]) + 1LL, 8);
    v5 = LODWORD(qword_4F8E448[1]);
    v4 = v13;
  }
  *(_QWORD *)(qword_4F8E448[0] + 8 * v5) = v4;
  qword_4F8E448[9] = &unk_49D9748;
  ++LODWORD(qword_4F8E448[1]);
  qword_4F8E448[8] = 0;
  qword_4F8E400 = &unk_49DC090;
  qword_4F8E448[10] = 0;
  qword_4F8E448[11] = &unk_49DC1D0;
  qword_4F8E448[15] = nullsub_23;
  qword_4F8E448[14] = sub_984030;
  sub_C53080(&qword_4F8E400, "use-iterative-bfi-inference", 27);
  unk_4F8E430 = 62;
  LOBYTE(word_4F8E40C) = word_4F8E40C & 0x9F | 0x20;
  unk_4F8E428 = "Apply an iterative post-processing to infer correct BFI counts";
  sub_C53130(&qword_4F8E400);
  __cxa_atexit(sub_984900, &qword_4F8E400, &qword_4A427C0);
  qword_4F8E320 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8E32C = word_4F8E32C & 0x8000;
  unk_4F8E328 = v6;
  qword_4F8E368[1] = 0x100000000LL;
  unk_4F8E330 = 0;
  unk_4F8E338 = 0;
  unk_4F8E340 = 0;
  unk_4F8E348 = 0;
  unk_4F8E350 = 0;
  unk_4F8E358 = 0;
  unk_4F8E360 = 0;
  qword_4F8E368[0] = &qword_4F8E368[2];
  qword_4F8E368[3] = 0;
  qword_4F8E368[4] = &qword_4F8E368[7];
  qword_4F8E368[5] = 1;
  LODWORD(qword_4F8E368[6]) = 0;
  BYTE4(qword_4F8E368[6]) = 1;
  v7 = sub_C57470();
  v8 = LODWORD(qword_4F8E368[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8E368[1]) + 1 > HIDWORD(qword_4F8E368[1]) )
  {
    sub_C8D5F0(qword_4F8E368, &qword_4F8E368[2], LODWORD(qword_4F8E368[1]) + 1LL, 8);
    v8 = LODWORD(qword_4F8E368[1]);
  }
  *(_QWORD *)(qword_4F8E368[0] + 8 * v8) = v7;
  ++LODWORD(qword_4F8E368[1]);
  qword_4F8E368[8] = 0;
  qword_4F8E368[9] = &unk_49D9728;
  qword_4F8E368[10] = 0;
  qword_4F8E320 = &unk_49DBF10;
  qword_4F8E368[11] = &unk_49DC290;
  qword_4F8E368[15] = nullsub_24;
  qword_4F8E368[14] = sub_984050;
  sub_C53080(&qword_4F8E320, "iterative-bfi-max-iterations-per-block", 38);
  LODWORD(qword_4F8E368[8]) = 1000;
  BYTE4(qword_4F8E368[10]) = 1;
  LODWORD(qword_4F8E368[10]) = 1000;
  unk_4F8E350 = 66;
  LOBYTE(word_4F8E32C) = word_4F8E32C & 0x9F | 0x20;
  unk_4F8E348 = "Iterative inference: maximum number of update iterations per block";
  sub_C53130(&qword_4F8E320);
  __cxa_atexit(sub_984970, &qword_4F8E320, &qword_4A427C0);
  qword_4F8E240 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8E24C = word_4F8E24C & 0x8000;
  unk_4F8E250 = 0;
  qword_4F8E288[1] = 0x100000000LL;
  unk_4F8E248 = v9;
  unk_4F8E258 = 0;
  unk_4F8E260 = 0;
  unk_4F8E268 = 0;
  unk_4F8E270 = 0;
  unk_4F8E278 = 0;
  unk_4F8E280 = 0;
  qword_4F8E288[0] = &qword_4F8E288[2];
  qword_4F8E288[3] = 0;
  qword_4F8E288[4] = &qword_4F8E288[7];
  qword_4F8E288[5] = 1;
  LODWORD(qword_4F8E288[6]) = 0;
  BYTE4(qword_4F8E288[6]) = 1;
  v10 = sub_C57470();
  v11 = LODWORD(qword_4F8E288[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8E288[1]) + 1 > HIDWORD(qword_4F8E288[1]) )
  {
    sub_C8D5F0(qword_4F8E288, &qword_4F8E288[2], LODWORD(qword_4F8E288[1]) + 1LL, 8);
    v11 = LODWORD(qword_4F8E288[1]);
  }
  *(_QWORD *)(qword_4F8E288[0] + 8 * v11) = v10;
  ++LODWORD(qword_4F8E288[1]);
  qword_4F8E288[8] = 0;
  qword_4F8E288[9] = &unk_49DE5F0;
  qword_4F8E288[10] = 0;
  LOBYTE(qword_4F8E288[11]) = 0;
  qword_4F8E240 = &unk_49DE610;
  qword_4F8E288[12] = &unk_49DC2F0;
  qword_4F8E288[16] = nullsub_190;
  qword_4F8E288[15] = sub_D83E80;
  sub_C53080(&qword_4F8E240, "iterative-bfi-precision", 23);
  LOBYTE(qword_4F8E288[11]) = 1;
  unk_4F8E270 = 127;
  qword_4F8E288[8] = 0x3D719799812DEA11LL;
  LOBYTE(word_4F8E24C) = word_4F8E24C & 0x9F | 0x20;
  unk_4F8E268 = "Iterative inference: delta convergence precision; smaller values typically lead to better results at the"
                " cost of worsen runtime";
  qword_4F8E288[10] = 0x3D719799812DEA11LL;
  sub_C53130(&qword_4F8E240);
  return __cxa_atexit(sub_D84280, &qword_4F8E240, &qword_4A427C0);
}
