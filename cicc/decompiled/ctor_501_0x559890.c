// Function: ctor_501
// Address: 0x559890
//
int ctor_501()
{
  int v0; // edx
  __int64 v1; // r13
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  qword_500A440 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_500A44C = word_500A44C & 0x8000;
  unk_500A450 = 0;
  qword_500A488[1] = 0x100000000LL;
  unk_500A448 = v0;
  unk_500A458 = 0;
  unk_500A460 = 0;
  unk_500A468 = 0;
  unk_500A470 = 0;
  unk_500A478 = 0;
  unk_500A480 = 0;
  qword_500A488[0] = &qword_500A488[2];
  qword_500A488[3] = 0;
  qword_500A488[4] = &qword_500A488[7];
  qword_500A488[5] = 1;
  LODWORD(qword_500A488[6]) = 0;
  BYTE4(qword_500A488[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_500A488[1]);
  if ( (unsigned __int64)LODWORD(qword_500A488[1]) + 1 > HIDWORD(qword_500A488[1]) )
  {
    sub_C8D5F0(qword_500A488, &qword_500A488[2], LODWORD(qword_500A488[1]) + 1LL, 8);
    v2 = LODWORD(qword_500A488[1]);
  }
  *(_QWORD *)(qword_500A488[0] + 8 * v2) = v1;
  qword_500A488[9] = &unk_49D9748;
  qword_500A440 = &unk_49DC090;
  qword_500A488[11] = &unk_49DC1D0;
  ++LODWORD(qword_500A488[1]);
  qword_500A488[15] = nullsub_23;
  qword_500A488[8] = 0;
  qword_500A488[14] = sub_984030;
  qword_500A488[10] = 0;
  sub_C53080(&qword_500A440, "unroll-runtime-convergent", 25);
  LOWORD(qword_500A488[10]) = 257;
  LOBYTE(qword_500A488[8]) = 1;
  unk_500A470 = 57;
  LOBYTE(word_500A44C) = word_500A44C & 0x9F | 0x20;
  unk_500A468 = "Allow unrolling in presence of 'convergent' instructions.";
  sub_C53130(&qword_500A440);
  __cxa_atexit(sub_984900, &qword_500A440, &qword_4A427C0);
  qword_500A360 = (__int64)&unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500A36C &= 0x8000u;
  word_500A370 = 0;
  qword_500A3B0 = 0x100000000LL;
  qword_500A3A8 = (__int64)&unk_500A3B8;
  qword_500A378 = 0;
  qword_500A380 = 0;
  dword_500A368 = v3;
  qword_500A388 = 0;
  qword_500A390 = 0;
  qword_500A398 = 0;
  qword_500A3A0 = 0;
  qword_500A3C0 = 0;
  qword_500A3C8 = (__int64)&unk_500A3E0;
  qword_500A3D0 = 1;
  dword_500A3D8 = 0;
  byte_500A3DC = 1;
  v4 = sub_C57470();
  v5 = (unsigned int)qword_500A3B0;
  if ( (unsigned __int64)(unsigned int)qword_500A3B0 + 1 > HIDWORD(qword_500A3B0) )
  {
    v19 = v4;
    sub_C8D5F0((char *)&unk_500A3B8 - 16, &unk_500A3B8, (unsigned int)qword_500A3B0 + 1LL, 8);
    v5 = (unsigned int)qword_500A3B0;
    v4 = v19;
  }
  *(_QWORD *)(qword_500A3A8 + 8 * v5) = v4;
  qword_500A3F0 = (__int64)&unk_49D9748;
  qword_500A360 = (__int64)&unk_49DC090;
  qword_500A400 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A3B0) = qword_500A3B0 + 1;
  qword_500A420 = (__int64)nullsub_23;
  qword_500A3E8 = 0;
  qword_500A418 = (__int64)sub_984030;
  qword_500A3F8 = 0;
  sub_C53080(&qword_500A360, "unroll-runtime-epilog", 21);
  LOWORD(qword_500A3F8) = 256;
  LOBYTE(qword_500A3E8) = 0;
  qword_500A390 = 74;
  LOBYTE(dword_500A36C) = dword_500A36C & 0x9F | 0x20;
  qword_500A388 = (__int64)"Allow runtime unrolled loops to be unrolled with epilog instead of prolog.";
  sub_C53130(&qword_500A360);
  __cxa_atexit(sub_984900, &qword_500A360, &qword_4A427C0);
  qword_500A280 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500A2D0 = 0x100000000LL;
  dword_500A28C &= 0x8000u;
  qword_500A2C8 = (__int64)&unk_500A2D8;
  word_500A290 = 0;
  qword_500A298 = 0;
  dword_500A288 = v6;
  qword_500A2A0 = 0;
  qword_500A2A8 = 0;
  qword_500A2B0 = 0;
  qword_500A2B8 = 0;
  qword_500A2C0 = 0;
  qword_500A2E0 = 0;
  qword_500A2E8 = (__int64)&unk_500A300;
  qword_500A2F0 = 1;
  dword_500A2F8 = 0;
  byte_500A2FC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_500A2D0;
  if ( (unsigned __int64)(unsigned int)qword_500A2D0 + 1 > HIDWORD(qword_500A2D0) )
  {
    v20 = v7;
    sub_C8D5F0((char *)&unk_500A2D8 - 16, &unk_500A2D8, (unsigned int)qword_500A2D0 + 1LL, 8);
    v8 = (unsigned int)qword_500A2D0;
    v7 = v20;
  }
  *(_QWORD *)(qword_500A2C8 + 8 * v8) = v7;
  qword_500A310 = (__int64)&unk_49D9748;
  qword_500A280 = (__int64)&unk_49DC090;
  qword_500A320 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A2D0) = qword_500A2D0 + 1;
  qword_500A340 = (__int64)nullsub_23;
  qword_500A308 = 0;
  qword_500A338 = (__int64)sub_984030;
  qword_500A318 = 0;
  sub_C53080(&qword_500A280, "unroll-verify-domtree", 21);
  qword_500A2B0 = 30;
  LOWORD(qword_500A318) = 256;
  LOBYTE(qword_500A308) = 0;
  LOBYTE(dword_500A28C) = dword_500A28C & 0x9F | 0x20;
  qword_500A2A8 = (__int64)"Verify domtree after unrolling";
  sub_C53130(&qword_500A280);
  __cxa_atexit(sub_984900, &qword_500A280, &qword_4A427C0);
  qword_500A1A0 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500A1F0 = 0x100000000LL;
  dword_500A1AC &= 0x8000u;
  word_500A1B0 = 0;
  qword_500A1E8 = (__int64)&unk_500A1F8;
  qword_500A1B8 = 0;
  dword_500A1A8 = v9;
  qword_500A1C0 = 0;
  qword_500A1C8 = 0;
  qword_500A1D0 = 0;
  qword_500A1D8 = 0;
  qword_500A1E0 = 0;
  qword_500A200 = 0;
  qword_500A208 = (__int64)&unk_500A220;
  qword_500A210 = 1;
  dword_500A218 = 0;
  byte_500A21C = 1;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_500A1F0;
  if ( (unsigned __int64)(unsigned int)qword_500A1F0 + 1 > HIDWORD(qword_500A1F0) )
  {
    v21 = v10;
    sub_C8D5F0((char *)&unk_500A1F8 - 16, &unk_500A1F8, (unsigned int)qword_500A1F0 + 1LL, 8);
    v11 = (unsigned int)qword_500A1F0;
    v10 = v21;
  }
  *(_QWORD *)(qword_500A1E8 + 8 * v11) = v10;
  qword_500A230 = (__int64)&unk_49D9748;
  qword_500A1A0 = (__int64)&unk_49DC090;
  qword_500A240 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A1F0) = qword_500A1F0 + 1;
  qword_500A260 = (__int64)nullsub_23;
  qword_500A228 = 0;
  qword_500A258 = (__int64)sub_984030;
  qword_500A238 = 0;
  sub_C53080(&qword_500A1A0, "unroll-verify-loopinfo", 22);
  LOWORD(qword_500A238) = 256;
  LOBYTE(qword_500A228) = 0;
  qword_500A1D0 = 31;
  LOBYTE(dword_500A1AC) = dword_500A1AC & 0x9F | 0x20;
  qword_500A1C8 = (__int64)"Verify loopinfo after unrolling";
  sub_C53130(&qword_500A1A0);
  __cxa_atexit(sub_984900, &qword_500A1A0, &qword_4A427C0);
  qword_500A0C0 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500A13C = 1;
  word_500A0D0 = 0;
  qword_500A110 = 0x100000000LL;
  dword_500A0CC &= 0x8000u;
  qword_500A108 = (__int64)&unk_500A118;
  qword_500A0D8 = 0;
  dword_500A0C8 = v12;
  qword_500A0E0 = 0;
  qword_500A0E8 = 0;
  qword_500A0F0 = 0;
  qword_500A0F8 = 0;
  qword_500A100 = 0;
  qword_500A120 = 0;
  qword_500A128 = (__int64)&unk_500A140;
  qword_500A130 = 1;
  dword_500A138 = 0;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_500A110;
  if ( (unsigned __int64)(unsigned int)qword_500A110 + 1 > HIDWORD(qword_500A110) )
  {
    v22 = v13;
    sub_C8D5F0((char *)&unk_500A118 - 16, &unk_500A118, (unsigned int)qword_500A110 + 1LL, 8);
    v14 = (unsigned int)qword_500A110;
    v13 = v22;
  }
  *(_QWORD *)(qword_500A108 + 8 * v14) = v13;
  qword_500A150 = (__int64)&unk_49D9748;
  qword_500A0C0 = (__int64)&unk_49DC090;
  qword_500A160 = (__int64)&unk_49DC1D0;
  LODWORD(qword_500A110) = qword_500A110 + 1;
  qword_500A180 = (__int64)nullsub_23;
  qword_500A148 = 0;
  qword_500A178 = (__int64)sub_984030;
  qword_500A158 = 0;
  sub_C53080(&qword_500A0C0, "waterfall-unrolling-force-epilogue", 34);
  LOBYTE(qword_500A148) = 1;
  LOWORD(qword_500A158) = 257;
  qword_500A0F0 = 86;
  LOBYTE(dword_500A0CC) = dword_500A0CC & 0x9F | 0x20;
  qword_500A0E8 = (__int64)"Forces loops that are profitable to be waterfall unrolled to have epilogue remainders.";
  sub_C53130(&qword_500A0C0);
  __cxa_atexit(sub_984900, &qword_500A0C0, &qword_4A427C0);
  qword_5009FE0 = &unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_5009FEC = word_5009FEC & 0x8000;
  unk_5009FE8 = v15;
  qword_500A028[1] = 0x100000000LL;
  unk_5009FF0 = 0;
  unk_5009FF8 = 0;
  unk_500A000 = 0;
  unk_500A008 = 0;
  unk_500A010 = 0;
  unk_500A018 = 0;
  unk_500A020 = 0;
  qword_500A028[0] = &qword_500A028[2];
  qword_500A028[3] = 0;
  qword_500A028[4] = &qword_500A028[7];
  qword_500A028[5] = 1;
  LODWORD(qword_500A028[6]) = 0;
  BYTE4(qword_500A028[6]) = 1;
  v16 = sub_C57470();
  v17 = LODWORD(qword_500A028[1]);
  if ( (unsigned __int64)LODWORD(qword_500A028[1]) + 1 > HIDWORD(qword_500A028[1]) )
  {
    sub_C8D5F0(qword_500A028, &qword_500A028[2], LODWORD(qword_500A028[1]) + 1LL, 8);
    v17 = LODWORD(qword_500A028[1]);
  }
  *(_QWORD *)(qword_500A028[0] + 8 * v17) = v16;
  ++LODWORD(qword_500A028[1]);
  qword_500A028[8] = 0;
  qword_500A028[9] = &unk_49D9728;
  qword_500A028[10] = 0;
  qword_5009FE0 = &unk_49DBF10;
  qword_500A028[11] = &unk_49DC290;
  qword_500A028[15] = nullsub_24;
  qword_500A028[14] = sub_984050;
  sub_C53080(&qword_5009FE0, "waterfall-unrolling-num-loops", 29);
  LODWORD(qword_500A028[8]) = 2;
  BYTE4(qword_500A028[10]) = 1;
  LODWORD(qword_500A028[10]) = 2;
  unk_500A010 = 129;
  LOBYTE(word_5009FEC) = word_5009FEC & 0x9F | 0x20;
  unk_500A008 = "The number of loops generated in waterfall unrolling, excluding the final remainder loop handled by upst"
                "ream NVVM unrolling code.";
  sub_C53130(&qword_5009FE0);
  return __cxa_atexit(sub_984970, &qword_5009FE0, &qword_4A427C0);
}
