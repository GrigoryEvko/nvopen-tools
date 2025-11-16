// Function: ctor_607
// Address: 0x584b60
//
int __fastcall ctor_607(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // edx
  __int64 v41; // rbx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v45; // [rsp+8h] [rbp-38h]
  __int64 v46; // [rsp+8h] [rbp-38h]
  __int64 v47; // [rsp+8h] [rbp-38h]
  __int64 v48; // [rsp+8h] [rbp-38h]
  __int64 v49; // [rsp+8h] [rbp-38h]
  __int64 v50; // [rsp+8h] [rbp-38h]

  qword_502B500 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_502B50C &= 0x8000u;
  word_502B510 = 0;
  qword_502B550 = 0x100000000LL;
  qword_502B518 = 0;
  qword_502B520 = 0;
  qword_502B528 = 0;
  dword_502B508 = v4;
  qword_502B530 = 0;
  qword_502B538 = 0;
  qword_502B540 = 0;
  qword_502B548 = (__int64)&unk_502B558;
  qword_502B560 = 0;
  qword_502B568 = (__int64)&unk_502B580;
  qword_502B570 = 1;
  dword_502B578 = 0;
  byte_502B57C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502B550;
  v7 = (unsigned int)qword_502B550 + 1LL;
  if ( v7 > HIDWORD(qword_502B550) )
  {
    sub_C8D5F0((char *)&unk_502B558 - 16, &unk_502B558, v7, 8);
    v6 = (unsigned int)qword_502B550;
  }
  *(_QWORD *)(qword_502B548 + 8 * v6) = v5;
  qword_502B590 = (__int64)&unk_49D9748;
  qword_502B500 = (__int64)&unk_49DC090;
  qword_502B5A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502B550) = qword_502B550 + 1;
  qword_502B5C0 = (__int64)nullsub_23;
  qword_502B588 = 0;
  qword_502B5B8 = (__int64)sub_984030;
  qword_502B598 = 0;
  sub_C53080(&qword_502B500, "nvptx-sched4reg", 15);
  qword_502B530 = 45;
  qword_502B528 = (__int64)"NVPTX Specific: schedule for register pressue";
  LOWORD(qword_502B598) = 256;
  LOBYTE(qword_502B588) = 0;
  sub_C53130(&qword_502B500);
  __cxa_atexit(sub_984900, &qword_502B500, &qword_4A427C0);
  qword_502B420 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502B500, v8, v9), 1u);
  dword_502B42C &= 0x8000u;
  word_502B430 = 0;
  qword_502B470 = 0x100000000LL;
  qword_502B468 = (__int64)&unk_502B478;
  qword_502B438 = 0;
  qword_502B440 = 0;
  dword_502B428 = v10;
  qword_502B448 = 0;
  qword_502B450 = 0;
  qword_502B458 = 0;
  qword_502B460 = 0;
  qword_502B480 = 0;
  qword_502B488 = (__int64)&unk_502B4A0;
  qword_502B490 = 1;
  dword_502B498 = 0;
  byte_502B49C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502B470;
  if ( (unsigned __int64)(unsigned int)qword_502B470 + 1 > HIDWORD(qword_502B470) )
  {
    v45 = v11;
    sub_C8D5F0((char *)&unk_502B478 - 16, &unk_502B478, (unsigned int)qword_502B470 + 1LL, 8);
    v12 = (unsigned int)qword_502B470;
    v11 = v45;
  }
  *(_QWORD *)(qword_502B468 + 8 * v12) = v11;
  LODWORD(qword_502B470) = qword_502B470 + 1;
  qword_502B4A8 = 0;
  qword_502B4B0 = (__int64)&unk_49D9728;
  qword_502B4B8 = 0;
  qword_502B420 = (__int64)&unk_49DBF10;
  qword_502B4C0 = (__int64)&unk_49DC290;
  qword_502B4E0 = (__int64)nullsub_24;
  qword_502B4D8 = (__int64)sub_984050;
  sub_C53080(&qword_502B420, "nvptx-fma-level", 15);
  qword_502B450 = 79;
  LODWORD(qword_502B4A8) = 2;
  BYTE4(qword_502B4B8) = 1;
  LODWORD(qword_502B4B8) = 2;
  LOBYTE(dword_502B42C) = dword_502B42C & 0x9F | 0x20;
  qword_502B448 = (__int64)"NVPTX Specific: FMA contraction (0: don't do it 1: do it  2: do it aggressively";
  sub_C53130(&qword_502B420);
  __cxa_atexit(sub_984970, &qword_502B420, &qword_4A427C0);
  qword_502B340 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_502B420, v13, v14), 1u);
  dword_502B34C &= 0x8000u;
  word_502B350 = 0;
  qword_502B390 = 0x100000000LL;
  qword_502B388 = (__int64)&unk_502B398;
  qword_502B358 = 0;
  qword_502B360 = 0;
  dword_502B348 = v15;
  qword_502B368 = 0;
  qword_502B370 = 0;
  qword_502B378 = 0;
  qword_502B380 = 0;
  qword_502B3A0 = 0;
  qword_502B3A8 = (__int64)&unk_502B3C0;
  qword_502B3B0 = 1;
  dword_502B3B8 = 0;
  byte_502B3BC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_502B390;
  if ( (unsigned __int64)(unsigned int)qword_502B390 + 1 > HIDWORD(qword_502B390) )
  {
    v46 = v16;
    sub_C8D5F0((char *)&unk_502B398 - 16, &unk_502B398, (unsigned int)qword_502B390 + 1LL, 8);
    v17 = (unsigned int)qword_502B390;
    v16 = v46;
  }
  *(_QWORD *)(qword_502B388 + 8 * v17) = v16;
  LODWORD(qword_502B390) = qword_502B390 + 1;
  qword_502B3C8 = 0;
  qword_502B3D0 = (__int64)&unk_49DA090;
  qword_502B3D8 = 0;
  qword_502B340 = (__int64)&unk_49DBF90;
  qword_502B3E0 = (__int64)&unk_49DC230;
  qword_502B400 = (__int64)nullsub_58;
  qword_502B3F8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502B340, "nvptx-prec-divf32", 17);
  qword_502B370 = 147;
  LODWORD(qword_502B3C8) = 2;
  BYTE4(qword_502B3D8) = 1;
  LODWORD(qword_502B3D8) = 2;
  LOBYTE(dword_502B34C) = dword_502B34C & 0x9F | 0x20;
  qword_502B368 = (__int64)"NVPTX Specifies: 0 use div.approx, 1 use div.full, 2 use IEEE Compliant F32 div.rnd with ftz "
                           "allowed, 3 use IEEE Compliant F32 div.rnd with no ftz.";
  sub_C53130(&qword_502B340);
  __cxa_atexit(sub_B2B680, &qword_502B340, &qword_4A427C0);
  qword_502B260 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502B340, v18, v19), 1u);
  dword_502B26C &= 0x8000u;
  word_502B270 = 0;
  qword_502B2B0 = 0x100000000LL;
  qword_502B2A8 = (__int64)&unk_502B2B8;
  qword_502B278 = 0;
  qword_502B280 = 0;
  dword_502B268 = v20;
  qword_502B288 = 0;
  qword_502B290 = 0;
  qword_502B298 = 0;
  qword_502B2A0 = 0;
  qword_502B2C0 = 0;
  qword_502B2C8 = (__int64)&unk_502B2E0;
  qword_502B2D0 = 1;
  dword_502B2D8 = 0;
  byte_502B2DC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_502B2B0;
  if ( (unsigned __int64)(unsigned int)qword_502B2B0 + 1 > HIDWORD(qword_502B2B0) )
  {
    v47 = v21;
    sub_C8D5F0((char *)&unk_502B2B8 - 16, &unk_502B2B8, (unsigned int)qword_502B2B0 + 1LL, 8);
    v22 = (unsigned int)qword_502B2B0;
    v21 = v47;
  }
  *(_QWORD *)(qword_502B2A8 + 8 * v22) = v21;
  qword_502B2F0 = (__int64)&unk_49D9748;
  qword_502B260 = (__int64)&unk_49DC090;
  qword_502B300 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502B2B0) = qword_502B2B0 + 1;
  qword_502B320 = (__int64)nullsub_23;
  qword_502B2E8 = 0;
  qword_502B318 = (__int64)sub_984030;
  qword_502B2F8 = 0;
  sub_C53080(&qword_502B260, "nvptx-prec-sqrtf32", 18);
  LOWORD(qword_502B2F8) = 257;
  LOBYTE(qword_502B2E8) = 1;
  qword_502B290 = 49;
  LOBYTE(dword_502B26C) = dword_502B26C & 0x9F | 0x20;
  qword_502B288 = (__int64)"NVPTX Specific: 0 use sqrt.approx, 1 use sqrt.rn.";
  sub_C53130(&qword_502B260);
  __cxa_atexit(sub_984900, &qword_502B260, &qword_4A427C0);
  qword_502B180 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502B260, v23, v24), 1u);
  qword_502B1D0 = 0x100000000LL;
  dword_502B18C &= 0x8000u;
  qword_502B1C8 = (__int64)&unk_502B1D8;
  word_502B190 = 0;
  qword_502B198 = 0;
  dword_502B188 = v25;
  qword_502B1A0 = 0;
  qword_502B1A8 = 0;
  qword_502B1B0 = 0;
  qword_502B1B8 = 0;
  qword_502B1C0 = 0;
  qword_502B1E0 = 0;
  qword_502B1E8 = (__int64)&unk_502B200;
  qword_502B1F0 = 1;
  dword_502B1F8 = 0;
  byte_502B1FC = 1;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_502B1D0;
  if ( (unsigned __int64)(unsigned int)qword_502B1D0 + 1 > HIDWORD(qword_502B1D0) )
  {
    v48 = v26;
    sub_C8D5F0((char *)&unk_502B1D8 - 16, &unk_502B1D8, (unsigned int)qword_502B1D0 + 1LL, 8);
    v27 = (unsigned int)qword_502B1D0;
    v26 = v48;
  }
  *(_QWORD *)(qword_502B1C8 + 8 * v27) = v26;
  qword_502B210 = (__int64)&unk_49D9748;
  qword_502B180 = (__int64)&unk_49DC090;
  qword_502B220 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502B1D0) = qword_502B1D0 + 1;
  qword_502B240 = (__int64)nullsub_23;
  qword_502B208 = 0;
  qword_502B238 = (__int64)sub_984030;
  qword_502B218 = 0;
  sub_C53080(&qword_502B180, "enable-bfi64", 12);
  LOWORD(qword_502B218) = 257;
  LOBYTE(qword_502B208) = 1;
  qword_502B1B0 = 44;
  LOBYTE(dword_502B18C) = dword_502B18C & 0x9F | 0x20;
  qword_502B1A8 = (__int64)"Enable generation of 64-bit BFI instructions";
  sub_C53130(&qword_502B180);
  __cxa_atexit(sub_984900, &qword_502B180, &qword_4A427C0);
  qword_502B0A0 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502B180, v28, v29), 1u);
  qword_502B0F0 = 0x100000000LL;
  dword_502B0AC &= 0x8000u;
  qword_502B0E8 = (__int64)&unk_502B0F8;
  word_502B0B0 = 0;
  qword_502B0B8 = 0;
  dword_502B0A8 = v30;
  qword_502B0C0 = 0;
  qword_502B0C8 = 0;
  qword_502B0D0 = 0;
  qword_502B0D8 = 0;
  qword_502B0E0 = 0;
  qword_502B100 = 0;
  qword_502B108 = (__int64)&unk_502B120;
  qword_502B110 = 1;
  dword_502B118 = 0;
  byte_502B11C = 1;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_502B0F0;
  if ( (unsigned __int64)(unsigned int)qword_502B0F0 + 1 > HIDWORD(qword_502B0F0) )
  {
    v49 = v31;
    sub_C8D5F0((char *)&unk_502B0F8 - 16, &unk_502B0F8, (unsigned int)qword_502B0F0 + 1LL, 8);
    v32 = (unsigned int)qword_502B0F0;
    v31 = v49;
  }
  *(_QWORD *)(qword_502B0E8 + 8 * v32) = v31;
  qword_502B130 = (__int64)&unk_49D9748;
  qword_502B0A0 = (__int64)&unk_49DC090;
  qword_502B140 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502B0F0) = qword_502B0F0 + 1;
  qword_502B160 = (__int64)nullsub_23;
  qword_502B128 = 0;
  qword_502B158 = (__int64)sub_984030;
  qword_502B138 = 0;
  sub_C53080(&qword_502B0A0, "nvptx-normalize-select", 22);
  LOWORD(qword_502B138) = 256;
  LOBYTE(qword_502B128) = 0;
  qword_502B0D0 = 77;
  LOBYTE(dword_502B0AC) = dword_502B0AC & 0x9F | 0x20;
  qword_502B0C8 = (__int64)"NVPTX Specific: override TLI::shouldNormalizeToSelectSequence to return false";
  sub_C53130(&qword_502B0A0);
  __cxa_atexit(sub_984900, &qword_502B0A0, &qword_4A427C0);
  qword_502AFC0 = (__int64)&unk_49DC150;
  v35 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502B0A0, v33, v34), 1u);
  qword_502B010 = 0x100000000LL;
  dword_502AFCC &= 0x8000u;
  qword_502B008 = (__int64)&unk_502B018;
  word_502AFD0 = 0;
  qword_502AFD8 = 0;
  dword_502AFC8 = v35;
  qword_502AFE0 = 0;
  qword_502AFE8 = 0;
  qword_502AFF0 = 0;
  qword_502AFF8 = 0;
  qword_502B000 = 0;
  qword_502B020 = 0;
  qword_502B028 = (__int64)&unk_502B040;
  qword_502B030 = 1;
  dword_502B038 = 0;
  byte_502B03C = 1;
  v36 = sub_C57470();
  v37 = (unsigned int)qword_502B010;
  if ( (unsigned __int64)(unsigned int)qword_502B010 + 1 > HIDWORD(qword_502B010) )
  {
    v50 = v36;
    sub_C8D5F0((char *)&unk_502B018 - 16, &unk_502B018, (unsigned int)qword_502B010 + 1LL, 8);
    v37 = (unsigned int)qword_502B010;
    v36 = v50;
  }
  *(_QWORD *)(qword_502B008 + 8 * v37) = v36;
  qword_502B050 = (__int64)&unk_49D9748;
  qword_502AFC0 = (__int64)&unk_49DC090;
  qword_502B060 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502B010) = qword_502B010 + 1;
  qword_502B080 = (__int64)nullsub_23;
  qword_502B048 = 0;
  qword_502B078 = (__int64)sub_984030;
  qword_502B058 = 0;
  sub_C53080(&qword_502AFC0, "nvptx-approx-log2f32", 20);
  qword_502AFE8 = (__int64)"NVPTX Specific: whether to use lg2.approx for log2";
  LOWORD(qword_502B058) = 256;
  qword_502AFF0 = 50;
  LOBYTE(qword_502B048) = 0;
  sub_C53130(&qword_502AFC0);
  __cxa_atexit(sub_984900, &qword_502AFC0, &qword_4A427C0);
  qword_502AEE0 = (__int64)&unk_49DC150;
  v40 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502AFC0, v38, v39), 1u);
  qword_502AF30 = 0x100000000LL;
  dword_502AEEC &= 0x8000u;
  word_502AEF0 = 0;
  qword_502AF28 = (__int64)&unk_502AF38;
  qword_502AEF8 = 0;
  dword_502AEE8 = v40;
  qword_502AF00 = 0;
  qword_502AF08 = 0;
  qword_502AF10 = 0;
  qword_502AF18 = 0;
  qword_502AF20 = 0;
  qword_502AF40 = 0;
  qword_502AF48 = (__int64)&unk_502AF60;
  qword_502AF50 = 1;
  dword_502AF58 = 0;
  byte_502AF5C = 1;
  v41 = sub_C57470();
  v42 = (unsigned int)qword_502AF30;
  v43 = (unsigned int)qword_502AF30 + 1LL;
  if ( v43 > HIDWORD(qword_502AF30) )
  {
    sub_C8D5F0((char *)&unk_502AF38 - 16, &unk_502AF38, v43, 8);
    v42 = (unsigned int)qword_502AF30;
  }
  *(_QWORD *)(qword_502AF28 + 8 * v42) = v41;
  qword_502AF70 = (__int64)&unk_49D9748;
  qword_502AEE0 = (__int64)&unk_49DC090;
  qword_502AF80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_502AF30) = qword_502AF30 + 1;
  qword_502AFA0 = (__int64)nullsub_23;
  qword_502AF68 = 0;
  qword_502AF98 = (__int64)sub_984030;
  qword_502AF78 = 0;
  sub_C53080(&qword_502AEE0, "nvptx-force-min-byval-param-align", 33);
  qword_502AF10 = 84;
  LOBYTE(qword_502AF68) = 0;
  LOBYTE(dword_502AEEC) = dword_502AEEC & 0x9F | 0x20;
  qword_502AF08 = (__int64)"NVPTX Specific: force 4-byte minimal alignment for byval params of device functions.";
  LOWORD(qword_502AF78) = 256;
  sub_C53130(&qword_502AEE0);
  return __cxa_atexit(sub_984900, &qword_502AEE0, &qword_4A427C0);
}
