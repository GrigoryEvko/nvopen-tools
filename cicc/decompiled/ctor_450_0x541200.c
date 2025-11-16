// Function: ctor_450
// Address: 0x541200
//
int ctor_450()
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

  qword_4FFC440 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFC490 = 0x100000000LL;
  dword_4FFC44C &= 0x8000u;
  word_4FFC450 = 0;
  qword_4FFC458 = 0;
  qword_4FFC460 = 0;
  dword_4FFC448 = v0;
  qword_4FFC468 = 0;
  qword_4FFC470 = 0;
  qword_4FFC478 = 0;
  qword_4FFC480 = 0;
  qword_4FFC488 = (__int64)&unk_4FFC498;
  qword_4FFC4A0 = 0;
  qword_4FFC4A8 = (__int64)&unk_4FFC4C0;
  qword_4FFC4B0 = 1;
  dword_4FFC4B8 = 0;
  byte_4FFC4BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFC490;
  v3 = (unsigned int)qword_4FFC490 + 1LL;
  if ( v3 > HIDWORD(qword_4FFC490) )
  {
    sub_C8D5F0((char *)&unk_4FFC498 - 16, &unk_4FFC498, v3, 8);
    v2 = (unsigned int)qword_4FFC490;
  }
  *(_QWORD *)(qword_4FFC488 + 8 * v2) = v1;
  qword_4FFC4D0 = (__int64)&unk_49DA090;
  qword_4FFC440 = (__int64)&unk_49DBF90;
  qword_4FFC4E0 = (__int64)&unk_49DC230;
  LODWORD(qword_4FFC490) = qword_4FFC490 + 1;
  qword_4FFC500 = (__int64)nullsub_58;
  qword_4FFC4C8 = 0;
  qword_4FFC4F8 = (__int64)sub_B2B5F0;
  qword_4FFC4D8 = 0;
  sub_C53080(&qword_4FFC440, "gvn-max-hoisted", 15);
  LODWORD(qword_4FFC4C8) = -1;
  BYTE4(qword_4FFC4D8) = 1;
  LODWORD(qword_4FFC4D8) = -1;
  qword_4FFC470 = 60;
  LOBYTE(dword_4FFC44C) = dword_4FFC44C & 0x9F | 0x20;
  qword_4FFC468 = (__int64)"Max number of instructions to hoist (default unlimited = -1)";
  sub_C53130(&qword_4FFC440);
  __cxa_atexit(sub_B2B680, &qword_4FFC440, &qword_4A427C0);
  qword_4FFC360 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFC3B0 = 0x100000000LL;
  dword_4FFC36C &= 0x8000u;
  word_4FFC370 = 0;
  qword_4FFC3A8 = (__int64)&unk_4FFC3B8;
  qword_4FFC378 = 0;
  dword_4FFC368 = v4;
  qword_4FFC380 = 0;
  qword_4FFC388 = 0;
  qword_4FFC390 = 0;
  qword_4FFC398 = 0;
  qword_4FFC3A0 = 0;
  qword_4FFC3C0 = 0;
  qword_4FFC3C8 = (__int64)&unk_4FFC3E0;
  qword_4FFC3D0 = 1;
  dword_4FFC3D8 = 0;
  byte_4FFC3DC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFC3B0;
  if ( (unsigned __int64)(unsigned int)qword_4FFC3B0 + 1 > HIDWORD(qword_4FFC3B0) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_4FFC3B8 - 16, &unk_4FFC3B8, (unsigned int)qword_4FFC3B0 + 1LL, 8);
    v6 = (unsigned int)qword_4FFC3B0;
    v5 = v15;
  }
  *(_QWORD *)(qword_4FFC3A8 + 8 * v6) = v5;
  qword_4FFC3F0 = (__int64)&unk_49DA090;
  qword_4FFC360 = (__int64)&unk_49DBF90;
  qword_4FFC400 = (__int64)&unk_49DC230;
  LODWORD(qword_4FFC3B0) = qword_4FFC3B0 + 1;
  qword_4FFC420 = (__int64)nullsub_58;
  qword_4FFC3E8 = 0;
  qword_4FFC418 = (__int64)sub_B2B5F0;
  qword_4FFC3F8 = 0;
  sub_C53080(&qword_4FFC360, "gvn-hoist-max-bbs", 17);
  LODWORD(qword_4FFC3E8) = 4;
  BYTE4(qword_4FFC3F8) = 1;
  LODWORD(qword_4FFC3F8) = 4;
  qword_4FFC390 = 95;
  LOBYTE(dword_4FFC36C) = dword_4FFC36C & 0x9F | 0x20;
  qword_4FFC388 = (__int64)"Max number of basic blocks on the path between hoisting locations (default = 4, unlimited = -1)";
  sub_C53130(&qword_4FFC360);
  __cxa_atexit(sub_B2B680, &qword_4FFC360, &qword_4A427C0);
  qword_4FFC280 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFC2FC = 1;
  qword_4FFC2D0 = 0x100000000LL;
  dword_4FFC28C &= 0x8000u;
  qword_4FFC2C8 = (__int64)&unk_4FFC2D8;
  qword_4FFC298 = 0;
  qword_4FFC2A0 = 0;
  dword_4FFC288 = v7;
  word_4FFC290 = 0;
  qword_4FFC2A8 = 0;
  qword_4FFC2B0 = 0;
  qword_4FFC2B8 = 0;
  qword_4FFC2C0 = 0;
  qword_4FFC2E0 = 0;
  qword_4FFC2E8 = (__int64)&unk_4FFC300;
  qword_4FFC2F0 = 1;
  dword_4FFC2F8 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FFC2D0;
  if ( (unsigned __int64)(unsigned int)qword_4FFC2D0 + 1 > HIDWORD(qword_4FFC2D0) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_4FFC2D8 - 16, &unk_4FFC2D8, (unsigned int)qword_4FFC2D0 + 1LL, 8);
    v9 = (unsigned int)qword_4FFC2D0;
    v8 = v16;
  }
  *(_QWORD *)(qword_4FFC2C8 + 8 * v9) = v8;
  qword_4FFC310 = (__int64)&unk_49DA090;
  qword_4FFC280 = (__int64)&unk_49DBF90;
  qword_4FFC320 = (__int64)&unk_49DC230;
  LODWORD(qword_4FFC2D0) = qword_4FFC2D0 + 1;
  qword_4FFC340 = (__int64)nullsub_58;
  qword_4FFC308 = 0;
  qword_4FFC338 = (__int64)sub_B2B5F0;
  qword_4FFC318 = 0;
  sub_C53080(&qword_4FFC280, "gvn-hoist-max-depth", 19);
  LODWORD(qword_4FFC308) = 100;
  BYTE4(qword_4FFC318) = 1;
  LODWORD(qword_4FFC318) = 100;
  qword_4FFC2B0 = 113;
  LOBYTE(dword_4FFC28C) = dword_4FFC28C & 0x9F | 0x20;
  qword_4FFC2A8 = (__int64)"Hoist instructions from the beginning of the BB up to the maximum specified depth (default = "
                           "100, unlimited = -1)";
  sub_C53130(&qword_4FFC280);
  __cxa_atexit(sub_B2B680, &qword_4FFC280, &qword_4A427C0);
  qword_4FFC1A0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFC1AC &= 0x8000u;
  word_4FFC1B0 = 0;
  qword_4FFC1F0 = 0x100000000LL;
  qword_4FFC1E8 = (__int64)&unk_4FFC1F8;
  qword_4FFC1B8 = 0;
  qword_4FFC1C0 = 0;
  dword_4FFC1A8 = v10;
  qword_4FFC1C8 = 0;
  qword_4FFC1D0 = 0;
  qword_4FFC1D8 = 0;
  qword_4FFC1E0 = 0;
  qword_4FFC200 = 0;
  qword_4FFC208 = (__int64)&unk_4FFC220;
  qword_4FFC210 = 1;
  dword_4FFC218 = 0;
  byte_4FFC21C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FFC1F0;
  v13 = (unsigned int)qword_4FFC1F0 + 1LL;
  if ( v13 > HIDWORD(qword_4FFC1F0) )
  {
    sub_C8D5F0((char *)&unk_4FFC1F8 - 16, &unk_4FFC1F8, v13, 8);
    v12 = (unsigned int)qword_4FFC1F0;
  }
  *(_QWORD *)(qword_4FFC1E8 + 8 * v12) = v11;
  qword_4FFC230 = (__int64)&unk_49DA090;
  qword_4FFC1A0 = (__int64)&unk_49DBF90;
  qword_4FFC240 = (__int64)&unk_49DC230;
  LODWORD(qword_4FFC1F0) = qword_4FFC1F0 + 1;
  qword_4FFC260 = (__int64)nullsub_58;
  qword_4FFC228 = 0;
  qword_4FFC258 = (__int64)sub_B2B5F0;
  qword_4FFC238 = 0;
  sub_C53080(&qword_4FFC1A0, "gvn-hoist-max-chain-length", 26);
  LODWORD(qword_4FFC228) = 10;
  BYTE4(qword_4FFC238) = 1;
  LODWORD(qword_4FFC238) = 10;
  qword_4FFC1D0 = 74;
  LOBYTE(dword_4FFC1AC) = dword_4FFC1AC & 0x9F | 0x20;
  qword_4FFC1C8 = (__int64)"Maximum length of dependent chains to hoist (default = 10, unlimited = -1)";
  sub_C53130(&qword_4FFC1A0);
  return __cxa_atexit(sub_B2B680, &qword_4FFC1A0, &qword_4A427C0);
}
