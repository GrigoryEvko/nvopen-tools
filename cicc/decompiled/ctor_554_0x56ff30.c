// Function: ctor_554
// Address: 0x56ff30
//
int ctor_554()
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

  qword_501E680 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501E6D0 = 0x100000000LL;
  word_501E690 = 0;
  dword_501E68C &= 0x8000u;
  qword_501E698 = 0;
  qword_501E6A0 = 0;
  dword_501E688 = v0;
  qword_501E6A8 = 0;
  qword_501E6B0 = 0;
  qword_501E6B8 = 0;
  qword_501E6C0 = 0;
  qword_501E6C8 = (__int64)&unk_501E6D8;
  qword_501E6E0 = 0;
  qword_501E6E8 = (__int64)&unk_501E700;
  qword_501E6F0 = 1;
  dword_501E6F8 = 0;
  byte_501E6FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501E6D0;
  v3 = (unsigned int)qword_501E6D0 + 1LL;
  if ( v3 > HIDWORD(qword_501E6D0) )
  {
    sub_C8D5F0((char *)&unk_501E6D8 - 16, &unk_501E6D8, v3, 8);
    v2 = (unsigned int)qword_501E6D0;
  }
  *(_QWORD *)(qword_501E6C8 + 8 * v2) = v1;
  qword_501E710 = (__int64)&unk_49D9748;
  qword_501E680 = (__int64)&unk_49DC090;
  qword_501E720 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501E6D0) = qword_501E6D0 + 1;
  qword_501E740 = (__int64)nullsub_23;
  qword_501E708 = 0;
  qword_501E738 = (__int64)sub_984030;
  qword_501E718 = 0;
  sub_C53080(&qword_501E680, "force-hardware-loops", 20);
  LOWORD(qword_501E718) = 256;
  LOBYTE(qword_501E708) = 0;
  qword_501E6B0 = 46;
  LOBYTE(dword_501E68C) = dword_501E68C & 0x9F | 0x20;
  qword_501E6A8 = (__int64)"Force hardware loops intrinsics to be inserted";
  sub_C53130(&qword_501E680);
  __cxa_atexit(sub_984900, &qword_501E680, &qword_4A427C0);
  qword_501E5A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501E5F0 = 0x100000000LL;
  dword_501E5AC &= 0x8000u;
  qword_501E5E8 = (__int64)&unk_501E5F8;
  word_501E5B0 = 0;
  qword_501E5B8 = 0;
  dword_501E5A8 = v4;
  qword_501E5C0 = 0;
  qword_501E5C8 = 0;
  qword_501E5D0 = 0;
  qword_501E5D8 = 0;
  qword_501E5E0 = 0;
  qword_501E600 = 0;
  qword_501E608 = (__int64)&unk_501E620;
  qword_501E610 = 1;
  dword_501E618 = 0;
  byte_501E61C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_501E5F0;
  if ( (unsigned __int64)(unsigned int)qword_501E5F0 + 1 > HIDWORD(qword_501E5F0) )
  {
    v21 = v5;
    sub_C8D5F0((char *)&unk_501E5F8 - 16, &unk_501E5F8, (unsigned int)qword_501E5F0 + 1LL, 8);
    v6 = (unsigned int)qword_501E5F0;
    v5 = v21;
  }
  *(_QWORD *)(qword_501E5E8 + 8 * v6) = v5;
  qword_501E630 = (__int64)&unk_49D9748;
  qword_501E5A0 = (__int64)&unk_49DC090;
  qword_501E640 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501E5F0) = qword_501E5F0 + 1;
  qword_501E660 = (__int64)nullsub_23;
  qword_501E628 = 0;
  qword_501E658 = (__int64)sub_984030;
  qword_501E638 = 0;
  sub_C53080(&qword_501E5A0, "force-hardware-loop-phi", 23);
  LOWORD(qword_501E638) = 256;
  LOBYTE(qword_501E628) = 0;
  qword_501E5D0 = 55;
  LOBYTE(dword_501E5AC) = dword_501E5AC & 0x9F | 0x20;
  qword_501E5C8 = (__int64)"Force hardware loop counter to be updated through a phi";
  sub_C53130(&qword_501E5A0);
  __cxa_atexit(sub_984900, &qword_501E5A0, &qword_4A427C0);
  qword_501E4C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501E510 = 0x100000000LL;
  dword_501E4CC &= 0x8000u;
  qword_501E508 = (__int64)&unk_501E518;
  word_501E4D0 = 0;
  qword_501E4D8 = 0;
  dword_501E4C8 = v7;
  qword_501E4E0 = 0;
  qword_501E4E8 = 0;
  qword_501E4F0 = 0;
  qword_501E4F8 = 0;
  qword_501E500 = 0;
  qword_501E520 = 0;
  qword_501E528 = (__int64)&unk_501E540;
  qword_501E530 = 1;
  dword_501E538 = 0;
  byte_501E53C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_501E510;
  if ( (unsigned __int64)(unsigned int)qword_501E510 + 1 > HIDWORD(qword_501E510) )
  {
    v22 = v8;
    sub_C8D5F0((char *)&unk_501E518 - 16, &unk_501E518, (unsigned int)qword_501E510 + 1LL, 8);
    v9 = (unsigned int)qword_501E510;
    v8 = v22;
  }
  *(_QWORD *)(qword_501E508 + 8 * v9) = v8;
  qword_501E550 = (__int64)&unk_49D9748;
  qword_501E4C0 = (__int64)&unk_49DC090;
  qword_501E560 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501E510) = qword_501E510 + 1;
  qword_501E580 = (__int64)nullsub_23;
  qword_501E548 = 0;
  qword_501E578 = (__int64)sub_984030;
  qword_501E558 = 0;
  sub_C53080(&qword_501E4C0, "force-nested-hardware-loop", 26);
  LOWORD(qword_501E558) = 256;
  LOBYTE(qword_501E548) = 0;
  qword_501E4F0 = 40;
  LOBYTE(dword_501E4CC) = dword_501E4CC & 0x9F | 0x20;
  qword_501E4E8 = (__int64)"Force allowance of nested hardware loops";
  sub_C53130(&qword_501E4C0);
  __cxa_atexit(sub_984900, &qword_501E4C0, &qword_4A427C0);
  qword_501E3E0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501E430 = 0x100000000LL;
  dword_501E3EC &= 0x8000u;
  qword_501E428 = (__int64)&unk_501E438;
  word_501E3F0 = 0;
  qword_501E3F8 = 0;
  dword_501E3E8 = v10;
  qword_501E400 = 0;
  qword_501E408 = 0;
  qword_501E410 = 0;
  qword_501E418 = 0;
  qword_501E420 = 0;
  qword_501E440 = 0;
  qword_501E448 = (__int64)&unk_501E460;
  qword_501E450 = 1;
  dword_501E458 = 0;
  byte_501E45C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_501E430;
  if ( (unsigned __int64)(unsigned int)qword_501E430 + 1 > HIDWORD(qword_501E430) )
  {
    v23 = v11;
    sub_C8D5F0((char *)&unk_501E438 - 16, &unk_501E438, (unsigned int)qword_501E430 + 1LL, 8);
    v12 = (unsigned int)qword_501E430;
    v11 = v23;
  }
  *(_QWORD *)(qword_501E428 + 8 * v12) = v11;
  LODWORD(qword_501E430) = qword_501E430 + 1;
  qword_501E468 = 0;
  qword_501E470 = (__int64)&unk_49D9728;
  qword_501E478 = 0;
  qword_501E3E0 = (__int64)&unk_49DBF10;
  qword_501E480 = (__int64)&unk_49DC290;
  qword_501E4A0 = (__int64)nullsub_24;
  qword_501E498 = (__int64)sub_984050;
  sub_C53080(&qword_501E3E0, "hardware-loop-decrement", 23);
  LODWORD(qword_501E468) = 1;
  BYTE4(qword_501E478) = 1;
  LODWORD(qword_501E478) = 1;
  qword_501E410 = 28;
  LOBYTE(dword_501E3EC) = dword_501E3EC & 0x9F | 0x20;
  qword_501E408 = (__int64)"Set the loop decrement value";
  sub_C53130(&qword_501E3E0);
  __cxa_atexit(sub_984970, &qword_501E3E0, &qword_4A427C0);
  qword_501E300 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501E350 = 0x100000000LL;
  dword_501E30C &= 0x8000u;
  word_501E310 = 0;
  qword_501E348 = (__int64)&unk_501E358;
  qword_501E318 = 0;
  dword_501E308 = v13;
  qword_501E320 = 0;
  qword_501E328 = 0;
  qword_501E330 = 0;
  qword_501E338 = 0;
  qword_501E340 = 0;
  qword_501E360 = 0;
  qword_501E368 = (__int64)&unk_501E380;
  qword_501E370 = 1;
  dword_501E378 = 0;
  byte_501E37C = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_501E350;
  if ( (unsigned __int64)(unsigned int)qword_501E350 + 1 > HIDWORD(qword_501E350) )
  {
    v24 = v14;
    sub_C8D5F0((char *)&unk_501E358 - 16, &unk_501E358, (unsigned int)qword_501E350 + 1LL, 8);
    v15 = (unsigned int)qword_501E350;
    v14 = v24;
  }
  *(_QWORD *)(qword_501E348 + 8 * v15) = v14;
  LODWORD(qword_501E350) = qword_501E350 + 1;
  qword_501E388 = 0;
  qword_501E390 = (__int64)&unk_49D9728;
  qword_501E398 = 0;
  qword_501E300 = (__int64)&unk_49DBF10;
  qword_501E3A0 = (__int64)&unk_49DC290;
  qword_501E3C0 = (__int64)nullsub_24;
  qword_501E3B8 = (__int64)sub_984050;
  sub_C53080(&qword_501E300, "hardware-loop-counter-bitwidth", 30);
  LODWORD(qword_501E388) = 32;
  BYTE4(qword_501E398) = 1;
  LODWORD(qword_501E398) = 32;
  qword_501E330 = 29;
  LOBYTE(dword_501E30C) = dword_501E30C & 0x9F | 0x20;
  qword_501E328 = (__int64)"Set the loop counter bitwidth";
  sub_C53130(&qword_501E300);
  __cxa_atexit(sub_984970, &qword_501E300, &qword_4A427C0);
  qword_501E220 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501E29C = 1;
  qword_501E270 = 0x100000000LL;
  dword_501E22C &= 0x8000u;
  qword_501E268 = (__int64)&unk_501E278;
  qword_501E238 = 0;
  qword_501E240 = 0;
  dword_501E228 = v16;
  word_501E230 = 0;
  qword_501E248 = 0;
  qword_501E250 = 0;
  qword_501E258 = 0;
  qword_501E260 = 0;
  qword_501E280 = 0;
  qword_501E288 = (__int64)&unk_501E2A0;
  qword_501E290 = 1;
  dword_501E298 = 0;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_501E270;
  v19 = (unsigned int)qword_501E270 + 1LL;
  if ( v19 > HIDWORD(qword_501E270) )
  {
    sub_C8D5F0((char *)&unk_501E278 - 16, &unk_501E278, v19, 8);
    v18 = (unsigned int)qword_501E270;
  }
  *(_QWORD *)(qword_501E268 + 8 * v18) = v17;
  qword_501E2B0 = (__int64)&unk_49D9748;
  qword_501E220 = (__int64)&unk_49DC090;
  qword_501E2C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_501E270) = qword_501E270 + 1;
  qword_501E2E0 = (__int64)nullsub_23;
  qword_501E2A8 = 0;
  qword_501E2D8 = (__int64)sub_984030;
  qword_501E2B8 = 0;
  sub_C53080(&qword_501E220, "force-hardware-loop-guard", 25);
  LOBYTE(qword_501E2A8) = 0;
  qword_501E250 = 40;
  LOBYTE(dword_501E22C) = dword_501E22C & 0x9F | 0x20;
  LOWORD(qword_501E2B8) = 256;
  qword_501E248 = (__int64)"Force generation of loop guard intrinsic";
  sub_C53130(&qword_501E220);
  return __cxa_atexit(sub_984900, &qword_501E220, &qword_4A427C0);
}
