// Function: ctor_404_0
// Address: 0x529b30
//
int ctor_404_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  _QWORD v33[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v34[8]; // [rsp+20h] [rbp-40h] BYREF

  qword_4FEA6E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEA6EC &= 0x8000u;
  word_4FEA6F0 = 0;
  qword_4FEA730 = 0x100000000LL;
  qword_4FEA6F8 = 0;
  qword_4FEA700 = 0;
  qword_4FEA708 = 0;
  dword_4FEA6E8 = v0;
  qword_4FEA710 = 0;
  qword_4FEA718 = 0;
  qword_4FEA720 = 0;
  qword_4FEA728 = (__int64)&unk_4FEA738;
  qword_4FEA740 = 0;
  qword_4FEA748 = (__int64)&unk_4FEA760;
  qword_4FEA750 = 1;
  dword_4FEA758 = 0;
  byte_4FEA75C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEA730;
  v3 = (unsigned int)qword_4FEA730 + 1LL;
  if ( v3 > HIDWORD(qword_4FEA730) )
  {
    sub_C8D5F0((char *)&unk_4FEA738 - 16, &unk_4FEA738, v3, 8);
    v2 = (unsigned int)qword_4FEA730;
  }
  *(_QWORD *)(qword_4FEA728 + 8 * v2) = v1;
  qword_4FEA768 = (__int64)&byte_4FEA778;
  qword_4FEA790 = (__int64)&byte_4FEA7A0;
  LODWORD(qword_4FEA730) = qword_4FEA730 + 1;
  qword_4FEA770 = 0;
  qword_4FEA788 = (__int64)&unk_49DC130;
  byte_4FEA778 = 0;
  byte_4FEA7A0 = 0;
  qword_4FEA6E0 = (__int64)&unk_49DC010;
  qword_4FEA798 = 0;
  byte_4FEA7B0 = 0;
  qword_4FEA7B8 = (__int64)&unk_49DC350;
  qword_4FEA7D8 = (__int64)nullsub_92;
  qword_4FEA7D0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FEA6E0, "nsan-shadow-type-mapping", 24);
  v33[0] = v34;
  LODWORD(v34[0]) = 7434596;
  v33[1] = 3;
  sub_2240AE0(&qword_4FEA768, v33);
  byte_4FEA7B0 = 1;
  sub_2240AE0(&qword_4FEA790, v33);
  if ( (_QWORD *)v33[0] != v34 )
    j_j___libc_free_0(v33[0], v34[0] + 1LL);
  qword_4FEA710 = 247;
  qword_4FEA708 = (__int64)"One shadow type id for each of `float`, `double`, `long double`. `d`,`l`,`q`,`e` mean double,"
                           " x86_fp80, fp128 (quad) and ppc_fp128 (extended double) respectively. The default is to shado"
                           "w `float` as `double`, and `double` and `x86_fp80` as `fp128`";
  LOBYTE(dword_4FEA6EC) = dword_4FEA6EC & 0x9F | 0x20;
  sub_C53130(&qword_4FEA6E0);
  __cxa_atexit(sub_BC5A40, &qword_4FEA6E0, &qword_4A427C0);
  qword_4FEA600 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEA60C &= 0x8000u;
  word_4FEA610 = 0;
  qword_4FEA650 = 0x100000000LL;
  qword_4FEA618 = 0;
  qword_4FEA620 = 0;
  qword_4FEA628 = 0;
  dword_4FEA608 = v4;
  qword_4FEA630 = 0;
  qword_4FEA638 = 0;
  qword_4FEA640 = 0;
  qword_4FEA648 = (__int64)&unk_4FEA658;
  qword_4FEA660 = 0;
  qword_4FEA668 = (__int64)&unk_4FEA680;
  qword_4FEA670 = 1;
  dword_4FEA678 = 0;
  byte_4FEA67C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FEA650;
  v7 = (unsigned int)qword_4FEA650 + 1LL;
  if ( v7 > HIDWORD(qword_4FEA650) )
  {
    sub_C8D5F0((char *)&unk_4FEA658 - 16, &unk_4FEA658, v7, 8);
    v6 = (unsigned int)qword_4FEA650;
  }
  *(_QWORD *)(qword_4FEA648 + 8 * v6) = v5;
  qword_4FEA690 = (__int64)&unk_49D9748;
  qword_4FEA600 = (__int64)&unk_49DC090;
  qword_4FEA6A0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FEA650) = qword_4FEA650 + 1;
  qword_4FEA6C0 = (__int64)nullsub_23;
  qword_4FEA688 = 0;
  qword_4FEA6B8 = (__int64)sub_984030;
  qword_4FEA698 = 0;
  sub_C53080(&qword_4FEA600, "nsan-instrument-fcmp", 20);
  qword_4FEA628 = (__int64)"Instrument floating-point comparisons";
  LOWORD(qword_4FEA698) = 257;
  LOBYTE(qword_4FEA688) = 1;
  qword_4FEA630 = 37;
  LOBYTE(dword_4FEA60C) = dword_4FEA60C & 0x9F | 0x20;
  sub_C53130(&qword_4FEA600);
  __cxa_atexit(sub_984900, &qword_4FEA600, &qword_4A427C0);
  qword_4FEA500 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEA50C &= 0x8000u;
  word_4FEA510 = 0;
  qword_4FEA550 = 0x100000000LL;
  qword_4FEA548 = (__int64)&unk_4FEA558;
  qword_4FEA518 = 0;
  qword_4FEA520 = 0;
  dword_4FEA508 = v8;
  qword_4FEA528 = 0;
  qword_4FEA530 = 0;
  qword_4FEA538 = 0;
  qword_4FEA540 = 0;
  qword_4FEA560 = 0;
  qword_4FEA568 = (__int64)&unk_4FEA580;
  qword_4FEA570 = 1;
  dword_4FEA578 = 0;
  byte_4FEA57C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FEA550;
  if ( (unsigned __int64)(unsigned int)qword_4FEA550 + 1 > HIDWORD(qword_4FEA550) )
  {
    v28 = v9;
    sub_C8D5F0((char *)&unk_4FEA558 - 16, &unk_4FEA558, (unsigned int)qword_4FEA550 + 1LL, 8);
    v10 = (unsigned int)qword_4FEA550;
    v9 = v28;
  }
  *(_QWORD *)(qword_4FEA548 + 8 * v10) = v9;
  qword_4FEA588 = (__int64)&byte_4FEA598;
  qword_4FEA5B0 = (__int64)&byte_4FEA5C0;
  LODWORD(qword_4FEA550) = qword_4FEA550 + 1;
  qword_4FEA590 = 0;
  qword_4FEA5A8 = (__int64)&unk_49DC130;
  byte_4FEA598 = 0;
  byte_4FEA5C0 = 0;
  qword_4FEA500 = (__int64)&unk_49DC010;
  qword_4FEA5B8 = 0;
  byte_4FEA5D0 = 0;
  qword_4FEA5D8 = (__int64)&unk_49DC350;
  qword_4FEA5F8 = (__int64)nullsub_92;
  qword_4FEA5F0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FEA500, "check-functions-filter", 22);
  qword_4FEA530 = 90;
  qword_4FEA528 = (__int64)"Only emit checks for arguments of functions whose names match the given regular expression";
  qword_4FEA538 = (__int64)"regex";
  qword_4FEA540 = 5;
  sub_C53130(&qword_4FEA500);
  __cxa_atexit(sub_BC5A40, &qword_4FEA500, &qword_4A427C0);
  qword_4FEA420 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEA470 = 0x100000000LL;
  dword_4FEA42C &= 0x8000u;
  qword_4FEA468 = (__int64)&unk_4FEA478;
  word_4FEA430 = 0;
  qword_4FEA438 = 0;
  dword_4FEA428 = v11;
  qword_4FEA440 = 0;
  qword_4FEA448 = 0;
  qword_4FEA450 = 0;
  qword_4FEA458 = 0;
  qword_4FEA460 = 0;
  qword_4FEA480 = 0;
  qword_4FEA488 = (__int64)&unk_4FEA4A0;
  qword_4FEA490 = 1;
  dword_4FEA498 = 0;
  byte_4FEA49C = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_4FEA470;
  if ( (unsigned __int64)(unsigned int)qword_4FEA470 + 1 > HIDWORD(qword_4FEA470) )
  {
    v29 = v12;
    sub_C8D5F0((char *)&unk_4FEA478 - 16, &unk_4FEA478, (unsigned int)qword_4FEA470 + 1LL, 8);
    v13 = (unsigned int)qword_4FEA470;
    v12 = v29;
  }
  *(_QWORD *)(qword_4FEA468 + 8 * v13) = v12;
  qword_4FEA4B0 = (__int64)&unk_49D9748;
  qword_4FEA420 = (__int64)&unk_49DC090;
  qword_4FEA4C0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FEA470) = qword_4FEA470 + 1;
  qword_4FEA4E0 = (__int64)nullsub_23;
  qword_4FEA4A8 = 0;
  qword_4FEA4D8 = (__int64)sub_984030;
  qword_4FEA4B8 = 0;
  sub_C53080(&qword_4FEA420, "nsan-truncate-fcmp-eq", 21);
  qword_4FEA448 = (__int64)"This flag controls the behaviour of fcmp equality comparisons.For equality comparisons such a"
                           "s `x == 0.0f`, we can perform the shadow check in the shadow (`x_shadow == 0.0) == (x == 0.0f"
                           ")`) or app  domain (`(trunc(x_shadow) == 0.0f) == (x == 0.0f)`). This helps catch the case wh"
                           "en `x_shadow` is accurate enough (and therefore close enough to zero) so that `trunc(x_shadow"
                           ")` is zero even though both `x` and `x_shadow` are not";
  LOWORD(qword_4FEA4B8) = 257;
  LOBYTE(qword_4FEA4A8) = 1;
  qword_4FEA450 = 426;
  LOBYTE(dword_4FEA42C) = dword_4FEA42C & 0x9F | 0x20;
  sub_C53130(&qword_4FEA420);
  __cxa_atexit(sub_984900, &qword_4FEA420, &qword_4A427C0);
  qword_4FEA340 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEA390 = 0x100000000LL;
  dword_4FEA34C &= 0x8000u;
  qword_4FEA388 = (__int64)&unk_4FEA398;
  word_4FEA350 = 0;
  qword_4FEA358 = 0;
  dword_4FEA348 = v14;
  qword_4FEA360 = 0;
  qword_4FEA368 = 0;
  qword_4FEA370 = 0;
  qword_4FEA378 = 0;
  qword_4FEA380 = 0;
  qword_4FEA3A0 = 0;
  qword_4FEA3A8 = (__int64)&unk_4FEA3C0;
  qword_4FEA3B0 = 1;
  dword_4FEA3B8 = 0;
  byte_4FEA3BC = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FEA390;
  if ( (unsigned __int64)(unsigned int)qword_4FEA390 + 1 > HIDWORD(qword_4FEA390) )
  {
    v30 = v15;
    sub_C8D5F0((char *)&unk_4FEA398 - 16, &unk_4FEA398, (unsigned int)qword_4FEA390 + 1LL, 8);
    v16 = (unsigned int)qword_4FEA390;
    v15 = v30;
  }
  *(_QWORD *)(qword_4FEA388 + 8 * v16) = v15;
  qword_4FEA3D0 = (__int64)&unk_49D9748;
  qword_4FEA340 = (__int64)&unk_49DC090;
  qword_4FEA3E0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FEA390) = qword_4FEA390 + 1;
  qword_4FEA400 = (__int64)nullsub_23;
  qword_4FEA3C8 = 0;
  qword_4FEA3F8 = (__int64)sub_984030;
  qword_4FEA3D8 = 0;
  sub_C53080(&qword_4FEA340, "nsan-check-loads", 16);
  qword_4FEA370 = 25;
  qword_4FEA368 = (__int64)"Check floating-point load";
  LOBYTE(dword_4FEA34C) = dword_4FEA34C & 0x9F | 0x20;
  sub_C53130(&qword_4FEA340);
  __cxa_atexit(sub_984900, &qword_4FEA340, &qword_4A427C0);
  qword_4FEA260 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FEA2B0 = 0x100000000LL;
  dword_4FEA26C &= 0x8000u;
  qword_4FEA2A8 = (__int64)&unk_4FEA2B8;
  word_4FEA270 = 0;
  qword_4FEA278 = 0;
  dword_4FEA268 = v17;
  qword_4FEA280 = 0;
  qword_4FEA288 = 0;
  qword_4FEA290 = 0;
  qword_4FEA298 = 0;
  qword_4FEA2A0 = 0;
  qword_4FEA2C0 = 0;
  qword_4FEA2C8 = (__int64)&unk_4FEA2E0;
  qword_4FEA2D0 = 1;
  dword_4FEA2D8 = 0;
  byte_4FEA2DC = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_4FEA2B0;
  if ( (unsigned __int64)(unsigned int)qword_4FEA2B0 + 1 > HIDWORD(qword_4FEA2B0) )
  {
    v31 = v18;
    sub_C8D5F0((char *)&unk_4FEA2B8 - 16, &unk_4FEA2B8, (unsigned int)qword_4FEA2B0 + 1LL, 8);
    v19 = (unsigned int)qword_4FEA2B0;
    v18 = v31;
  }
  *(_QWORD *)(qword_4FEA2A8 + 8 * v19) = v18;
  qword_4FEA2F0 = (__int64)&unk_49D9748;
  qword_4FEA260 = (__int64)&unk_49DC090;
  qword_4FEA300 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FEA2B0) = qword_4FEA2B0 + 1;
  qword_4FEA320 = (__int64)nullsub_23;
  qword_4FEA2E8 = 0;
  qword_4FEA318 = (__int64)sub_984030;
  qword_4FEA2F8 = 0;
  sub_C53080(&qword_4FEA260, "nsan-check-stores", 17);
  qword_4FEA288 = (__int64)"Check floating-point stores";
  LOWORD(qword_4FEA2F8) = 257;
  LOBYTE(qword_4FEA2E8) = 1;
  qword_4FEA290 = 27;
  LOBYTE(dword_4FEA26C) = dword_4FEA26C & 0x9F | 0x20;
  sub_C53130(&qword_4FEA260);
  __cxa_atexit(sub_984900, &qword_4FEA260, &qword_4A427C0);
  qword_4FEA180 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FEA1FC = 1;
  word_4FEA190 = 0;
  qword_4FEA1D0 = 0x100000000LL;
  dword_4FEA18C &= 0x8000u;
  qword_4FEA1C8 = (__int64)&unk_4FEA1D8;
  qword_4FEA198 = 0;
  dword_4FEA188 = v20;
  qword_4FEA1A0 = 0;
  qword_4FEA1A8 = 0;
  qword_4FEA1B0 = 0;
  qword_4FEA1B8 = 0;
  qword_4FEA1C0 = 0;
  qword_4FEA1E0 = 0;
  qword_4FEA1E8 = (__int64)&unk_4FEA200;
  qword_4FEA1F0 = 1;
  dword_4FEA1F8 = 0;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_4FEA1D0;
  if ( (unsigned __int64)(unsigned int)qword_4FEA1D0 + 1 > HIDWORD(qword_4FEA1D0) )
  {
    v32 = v21;
    sub_C8D5F0((char *)&unk_4FEA1D8 - 16, &unk_4FEA1D8, (unsigned int)qword_4FEA1D0 + 1LL, 8);
    v22 = (unsigned int)qword_4FEA1D0;
    v21 = v32;
  }
  *(_QWORD *)(qword_4FEA1C8 + 8 * v22) = v21;
  qword_4FEA210 = (__int64)&unk_49D9748;
  qword_4FEA180 = (__int64)&unk_49DC090;
  qword_4FEA220 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FEA1D0) = qword_4FEA1D0 + 1;
  qword_4FEA240 = (__int64)nullsub_23;
  qword_4FEA208 = 0;
  qword_4FEA238 = (__int64)sub_984030;
  qword_4FEA218 = 0;
  sub_C53080(&qword_4FEA180, "nsan-check-ret", 14);
  LOBYTE(qword_4FEA208) = 1;
  LOWORD(qword_4FEA218) = 257;
  qword_4FEA1A8 = (__int64)"Check floating-point return values";
  qword_4FEA1B0 = 34;
  LOBYTE(dword_4FEA18C) = dword_4FEA18C & 0x9F | 0x20;
  sub_C53130(&qword_4FEA180);
  __cxa_atexit(sub_984900, &qword_4FEA180, &qword_4A427C0);
  qword_4FEA0A0 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FEA11C = 1;
  qword_4FEA0F0 = 0x100000000LL;
  dword_4FEA0AC &= 0x8000u;
  qword_4FEA0E8 = (__int64)&unk_4FEA0F8;
  qword_4FEA0B8 = 0;
  qword_4FEA0C0 = 0;
  dword_4FEA0A8 = v23;
  word_4FEA0B0 = 0;
  qword_4FEA0C8 = 0;
  qword_4FEA0D0 = 0;
  qword_4FEA0D8 = 0;
  qword_4FEA0E0 = 0;
  qword_4FEA100 = 0;
  qword_4FEA108 = (__int64)&unk_4FEA120;
  qword_4FEA110 = 1;
  dword_4FEA118 = 0;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_4FEA0F0;
  v26 = (unsigned int)qword_4FEA0F0 + 1LL;
  if ( v26 > HIDWORD(qword_4FEA0F0) )
  {
    sub_C8D5F0((char *)&unk_4FEA0F8 - 16, &unk_4FEA0F8, v26, 8);
    v25 = (unsigned int)qword_4FEA0F0;
  }
  *(_QWORD *)(qword_4FEA0E8 + 8 * v25) = v24;
  qword_4FEA130 = (__int64)&unk_49D9748;
  qword_4FEA0A0 = (__int64)&unk_49DC090;
  qword_4FEA140 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FEA0F0) = qword_4FEA0F0 + 1;
  qword_4FEA160 = (__int64)nullsub_23;
  qword_4FEA128 = 0;
  qword_4FEA158 = (__int64)sub_984030;
  qword_4FEA138 = 0;
  sub_C53080(&qword_4FEA0A0, "nsan-propagate-non-ft-const-stores-as-ft", 40);
  qword_4FEA0D0 = 94;
  qword_4FEA0C8 = (__int64)"Propagate non floating-point const stores as floating point values.For debugging purposes only";
  LOBYTE(dword_4FEA0AC) = dword_4FEA0AC & 0x9F | 0x20;
  sub_C53130(&qword_4FEA0A0);
  return __cxa_atexit(sub_984900, &qword_4FEA0A0, &qword_4A427C0);
}
