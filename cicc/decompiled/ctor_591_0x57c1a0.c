// Function: ctor_591
// Address: 0x57c1a0
//
int __fastcall ctor_591(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v32; // [rsp+0h] [rbp-60h]
  int v33; // [rsp+10h] [rbp-50h] BYREF
  int v34; // [rsp+14h] [rbp-4Ch] BYREF
  int *v35; // [rsp+18h] [rbp-48h] BYREF
  const char *v36; // [rsp+20h] [rbp-40h] BYREF
  __int64 v37; // [rsp+28h] [rbp-38h]

  qword_5025B40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5025B90 = 0x100000000LL;
  dword_5025B4C &= 0x8000u;
  word_5025B50 = 0;
  qword_5025B58 = 0;
  qword_5025B60 = 0;
  dword_5025B48 = v4;
  qword_5025B68 = 0;
  qword_5025B70 = 0;
  qword_5025B78 = 0;
  qword_5025B80 = 0;
  qword_5025B88 = (__int64)&unk_5025B98;
  qword_5025BA0 = 0;
  qword_5025BA8 = (__int64)&unk_5025BC0;
  qword_5025BB0 = 1;
  dword_5025BB8 = 0;
  byte_5025BBC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5025B90;
  v7 = (unsigned int)qword_5025B90 + 1LL;
  if ( v7 > HIDWORD(qword_5025B90) )
  {
    sub_C8D5F0((char *)&unk_5025B98 - 16, &unk_5025B98, v7, 8);
    v6 = (unsigned int)qword_5025B90;
  }
  *(_QWORD *)(qword_5025B88 + 8 * v6) = v5;
  LODWORD(qword_5025B90) = qword_5025B90 + 1;
  qword_5025BC8 = 0;
  qword_5025BD0 = (__int64)&unk_49D9728;
  qword_5025BD8 = 0;
  qword_5025B40 = (__int64)&unk_49DBF10;
  qword_5025BE0 = (__int64)&unk_49DC290;
  qword_5025C00 = (__int64)nullsub_24;
  qword_5025BF8 = (__int64)sub_984050;
  sub_C53080(&qword_5025B40, "cold-operand-threshold", 22);
  qword_5025B70 = 63;
  qword_5025B68 = (__int64)"Maximum frequency of path for an operand to be considered cold.";
  LODWORD(qword_5025BC8) = 20;
  BYTE4(qword_5025BD8) = 1;
  LODWORD(qword_5025BD8) = 20;
  LOBYTE(dword_5025B4C) = dword_5025B4C & 0x9F | 0x20;
  sub_C53130(&qword_5025B40);
  __cxa_atexit(sub_984970, &qword_5025B40, &qword_4A427C0);
  qword_5025A60 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5025B40, v8, v9), 1u);
  qword_5025AB0 = 0x100000000LL;
  dword_5025A6C &= 0x8000u;
  word_5025A70 = 0;
  qword_5025A78 = 0;
  qword_5025A80 = 0;
  dword_5025A68 = v10;
  qword_5025A88 = 0;
  qword_5025A90 = 0;
  qword_5025A98 = 0;
  qword_5025AA0 = 0;
  qword_5025AA8 = (__int64)&unk_5025AB8;
  qword_5025AC0 = 0;
  qword_5025AC8 = (__int64)&unk_5025AE0;
  qword_5025AD0 = 1;
  dword_5025AD8 = 0;
  byte_5025ADC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5025AB0;
  v13 = (unsigned int)qword_5025AB0 + 1LL;
  if ( v13 > HIDWORD(qword_5025AB0) )
  {
    sub_C8D5F0((char *)&unk_5025AB8 - 16, &unk_5025AB8, v13, 8);
    v12 = (unsigned int)qword_5025AB0;
  }
  *(_QWORD *)(qword_5025AA8 + 8 * v12) = v11;
  LODWORD(qword_5025AB0) = qword_5025AB0 + 1;
  qword_5025AE8 = 0;
  qword_5025AF0 = (__int64)&unk_49D9728;
  qword_5025AF8 = 0;
  qword_5025A60 = (__int64)&unk_49DBF10;
  qword_5025B00 = (__int64)&unk_49DC290;
  qword_5025B20 = (__int64)nullsub_24;
  qword_5025B18 = (__int64)sub_984050;
  sub_C53080(&qword_5025A60, "cold-operand-max-cost-multiplier", 32);
  qword_5025A90 = 113;
  qword_5025A88 = (__int64)"Maximum cost multiplier of TCC_expensive for the dependence slice of a cold operand to be con"
                           "sidered inexpensive.";
  LODWORD(qword_5025AE8) = 1;
  BYTE4(qword_5025AF8) = 1;
  LODWORD(qword_5025AF8) = 1;
  LOBYTE(dword_5025A6C) = dword_5025A6C & 0x9F | 0x20;
  sub_C53130(&qword_5025A60);
  __cxa_atexit(sub_984970, &qword_5025A60, &qword_4A427C0);
  v35 = &v34;
  v36 = "Gradient gain threshold (%).";
  v33 = 1;
  v34 = 25;
  v37 = 28;
  sub_2F9C520(&unk_5025980, "select-opti-loop-gradient-gain-threshold", &v36, &v35, &v33);
  __cxa_atexit(sub_984970, &unk_5025980, &qword_4A427C0);
  qword_50258A0 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &unk_5025980, v14, v15), 1u);
  byte_502591C = 1;
  word_50258B0 = 0;
  qword_50258F0 = 0x100000000LL;
  dword_50258AC &= 0x8000u;
  qword_50258E8 = (__int64)&unk_50258F8;
  qword_50258B8 = 0;
  dword_50258A8 = v16;
  qword_50258C0 = 0;
  qword_50258C8 = 0;
  qword_50258D0 = 0;
  qword_50258D8 = 0;
  qword_50258E0 = 0;
  qword_5025900 = 0;
  qword_5025908 = (__int64)&unk_5025920;
  qword_5025910 = 1;
  dword_5025918 = 0;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_50258F0;
  if ( (unsigned __int64)(unsigned int)qword_50258F0 + 1 > HIDWORD(qword_50258F0) )
  {
    v32 = v17;
    sub_C8D5F0((char *)&unk_50258F8 - 16, &unk_50258F8, (unsigned int)qword_50258F0 + 1LL, 8);
    v18 = (unsigned int)qword_50258F0;
    v17 = v32;
  }
  *(_QWORD *)(qword_50258E8 + 8 * v18) = v17;
  LODWORD(qword_50258F0) = qword_50258F0 + 1;
  qword_5025928 = 0;
  qword_5025930 = (__int64)&unk_49D9728;
  qword_5025938 = 0;
  qword_50258A0 = (__int64)&unk_49DBF10;
  qword_5025940 = (__int64)&unk_49DC290;
  qword_5025960 = (__int64)nullsub_24;
  qword_5025958 = (__int64)sub_984050;
  sub_C53080(&qword_50258A0, "select-opti-loop-cycle-gain-threshold", 37);
  qword_50258D0 = 44;
  qword_50258C8 = (__int64)"Minimum gain per loop (in cycles) threshold.";
  LODWORD(qword_5025928) = 4;
  BYTE4(qword_5025938) = 1;
  LODWORD(qword_5025938) = 4;
  LOBYTE(dword_50258AC) = dword_50258AC & 0x9F | 0x20;
  sub_C53130(&qword_50258A0);
  __cxa_atexit(sub_984970, &qword_50258A0, &qword_4A427C0);
  v33 = 1;
  v35 = &v34;
  v36 = "Minimum relative gain per loop threshold (1/X). Defaults to 12.5%";
  v34 = 8;
  v37 = 65;
  sub_2F9C520(&unk_50257C0, "select-opti-loop-relative-gain-threshold", &v36, &v35, &v33);
  __cxa_atexit(sub_984970, &unk_50257C0, &qword_4A427C0);
  qword_50256E0 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &unk_50257C0, v19, v20), 1u);
  qword_5025730 = 0x100000000LL;
  dword_50256EC &= 0x8000u;
  word_50256F0 = 0;
  qword_5025728 = (__int64)&unk_5025738;
  qword_50256F8 = 0;
  dword_50256E8 = v21;
  qword_5025700 = 0;
  qword_5025708 = 0;
  qword_5025710 = 0;
  qword_5025718 = 0;
  qword_5025720 = 0;
  qword_5025740 = 0;
  qword_5025748 = (__int64)&unk_5025760;
  qword_5025750 = 1;
  dword_5025758 = 0;
  byte_502575C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_5025730;
  v24 = (unsigned int)qword_5025730 + 1LL;
  if ( v24 > HIDWORD(qword_5025730) )
  {
    sub_C8D5F0((char *)&unk_5025738 - 16, &unk_5025738, v24, 8);
    v23 = (unsigned int)qword_5025730;
  }
  *(_QWORD *)(qword_5025728 + 8 * v23) = v22;
  LODWORD(qword_5025730) = qword_5025730 + 1;
  qword_5025768 = 0;
  qword_5025770 = (__int64)&unk_49D9728;
  qword_5025778 = 0;
  qword_50256E0 = (__int64)&unk_49DBF10;
  qword_5025780 = (__int64)&unk_49DC290;
  qword_50257A0 = (__int64)nullsub_24;
  qword_5025798 = (__int64)sub_984050;
  sub_C53080(&qword_50256E0, "mispredict-default-rate", 23);
  LODWORD(qword_5025768) = 25;
  BYTE4(qword_5025778) = 1;
  LODWORD(qword_5025778) = 25;
  qword_5025710 = 45;
  LOBYTE(dword_50256EC) = dword_50256EC & 0x9F | 0x20;
  qword_5025708 = (__int64)"Default mispredict rate (initialized to 25%).";
  sub_C53130(&qword_50256E0);
  __cxa_atexit(sub_984970, &qword_50256E0, &qword_4A427C0);
  qword_5025600 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_50256E0, v25, v26), 1u);
  byte_502567C = 1;
  qword_5025650 = 0x100000000LL;
  dword_502560C &= 0x8000u;
  qword_5025618 = 0;
  qword_5025620 = 0;
  qword_5025628 = 0;
  dword_5025608 = v27;
  word_5025610 = 0;
  qword_5025630 = 0;
  qword_5025638 = 0;
  qword_5025640 = 0;
  qword_5025648 = (__int64)&unk_5025658;
  qword_5025660 = 0;
  qword_5025668 = (__int64)&unk_5025680;
  qword_5025670 = 1;
  dword_5025678 = 0;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_5025650;
  v30 = (unsigned int)qword_5025650 + 1LL;
  if ( v30 > HIDWORD(qword_5025650) )
  {
    sub_C8D5F0((char *)&unk_5025658 - 16, &unk_5025658, v30, 8);
    v29 = (unsigned int)qword_5025650;
  }
  *(_QWORD *)(qword_5025648 + 8 * v29) = v28;
  LODWORD(qword_5025650) = qword_5025650 + 1;
  qword_5025688 = 0;
  qword_5025690 = (__int64)&unk_49D9748;
  qword_5025698 = 0;
  qword_5025600 = (__int64)&unk_49DC090;
  qword_50256A0 = (__int64)&unk_49DC1D0;
  qword_50256C0 = (__int64)nullsub_23;
  qword_50256B8 = (__int64)sub_984030;
  sub_C53080(&qword_5025600, "disable-loop-level-heuristics", 29);
  LOBYTE(qword_5025688) = 0;
  qword_5025630 = 30;
  LOBYTE(dword_502560C) = dword_502560C & 0x9F | 0x20;
  LOWORD(qword_5025698) = 256;
  qword_5025628 = (__int64)"Disable loop-level heuristics.";
  sub_C53130(&qword_5025600);
  return __cxa_atexit(sub_984900, &qword_5025600, &qword_4A427C0);
}
