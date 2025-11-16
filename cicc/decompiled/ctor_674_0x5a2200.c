// Function: ctor_674
// Address: 0x5a2200
//
int __fastcall ctor_674(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_503D6C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503D710 = 0x100000000LL;
  dword_503D6CC &= 0x8000u;
  word_503D6D0 = 0;
  qword_503D6D8 = 0;
  qword_503D6E0 = 0;
  dword_503D6C8 = v4;
  qword_503D6E8 = 0;
  qword_503D6F0 = 0;
  qword_503D6F8 = 0;
  qword_503D700 = 0;
  qword_503D708 = (__int64)&unk_503D718;
  qword_503D720 = 0;
  qword_503D728 = (__int64)&unk_503D740;
  qword_503D730 = 1;
  dword_503D738 = 0;
  byte_503D73C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503D710;
  v7 = (unsigned int)qword_503D710 + 1LL;
  if ( v7 > HIDWORD(qword_503D710) )
  {
    sub_C8D5F0((char *)&unk_503D718 - 16, &unk_503D718, v7, 8);
    v6 = (unsigned int)qword_503D710;
  }
  *(_QWORD *)(qword_503D708 + 8 * v6) = v5;
  qword_503D750 = (__int64)&unk_49D9728;
  qword_503D6C0 = (__int64)&unk_49DBF10;
  LODWORD(qword_503D710) = qword_503D710 + 1;
  qword_503D748 = 0;
  qword_503D760 = (__int64)&unk_49DC290;
  qword_503D758 = 0;
  qword_503D780 = (__int64)nullsub_24;
  qword_503D778 = (__int64)sub_984050;
  sub_C53080(&qword_503D6C0, "mfs-psi-cutoff", 14);
  qword_503D6F0 = 87;
  qword_503D6E8 = (__int64)"Percentile profile summary cutoff used to determine cold blocks. Unused if set to zero.";
  LODWORD(qword_503D748) = 999950;
  BYTE4(qword_503D758) = 1;
  LODWORD(qword_503D758) = 999950;
  LOBYTE(dword_503D6CC) = dword_503D6CC & 0x9F | 0x20;
  sub_C53130(&qword_503D6C0);
  __cxa_atexit(sub_984970, &qword_503D6C0, &qword_4A427C0);
  qword_503D5E0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503D6C0, v8, v9), 1u);
  qword_503D630 = 0x100000000LL;
  word_503D5F0 = 0;
  dword_503D5EC &= 0x8000u;
  qword_503D5F8 = 0;
  qword_503D600 = 0;
  dword_503D5E8 = v10;
  qword_503D608 = 0;
  qword_503D610 = 0;
  qword_503D618 = 0;
  qword_503D620 = 0;
  qword_503D628 = (__int64)&unk_503D638;
  qword_503D640 = 0;
  qword_503D648 = (__int64)&unk_503D660;
  qword_503D650 = 1;
  dword_503D658 = 0;
  byte_503D65C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503D630;
  if ( (unsigned __int64)(unsigned int)qword_503D630 + 1 > HIDWORD(qword_503D630) )
  {
    v20 = v11;
    sub_C8D5F0((char *)&unk_503D638 - 16, &unk_503D638, (unsigned int)qword_503D630 + 1LL, 8);
    v12 = (unsigned int)qword_503D630;
    v11 = v20;
  }
  *(_QWORD *)(qword_503D628 + 8 * v12) = v11;
  qword_503D670 = (__int64)&unk_49D9728;
  qword_503D5E0 = (__int64)&unk_49DBF10;
  LODWORD(qword_503D630) = qword_503D630 + 1;
  qword_503D668 = 0;
  qword_503D680 = (__int64)&unk_49DC290;
  qword_503D678 = 0;
  qword_503D6A0 = (__int64)nullsub_24;
  qword_503D698 = (__int64)sub_984050;
  sub_C53080(&qword_503D5E0, "mfs-count-threshold", 19);
  qword_503D610 = 64;
  qword_503D608 = (__int64)"Minimum number of times a block must be executed to be retained.";
  LODWORD(qword_503D668) = 1;
  BYTE4(qword_503D678) = 1;
  LODWORD(qword_503D678) = 1;
  LOBYTE(dword_503D5EC) = dword_503D5EC & 0x9F | 0x20;
  sub_C53130(&qword_503D5E0);
  __cxa_atexit(sub_984970, &qword_503D5E0, &qword_4A427C0);
  qword_503D500 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_503D5E0, v13, v14), 1u);
  byte_503D57C = 1;
  qword_503D550 = 0x100000000LL;
  dword_503D50C &= 0x8000u;
  qword_503D518 = 0;
  qword_503D520 = 0;
  qword_503D528 = 0;
  dword_503D508 = v15;
  word_503D510 = 0;
  qword_503D530 = 0;
  qword_503D538 = 0;
  qword_503D540 = 0;
  qword_503D548 = (__int64)&unk_503D558;
  qword_503D560 = 0;
  qword_503D568 = (__int64)&unk_503D580;
  qword_503D570 = 1;
  dword_503D578 = 0;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_503D550;
  v18 = (unsigned int)qword_503D550 + 1LL;
  if ( v18 > HIDWORD(qword_503D550) )
  {
    sub_C8D5F0((char *)&unk_503D558 - 16, &unk_503D558, v18, 8);
    v17 = (unsigned int)qword_503D550;
  }
  *(_QWORD *)(qword_503D548 + 8 * v17) = v16;
  LODWORD(qword_503D550) = qword_503D550 + 1;
  qword_503D588 = 0;
  qword_503D590 = (__int64)&unk_49D9748;
  qword_503D598 = 0;
  qword_503D500 = (__int64)&unk_49DC090;
  qword_503D5A0 = (__int64)&unk_49DC1D0;
  qword_503D5C0 = (__int64)nullsub_23;
  qword_503D5B8 = (__int64)sub_984030;
  sub_C53080(&qword_503D500, "mfs-split-ehcode", 16);
  qword_503D530 = 51;
  qword_503D528 = (__int64)"Splits all EH code and it's descendants by default.";
  LOWORD(qword_503D598) = 256;
  LOBYTE(qword_503D588) = 0;
  LOBYTE(dword_503D50C) = dword_503D50C & 0x9F | 0x20;
  sub_C53130(&qword_503D500);
  return __cxa_atexit(sub_984900, &qword_503D500, &qword_4A427C0);
}
