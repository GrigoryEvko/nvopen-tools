// Function: ctor_066
// Address: 0x495bf0
//
int ctor_066()
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
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  qword_4F8A7C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8A810 = 0x100000000LL;
  dword_4F8A7CC &= 0x8000u;
  word_4F8A7D0 = 0;
  qword_4F8A7D8 = 0;
  qword_4F8A7E0 = 0;
  dword_4F8A7C8 = v0;
  qword_4F8A7E8 = 0;
  qword_4F8A7F0 = 0;
  qword_4F8A7F8 = 0;
  qword_4F8A800 = 0;
  qword_4F8A808 = (__int64)&unk_4F8A818;
  qword_4F8A820 = 0;
  qword_4F8A828 = (__int64)&unk_4F8A840;
  qword_4F8A830 = 1;
  dword_4F8A838 = 0;
  byte_4F8A83C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8A810;
  v3 = (unsigned int)qword_4F8A810 + 1LL;
  if ( v3 > HIDWORD(qword_4F8A810) )
  {
    sub_C8D5F0((char *)&unk_4F8A818 - 16, &unk_4F8A818, v3, 8);
    v2 = (unsigned int)qword_4F8A810;
  }
  *(_QWORD *)(qword_4F8A808 + 8 * v2) = v1;
  qword_4F8A850 = (__int64)&unk_49D9748;
  qword_4F8A7C0 = (__int64)&unk_49DC090;
  LODWORD(qword_4F8A810) = qword_4F8A810 + 1;
  qword_4F8A848 = 0;
  qword_4F8A860 = (__int64)&unk_49DC1D0;
  qword_4F8A858 = 0;
  qword_4F8A880 = (__int64)nullsub_23;
  qword_4F8A878 = (__int64)sub_984030;
  sub_C53080(&qword_4F8A7C0, "static-func-full-module-prefix", 30);
  LOWORD(qword_4F8A858) = 257;
  LOBYTE(qword_4F8A848) = 1;
  qword_4F8A7F0 = 78;
  LOBYTE(dword_4F8A7CC) = dword_4F8A7CC & 0x9F | 0x20;
  qword_4F8A7E8 = (__int64)"Use full module build paths in the profile counter names for static functions.";
  sub_C53130(&qword_4F8A7C0);
  __cxa_atexit(sub_984900, &qword_4F8A7C0, &qword_4A427C0);
  qword_4F8A6E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8A730 = 0x100000000LL;
  dword_4F8A6EC &= 0x8000u;
  word_4F8A6F0 = 0;
  qword_4F8A6F8 = 0;
  qword_4F8A700 = 0;
  dword_4F8A6E8 = v4;
  qword_4F8A708 = 0;
  qword_4F8A710 = 0;
  qword_4F8A718 = 0;
  qword_4F8A720 = 0;
  qword_4F8A728 = (__int64)&unk_4F8A738;
  qword_4F8A740 = 0;
  qword_4F8A748 = (__int64)&unk_4F8A760;
  qword_4F8A750 = 1;
  dword_4F8A758 = 0;
  byte_4F8A75C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8A730;
  if ( (unsigned __int64)(unsigned int)qword_4F8A730 + 1 > HIDWORD(qword_4F8A730) )
  {
    v17 = v5;
    sub_C8D5F0((char *)&unk_4F8A738 - 16, &unk_4F8A738, (unsigned int)qword_4F8A730 + 1LL, 8);
    v6 = (unsigned int)qword_4F8A730;
    v5 = v17;
  }
  *(_QWORD *)(qword_4F8A728 + 8 * v6) = v5;
  LODWORD(qword_4F8A730) = qword_4F8A730 + 1;
  qword_4F8A768 = 0;
  qword_4F8A770 = (__int64)&unk_49D9728;
  qword_4F8A778 = 0;
  qword_4F8A6E0 = (__int64)&unk_49DBF10;
  qword_4F8A780 = (__int64)&unk_49DC290;
  qword_4F8A7A0 = (__int64)nullsub_24;
  qword_4F8A798 = (__int64)sub_984050;
  sub_C53080(&qword_4F8A6E0, "static-func-strip-dirname-prefix", 32);
  LODWORD(qword_4F8A768) = 0;
  BYTE4(qword_4F8A778) = 1;
  LODWORD(qword_4F8A778) = 0;
  qword_4F8A710 = 106;
  LOBYTE(dword_4F8A6EC) = dword_4F8A6EC & 0x9F | 0x20;
  qword_4F8A708 = (__int64)"Strip specified level of directory name from source path in the profile counter name for static functions.";
  sub_C53130(&qword_4F8A6E0);
  __cxa_atexit(sub_984970, &qword_4F8A6E0, &qword_4A427C0);
  qword_4F8A600 = &unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8A60C = word_4F8A60C & 0x8000;
  qword_4F8A648[1] = 0x100000000LL;
  unk_4F8A608 = v7;
  qword_4F8A648[0] = &qword_4F8A648[2];
  unk_4F8A610 = 0;
  unk_4F8A618 = 0;
  unk_4F8A620 = 0;
  unk_4F8A628 = 0;
  unk_4F8A630 = 0;
  unk_4F8A638 = 0;
  unk_4F8A640 = 0;
  qword_4F8A648[3] = 0;
  qword_4F8A648[4] = &qword_4F8A648[7];
  qword_4F8A648[5] = 1;
  LODWORD(qword_4F8A648[6]) = 0;
  BYTE4(qword_4F8A648[6]) = 1;
  v8 = sub_C57470();
  v9 = LODWORD(qword_4F8A648[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8A648[1]) + 1 > HIDWORD(qword_4F8A648[1]) )
  {
    v18 = v8;
    sub_C8D5F0(qword_4F8A648, &qword_4F8A648[2], LODWORD(qword_4F8A648[1]) + 1LL, 8);
    v9 = LODWORD(qword_4F8A648[1]);
    v8 = v18;
  }
  *(_QWORD *)(qword_4F8A648[0] + 8 * v9) = v8;
  qword_4F8A648[9] = &unk_49D9748;
  qword_4F8A600 = &unk_49DC090;
  ++LODWORD(qword_4F8A648[1]);
  qword_4F8A648[8] = 0;
  qword_4F8A648[11] = &unk_49DC1D0;
  qword_4F8A648[10] = 0;
  qword_4F8A648[15] = nullsub_23;
  qword_4F8A648[14] = sub_984030;
  sub_C53080(&qword_4F8A600, "enable-name-compression", 23);
  unk_4F8A628 = "Enable name/filename string compression";
  LOWORD(qword_4F8A648[10]) = 257;
  unk_4F8A630 = 39;
  LOBYTE(qword_4F8A648[8]) = 1;
  sub_C53130(&qword_4F8A600);
  __cxa_atexit(sub_984900, &qword_4F8A600, &qword_4A427C0);
  qword_4F8A520 = &unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8A52C = word_4F8A52C & 0x8000;
  qword_4F8A568[1] = 0x100000000LL;
  unk_4F8A528 = v10;
  qword_4F8A568[0] = &qword_4F8A568[2];
  unk_4F8A530 = 0;
  unk_4F8A538 = 0;
  unk_4F8A540 = 0;
  unk_4F8A548 = 0;
  unk_4F8A550 = 0;
  unk_4F8A558 = 0;
  unk_4F8A560 = 0;
  qword_4F8A568[3] = 0;
  qword_4F8A568[4] = &qword_4F8A568[7];
  qword_4F8A568[5] = 1;
  LODWORD(qword_4F8A568[6]) = 0;
  BYTE4(qword_4F8A568[6]) = 1;
  v11 = sub_C57470();
  v12 = LODWORD(qword_4F8A568[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8A568[1]) + 1 > HIDWORD(qword_4F8A568[1]) )
  {
    v19 = v11;
    sub_C8D5F0(qword_4F8A568, &qword_4F8A568[2], LODWORD(qword_4F8A568[1]) + 1LL, 8);
    v12 = LODWORD(qword_4F8A568[1]);
    v11 = v19;
  }
  *(_QWORD *)(qword_4F8A568[0] + 8 * v12) = v11;
  qword_4F8A568[9] = &unk_49D9748;
  qword_4F8A520 = &unk_49DC090;
  ++LODWORD(qword_4F8A568[1]);
  qword_4F8A568[8] = 0;
  qword_4F8A568[11] = &unk_49DC1D0;
  qword_4F8A568[10] = 0;
  qword_4F8A568[15] = nullsub_23;
  qword_4F8A568[14] = sub_984030;
  sub_C53080(&qword_4F8A520, "enable-vtable-value-profiling", 29);
  LOWORD(qword_4F8A568[10]) = 256;
  unk_4F8A548 = "If true, the virtual table address will be instrumented to know the types of a C++ pointer. The informat"
                "ion is used in indirect call promotion to do selective vtable-based comparison.";
  LOBYTE(qword_4F8A568[8]) = 0;
  unk_4F8A550 = 183;
  sub_C53130(&qword_4F8A520);
  __cxa_atexit(sub_984900, &qword_4F8A520, &qword_4A427C0);
  qword_4F8A440 = &unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8A44C = word_4F8A44C & 0x8000;
  qword_4F8A488[1] = 0x100000000LL;
  unk_4F8A448 = v13;
  unk_4F8A450 = 0;
  qword_4F8A488[0] = &qword_4F8A488[2];
  unk_4F8A458 = 0;
  unk_4F8A460 = 0;
  unk_4F8A468 = 0;
  unk_4F8A470 = 0;
  unk_4F8A478 = 0;
  unk_4F8A480 = 0;
  qword_4F8A488[3] = 0;
  qword_4F8A488[4] = &qword_4F8A488[7];
  qword_4F8A488[5] = 1;
  LODWORD(qword_4F8A488[6]) = 0;
  BYTE4(qword_4F8A488[6]) = 1;
  v14 = sub_C57470();
  v15 = LODWORD(qword_4F8A488[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8A488[1]) + 1 > HIDWORD(qword_4F8A488[1]) )
  {
    sub_C8D5F0(qword_4F8A488, &qword_4F8A488[2], LODWORD(qword_4F8A488[1]) + 1LL, 8);
    v15 = LODWORD(qword_4F8A488[1]);
  }
  *(_QWORD *)(qword_4F8A488[0] + 8 * v15) = v14;
  qword_4F8A488[9] = &unk_49D9748;
  qword_4F8A440 = &unk_49DC090;
  ++LODWORD(qword_4F8A488[1]);
  qword_4F8A488[8] = 0;
  qword_4F8A488[11] = &unk_49DC1D0;
  qword_4F8A488[10] = 0;
  qword_4F8A488[15] = nullsub_23;
  qword_4F8A488[14] = sub_984030;
  sub_C53080(&qword_4F8A440, "enable-vtable-profile-use", 25);
  LOBYTE(qword_4F8A488[8]) = 0;
  LOWORD(qword_4F8A488[10]) = 256;
  unk_4F8A468 = "If ThinLTO and WPD is enabled and this option is true, vtable profiles will be used by ICP pass for more"
                " efficient indirect call sequence. If false, type profiles won't be used.";
  unk_4F8A470 = 177;
  sub_C53130(&qword_4F8A440);
  return __cxa_atexit(sub_984900, &qword_4F8A440, &qword_4A427C0);
}
