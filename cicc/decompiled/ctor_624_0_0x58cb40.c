// Function: ctor_624_0
// Address: 0x58cb40
//
int __fastcall ctor_624_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  int v42; // edx
  __int64 v43; // r15
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rcx
  int v48; // edx
  __int64 v49; // r15
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rcx
  int v54; // edx
  __int64 v55; // r15
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rcx
  int v60; // edx
  __int64 v61; // rbx
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v65; // [rsp+8h] [rbp-38h]
  __int64 v66; // [rsp+8h] [rbp-38h]
  __int64 v67; // [rsp+8h] [rbp-38h]
  __int64 v68; // [rsp+8h] [rbp-38h]

  qword_502F9E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_502FA5C = 1;
  qword_502FA30 = 0x100000000LL;
  dword_502F9EC &= 0x8000u;
  qword_502F9F8 = 0;
  qword_502FA00 = 0;
  qword_502FA08 = 0;
  dword_502F9E8 = v4;
  word_502F9F0 = 0;
  qword_502FA10 = 0;
  qword_502FA18 = 0;
  qword_502FA20 = 0;
  qword_502FA28 = (__int64)&unk_502FA38;
  qword_502FA40 = 0;
  qword_502FA48 = (__int64)&unk_502FA60;
  qword_502FA50 = 1;
  dword_502FA58 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502FA30;
  v7 = (unsigned int)qword_502FA30 + 1LL;
  if ( v7 > HIDWORD(qword_502FA30) )
  {
    sub_C8D5F0((char *)&unk_502FA38 - 16, &unk_502FA38, v7, 8);
    v6 = (unsigned int)qword_502FA30;
  }
  *(_QWORD *)(qword_502FA28 + 8 * v6) = v5;
  LODWORD(qword_502FA30) = qword_502FA30 + 1;
  qword_502FA68 = 0;
  qword_502FA70 = (__int64)&unk_49D9748;
  qword_502FA78 = 0;
  qword_502F9E0 = (__int64)&unk_49DC090;
  qword_502FA80 = (__int64)&unk_49DC1D0;
  qword_502FAA0 = (__int64)nullsub_23;
  qword_502FA98 = (__int64)sub_984030;
  sub_C53080(&qword_502F9E0, "inline-remark-attribute", 23);
  LOBYTE(qword_502FA68) = 0;
  LOWORD(qword_502FA78) = 256;
  qword_502FA10 = 101;
  LOBYTE(dword_502F9EC) = dword_502F9EC & 0x9F | 0x20;
  qword_502FA08 = (__int64)"Enable adding inline-remark attribute to callsites processed by inliner but decided to be not inlined";
  sub_C53130(&qword_502F9E0);
  __cxa_atexit(sub_984900, &qword_502F9E0, &qword_4A427C0);
  qword_502F900 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502F9E0, v8, v9), 1u);
  dword_502F90C &= 0x8000u;
  word_502F910 = 0;
  qword_502F950 = 0x100000000LL;
  qword_502F918 = 0;
  qword_502F920 = 0;
  qword_502F928 = 0;
  dword_502F908 = v10;
  qword_502F930 = 0;
  qword_502F938 = 0;
  qword_502F940 = 0;
  qword_502F948 = (__int64)&unk_502F958;
  qword_502F960 = 0;
  qword_502F968 = (__int64)&unk_502F980;
  qword_502F970 = 1;
  dword_502F978 = 0;
  byte_502F97C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502F950;
  v13 = (unsigned int)qword_502F950 + 1LL;
  if ( v13 > HIDWORD(qword_502F950) )
  {
    sub_C8D5F0((char *)&unk_502F958 - 16, &unk_502F958, v13, 8);
    v12 = (unsigned int)qword_502F950;
  }
  *(_QWORD *)(qword_502F948 + 8 * v12) = v11;
  LODWORD(qword_502F950) = qword_502F950 + 1;
  qword_502F988 = 0;
  qword_502F990 = (__int64)&unk_49D9748;
  qword_502F998 = 0;
  qword_502F900 = (__int64)&unk_49DC090;
  qword_502F9A0 = (__int64)&unk_49DC1D0;
  qword_502F9C0 = (__int64)nullsub_23;
  qword_502F9B8 = (__int64)sub_984030;
  sub_C53080(&qword_502F900, "inline-deferral", 15);
  LOBYTE(qword_502F988) = 0;
  LOWORD(qword_502F998) = 256;
  qword_502F930 = 24;
  LOBYTE(dword_502F90C) = dword_502F90C & 0x9F | 0x20;
  qword_502F928 = (__int64)"Enable deferred inlining";
  sub_C53130(&qword_502F900);
  __cxa_atexit(sub_984900, &qword_502F900, &qword_4A427C0);
  qword_502F820 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502F900, v14, v15), 1u);
  dword_502F82C &= 0x8000u;
  word_502F830 = 0;
  qword_502F870 = 0x100000000LL;
  qword_502F838 = 0;
  qword_502F840 = 0;
  qword_502F848 = 0;
  dword_502F828 = v16;
  qword_502F850 = 0;
  qword_502F858 = 0;
  qword_502F860 = 0;
  qword_502F868 = (__int64)&unk_502F878;
  qword_502F880 = 0;
  qword_502F888 = (__int64)&unk_502F8A0;
  qword_502F890 = 1;
  dword_502F898 = 0;
  byte_502F89C = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_502F870;
  v19 = (unsigned int)qword_502F870 + 1LL;
  if ( v19 > HIDWORD(qword_502F870) )
  {
    sub_C8D5F0((char *)&unk_502F878 - 16, &unk_502F878, v19, 8);
    v18 = (unsigned int)qword_502F870;
  }
  *(_QWORD *)(qword_502F868 + 8 * v18) = v17;
  qword_502F8B0 = (__int64)&unk_49DA090;
  qword_502F820 = (__int64)&unk_49DBF90;
  LODWORD(qword_502F870) = qword_502F870 + 1;
  qword_502F8A8 = 0;
  qword_502F8C0 = (__int64)&unk_49DC230;
  qword_502F8B8 = 0;
  qword_502F8E0 = (__int64)nullsub_58;
  qword_502F8D8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502F820, "inline-deferral-scale", 21);
  qword_502F850 = 42;
  qword_502F848 = (__int64)"Scale to limit the cost of inline deferral";
  LODWORD(qword_502F8A8) = 2;
  BYTE4(qword_502F8B8) = 1;
  LODWORD(qword_502F8B8) = 2;
  LOBYTE(dword_502F82C) = dword_502F82C & 0x9F | 0x20;
  sub_C53130(&qword_502F820);
  __cxa_atexit(sub_B2B680, &qword_502F820, &qword_4A427C0);
  qword_502F740 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502F820, v20, v21), 1u);
  dword_502F74C &= 0x8000u;
  word_502F750 = 0;
  qword_502F790 = 0x100000000LL;
  qword_502F788 = (__int64)&unk_502F798;
  qword_502F758 = 0;
  qword_502F760 = 0;
  dword_502F748 = v22;
  qword_502F768 = 0;
  qword_502F770 = 0;
  qword_502F778 = 0;
  qword_502F780 = 0;
  qword_502F7A0 = 0;
  qword_502F7A8 = (__int64)&unk_502F7C0;
  qword_502F7B0 = 1;
  dword_502F7B8 = 0;
  byte_502F7BC = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_502F790;
  if ( (unsigned __int64)(unsigned int)qword_502F790 + 1 > HIDWORD(qword_502F790) )
  {
    v65 = v23;
    sub_C8D5F0((char *)&unk_502F798 - 16, &unk_502F798, (unsigned int)qword_502F790 + 1LL, 8);
    v24 = (unsigned int)qword_502F790;
    v23 = v65;
  }
  *(_QWORD *)(qword_502F788 + 8 * v24) = v23;
  LODWORD(qword_502F790) = qword_502F790 + 1;
  qword_502F7C8 = 0;
  qword_502F7D0 = (__int64)&unk_49D9748;
  qword_502F7D8 = 0;
  qword_502F740 = (__int64)&unk_49DC090;
  qword_502F7E0 = (__int64)&unk_49DC1D0;
  qword_502F800 = (__int64)nullsub_23;
  qword_502F7F8 = (__int64)sub_984030;
  sub_C53080(&qword_502F740, "annotate-inline-phase", 21);
  LOWORD(qword_502F7D8) = 256;
  LOBYTE(qword_502F7C8) = 0;
  qword_502F770 = 71;
  LOBYTE(dword_502F74C) = dword_502F74C & 0x9F | 0x20;
  qword_502F768 = (__int64)"If true, annotate inline advisor remarks with LTO and pass information.";
  sub_C53130(&qword_502F740);
  __cxa_atexit(sub_984900, &qword_502F740, &qword_4A427C0);
  qword_502F660 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502F740, v25, v26), 1u);
  qword_502F6B0 = 0x100000000LL;
  dword_502F66C &= 0x8000u;
  qword_502F6A8 = (__int64)&unk_502F6B8;
  word_502F670 = 0;
  qword_502F678 = 0;
  dword_502F668 = v27;
  qword_502F680 = 0;
  qword_502F688 = 0;
  qword_502F690 = 0;
  qword_502F698 = 0;
  qword_502F6A0 = 0;
  qword_502F6C0 = 0;
  qword_502F6C8 = (__int64)&unk_502F6E0;
  qword_502F6D0 = 1;
  dword_502F6D8 = 0;
  byte_502F6DC = 1;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_502F6B0;
  if ( (unsigned __int64)(unsigned int)qword_502F6B0 + 1 > HIDWORD(qword_502F6B0) )
  {
    v66 = v28;
    sub_C8D5F0((char *)&unk_502F6B8 - 16, &unk_502F6B8, (unsigned int)qword_502F6B0 + 1LL, 8);
    v29 = (unsigned int)qword_502F6B0;
    v28 = v66;
  }
  *(_QWORD *)(qword_502F6A8 + 8 * v29) = v28;
  LODWORD(qword_502F6B0) = qword_502F6B0 + 1;
  qword_502F6E8 = 0;
  qword_502F6F0 = (__int64)&unk_49D9748;
  qword_502F6F8 = 0;
  qword_502F660 = (__int64)&unk_49DC090;
  qword_502F700 = (__int64)&unk_49DC1D0;
  qword_502F720 = (__int64)nullsub_23;
  qword_502F718 = (__int64)sub_984030;
  sub_C53080(&qword_502F660, "inline-use-budget", 17);
  qword_502F690 = 59;
  LOBYTE(dword_502F66C) = dword_502F66C & 0x9F | 0x20;
  qword_502F688 = (__int64)"Control whether or not to use NV inlining budget heuristics";
  sub_C53130(&qword_502F660);
  __cxa_atexit(sub_984900, &qword_502F660, &qword_4A427C0);
  qword_502F580 = (__int64)&unk_49DC150;
  v32 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502F660, v30, v31), 1u);
  qword_502F5D0 = 0x100000000LL;
  dword_502F58C &= 0x8000u;
  qword_502F5C8 = (__int64)&unk_502F5D8;
  word_502F590 = 0;
  qword_502F598 = 0;
  dword_502F588 = v32;
  qword_502F5A0 = 0;
  qword_502F5A8 = 0;
  qword_502F5B0 = 0;
  qword_502F5B8 = 0;
  qword_502F5C0 = 0;
  qword_502F5E0 = 0;
  qword_502F5E8 = (__int64)&unk_502F600;
  qword_502F5F0 = 1;
  dword_502F5F8 = 0;
  byte_502F5FC = 1;
  v33 = sub_C57470();
  v34 = (unsigned int)qword_502F5D0;
  if ( (unsigned __int64)(unsigned int)qword_502F5D0 + 1 > HIDWORD(qword_502F5D0) )
  {
    v67 = v33;
    sub_C8D5F0((char *)&unk_502F5D8 - 16, &unk_502F5D8, (unsigned int)qword_502F5D0 + 1LL, 8);
    v34 = (unsigned int)qword_502F5D0;
    v33 = v67;
  }
  *(_QWORD *)(qword_502F5C8 + 8 * v34) = v33;
  qword_502F610 = (__int64)&unk_49DA090;
  qword_502F580 = (__int64)&unk_49DBF90;
  LODWORD(qword_502F5D0) = qword_502F5D0 + 1;
  qword_502F608 = 0;
  qword_502F620 = (__int64)&unk_49DC230;
  qword_502F618 = 0;
  qword_502F640 = (__int64)nullsub_58;
  qword_502F638 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502F580, "inline-total-budget", 19);
  LODWORD(qword_502F608) = 500000;
  BYTE4(qword_502F618) = 1;
  LODWORD(qword_502F618) = 500000;
  qword_502F5B0 = 21;
  LOBYTE(dword_502F58C) = dword_502F58C & 0x9F | 0x20;
  qword_502F5A8 = (__int64)"Total inlining budget";
  sub_C53130(&qword_502F580);
  __cxa_atexit(sub_B2B680, &qword_502F580, &qword_4A427C0);
  qword_502F4A0 = (__int64)&unk_49DC150;
  v37 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502F580, v35, v36), 1u);
  qword_502F4F0 = 0x100000000LL;
  dword_502F4AC &= 0x8000u;
  word_502F4B0 = 0;
  qword_502F4E8 = (__int64)&unk_502F4F8;
  qword_502F4B8 = 0;
  dword_502F4A8 = v37;
  qword_502F4C0 = 0;
  qword_502F4C8 = 0;
  qword_502F4D0 = 0;
  qword_502F4D8 = 0;
  qword_502F4E0 = 0;
  qword_502F500 = 0;
  qword_502F508 = (__int64)&unk_502F520;
  qword_502F510 = 1;
  dword_502F518 = 0;
  byte_502F51C = 1;
  v38 = sub_C57470();
  v39 = (unsigned int)qword_502F4F0;
  if ( (unsigned __int64)(unsigned int)qword_502F4F0 + 1 > HIDWORD(qword_502F4F0) )
  {
    v68 = v38;
    sub_C8D5F0((char *)&unk_502F4F8 - 16, &unk_502F4F8, (unsigned int)qword_502F4F0 + 1LL, 8);
    v39 = (unsigned int)qword_502F4F0;
    v38 = v68;
  }
  *(_QWORD *)(qword_502F4E8 + 8 * v39) = v38;
  LODWORD(qword_502F4F0) = qword_502F4F0 + 1;
  qword_502F528 = 0;
  qword_502F530 = (__int64)&unk_49D9748;
  qword_502F538 = 0;
  qword_502F4A0 = (__int64)&unk_49DC090;
  qword_502F540 = (__int64)&unk_49DC1D0;
  qword_502F560 = (__int64)nullsub_23;
  qword_502F558 = (__int64)sub_984030;
  sub_C53080(&qword_502F4A0, "inline-switchctrl", 17);
  LOWORD(qword_502F538) = 257;
  LOBYTE(qword_502F528) = 1;
  qword_502F4D0 = 52;
  LOBYTE(dword_502F4AC) = dword_502F4AC & 0x9F | 0x20;
  qword_502F4C8 = (__int64)"Control to tuning inline heuristic based on switches";
  sub_C53130(&qword_502F4A0);
  __cxa_atexit(sub_984900, &qword_502F4A0, &qword_4A427C0);
  qword_502F3C0 = (__int64)&unk_49DC150;
  v42 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502F4A0, v40, v41), 1u);
  qword_502F410 = 0x100000000LL;
  dword_502F3CC &= 0x8000u;
  qword_502F408 = (__int64)&unk_502F418;
  word_502F3D0 = 0;
  qword_502F3D8 = 0;
  dword_502F3C8 = v42;
  qword_502F3E0 = 0;
  qword_502F3E8 = 0;
  qword_502F3F0 = 0;
  qword_502F3F8 = 0;
  qword_502F400 = 0;
  qword_502F420 = 0;
  qword_502F428 = (__int64)&unk_502F440;
  qword_502F430 = 1;
  dword_502F438 = 0;
  byte_502F43C = 1;
  v43 = sub_C57470();
  v44 = (unsigned int)qword_502F410;
  v45 = (unsigned int)qword_502F410 + 1LL;
  if ( v45 > HIDWORD(qword_502F410) )
  {
    sub_C8D5F0((char *)&unk_502F418 - 16, &unk_502F418, v45, 8);
    v44 = (unsigned int)qword_502F410;
  }
  *(_QWORD *)(qword_502F408 + 8 * v44) = v43;
  qword_502F450 = (__int64)&unk_49DA090;
  qword_502F3C0 = (__int64)&unk_49DBF90;
  LODWORD(qword_502F410) = qword_502F410 + 1;
  qword_502F448 = 0;
  qword_502F460 = (__int64)&unk_49DC230;
  qword_502F458 = 0;
  qword_502F480 = (__int64)nullsub_58;
  qword_502F478 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502F3C0, "inline-numswitchfunc", 20);
  LODWORD(qword_502F448) = 5;
  BYTE4(qword_502F458) = 1;
  LODWORD(qword_502F458) = 5;
  qword_502F3F0 = 47;
  LOBYTE(dword_502F3CC) = dword_502F3CC & 0x98 | 0x21;
  qword_502F3E8 = (__int64)"Control of inline heuristic on switch functions";
  sub_C53130(&qword_502F3C0);
  __cxa_atexit(sub_B2B680, &qword_502F3C0, &qword_4A427C0);
  qword_502F2E0 = (__int64)&unk_49DC150;
  v48 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502F3C0, v46, v47), 1u);
  qword_502F330 = 0x100000000LL;
  dword_502F2EC &= 0x8000u;
  word_502F2F0 = 0;
  qword_502F328 = (__int64)&unk_502F338;
  qword_502F2F8 = 0;
  dword_502F2E8 = v48;
  qword_502F300 = 0;
  qword_502F308 = 0;
  qword_502F310 = 0;
  qword_502F318 = 0;
  qword_502F320 = 0;
  qword_502F340 = 0;
  qword_502F348 = (__int64)&unk_502F360;
  qword_502F350 = 1;
  dword_502F358 = 0;
  byte_502F35C = 1;
  v49 = sub_C57470();
  v50 = (unsigned int)qword_502F330;
  v51 = (unsigned int)qword_502F330 + 1LL;
  if ( v51 > HIDWORD(qword_502F330) )
  {
    sub_C8D5F0((char *)&unk_502F338 - 16, &unk_502F338, v51, 8);
    v50 = (unsigned int)qword_502F330;
  }
  *(_QWORD *)(qword_502F328 + 8 * v50) = v49;
  qword_502F370 = (__int64)&unk_49DA090;
  qword_502F2E0 = (__int64)&unk_49DBF90;
  LODWORD(qword_502F330) = qword_502F330 + 1;
  qword_502F368 = 0;
  qword_502F380 = (__int64)&unk_49DC230;
  qword_502F378 = 0;
  qword_502F3A0 = (__int64)nullsub_58;
  qword_502F398 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502F2E0, "inline-maxswitchcases", 21);
  LODWORD(qword_502F368) = 71;
  BYTE4(qword_502F378) = 1;
  LODWORD(qword_502F378) = 71;
  qword_502F310 = 43;
  LOBYTE(dword_502F2EC) = dword_502F2EC & 0x98 | 0x21;
  qword_502F308 = (__int64)"Control of inline heuristic on switch cases";
  sub_C53130(&qword_502F2E0);
  __cxa_atexit(sub_B2B680, &qword_502F2E0, &qword_4A427C0);
  qword_502F200 = (__int64)&unk_49DC150;
  v54 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502F2E0, v52, v53), 1u);
  byte_502F27C = 1;
  qword_502F250 = 0x100000000LL;
  dword_502F20C &= 0x8000u;
  qword_502F248 = (__int64)&unk_502F258;
  qword_502F218 = 0;
  qword_502F220 = 0;
  dword_502F208 = v54;
  word_502F210 = 0;
  qword_502F228 = 0;
  qword_502F230 = 0;
  qword_502F238 = 0;
  qword_502F240 = 0;
  qword_502F260 = 0;
  qword_502F268 = (__int64)&unk_502F280;
  qword_502F270 = 1;
  dword_502F278 = 0;
  v55 = sub_C57470();
  v56 = (unsigned int)qword_502F250;
  v57 = (unsigned int)qword_502F250 + 1LL;
  if ( v57 > HIDWORD(qword_502F250) )
  {
    sub_C8D5F0((char *)&unk_502F258 - 16, &unk_502F258, v57, 8);
    v56 = (unsigned int)qword_502F250;
  }
  *(_QWORD *)(qword_502F248 + 8 * v56) = v55;
  qword_502F290 = (__int64)&unk_49DA090;
  qword_502F200 = (__int64)&unk_49DBF90;
  LODWORD(qword_502F250) = qword_502F250 + 1;
  qword_502F288 = 0;
  qword_502F2A0 = (__int64)&unk_49DC230;
  qword_502F298 = 0;
  qword_502F2C0 = (__int64)nullsub_58;
  qword_502F2B8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502F200, "inline-adj-budget1", 18);
  LODWORD(qword_502F288) = 1;
  BYTE4(qword_502F298) = 1;
  LODWORD(qword_502F298) = 1;
  qword_502F230 = 65;
  LOBYTE(dword_502F20C) = dword_502F20C & 0x98 | 0x21;
  qword_502F228 = (__int64)"Adjusted control the amount of inlining to perform to each caller";
  sub_C53130(&qword_502F200);
  __cxa_atexit(sub_B2B680, &qword_502F200, &qword_4A427C0);
  qword_502F120 = (__int64)&unk_49DC150;
  v60 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_502F200, v58, v59), 1u);
  dword_502F12C &= 0x8000u;
  word_502F130 = 0;
  qword_502F170 = 0x100000000LL;
  qword_502F138 = 0;
  qword_502F140 = 0;
  qword_502F148 = 0;
  dword_502F128 = v60;
  qword_502F150 = 0;
  qword_502F158 = 0;
  qword_502F160 = 0;
  qword_502F168 = (__int64)&unk_502F178;
  qword_502F180 = 0;
  qword_502F188 = (__int64)&unk_502F1A0;
  qword_502F190 = 1;
  dword_502F198 = 0;
  byte_502F19C = 1;
  v61 = sub_C57470();
  v62 = (unsigned int)qword_502F170;
  v63 = (unsigned int)qword_502F170 + 1LL;
  if ( v63 > HIDWORD(qword_502F170) )
  {
    sub_C8D5F0((char *)&unk_502F178 - 16, &unk_502F178, v63, 8);
    v62 = (unsigned int)qword_502F170;
  }
  *(_QWORD *)(qword_502F168 + 8 * v62) = v61;
  qword_502F1B0 = (__int64)&unk_49DA090;
  qword_502F120 = (__int64)&unk_49DBF90;
  LODWORD(qword_502F170) = qword_502F170 + 1;
  qword_502F1A8 = 0;
  qword_502F1C0 = (__int64)&unk_49DC230;
  qword_502F1B8 = 0;
  qword_502F1E0 = (__int64)nullsub_58;
  qword_502F1D8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_502F120, "inline-budget", 13);
  LODWORD(qword_502F1A8) = 20000;
  BYTE4(qword_502F1B8) = 1;
  LODWORD(qword_502F1B8) = 20000;
  qword_502F150 = 74;
  LOBYTE(dword_502F12C) = dword_502F12C & 0x98 | 0x21;
  qword_502F148 = (__int64)"Control the amount of inlining to perform to each caller (default = 20000)";
  sub_C53130(&qword_502F120);
  return __cxa_atexit(sub_B2B680, &qword_502F120, &qword_4A427C0);
}
