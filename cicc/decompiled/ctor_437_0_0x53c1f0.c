// Function: ctor_437_0
// Address: 0x53c1f0
//
int __fastcall ctor_437_0(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // edx
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  int v33; // edx
  __int64 v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // ecx
  int v38; // r8d
  int v39; // r9d
  __int128 v41; // [rsp-78h] [rbp-2B0h]
  __int128 v42; // [rsp-78h] [rbp-2B0h]
  __int128 v43; // [rsp-68h] [rbp-2A0h]
  __int128 v44; // [rsp-68h] [rbp-2A0h]
  __int128 v45; // [rsp-50h] [rbp-288h]
  __int128 v46; // [rsp-50h] [rbp-288h]
  __int128 v47; // [rsp-40h] [rbp-278h]
  __int128 v48; // [rsp-40h] [rbp-278h]
  __int128 v49; // [rsp-28h] [rbp-260h]
  __int128 v50; // [rsp-28h] [rbp-260h]
  __int128 v51; // [rsp-18h] [rbp-250h]
  __int128 v52; // [rsp-18h] [rbp-250h]
  __int64 v53; // [rsp+8h] [rbp-230h]
  __int64 v54; // [rsp+8h] [rbp-230h]
  __int64 v55; // [rsp+8h] [rbp-230h]
  __int64 v56; // [rsp+8h] [rbp-230h]
  __int64 v57; // [rsp+8h] [rbp-230h]
  int v58; // [rsp+24h] [rbp-214h] BYREF
  _QWORD v59[4]; // [rsp+28h] [rbp-210h] BYREF
  __int64 v60; // [rsp+48h] [rbp-1F0h]
  const char *v61; // [rsp+50h] [rbp-1E8h]
  __int64 v62; // [rsp+58h] [rbp-1E0h]
  _QWORD v63[2]; // [rsp+68h] [rbp-1D0h] BYREF
  __int64 v64; // [rsp+78h] [rbp-1C0h]
  const char *v65; // [rsp+80h] [rbp-1B8h]
  __int64 v66; // [rsp+88h] [rbp-1B0h]
  char *v67; // [rsp+98h] [rbp-1A0h] BYREF
  __int64 v68; // [rsp+A0h] [rbp-198h]
  __int64 v69; // [rsp+A8h] [rbp-190h]
  const char *v70; // [rsp+B0h] [rbp-188h]
  __int64 v71; // [rsp+B8h] [rbp-180h]
  char *v72; // [rsp+C8h] [rbp-170h]
  __int64 v73; // [rsp+D0h] [rbp-168h]
  __int64 v74; // [rsp+D8h] [rbp-160h]
  const char *v75; // [rsp+E0h] [rbp-158h]
  __int64 v76; // [rsp+E8h] [rbp-150h]
  char *v77; // [rsp+F8h] [rbp-140h]
  __int64 v78; // [rsp+100h] [rbp-138h]
  __int64 v79; // [rsp+108h] [rbp-130h]
  const char *v80; // [rsp+110h] [rbp-128h]
  __int64 v81; // [rsp+118h] [rbp-120h]
  char *v82; // [rsp+128h] [rbp-110h]
  __int64 v83; // [rsp+130h] [rbp-108h]
  __int64 v84; // [rsp+138h] [rbp-100h]
  const char *v85; // [rsp+140h] [rbp-F8h]
  __int64 v86; // [rsp+148h] [rbp-F0h]
  _QWORD v87[2]; // [rsp+158h] [rbp-E0h] BYREF
  _BYTE v88[208]; // [rsp+168h] [rbp-D0h] BYREF

  v85 = "Export typeid resolutions to summary and globals";
  v80 = "Import typeid resolutions from summary and globals";
  v82 = "export";
  v77 = "import";
  v75 = "Do nothing";
  v83 = 6;
  LODWORD(v84) = 2;
  v86 = 48;
  LODWORD(v63[0]) = 1;
  v78 = 6;
  *((_QWORD *)&v51 + 1) = "Export typeid resolutions to summary and globals";
  LODWORD(v79) = 1;
  *(_QWORD *)&v51 = v84;
  v81 = 50;
  *((_QWORD *)&v49 + 1) = 6;
  v72 = "none";
  *(_QWORD *)&v49 = "export";
  v73 = 4;
  LODWORD(v74) = 0;
  v76 = 10;
  *((_QWORD *)&v47 + 1) = "Import typeid resolutions from summary and globals";
  *(_QWORD *)&v47 = v79;
  *((_QWORD *)&v45 + 1) = 6;
  *(_QWORD *)&v45 = "import";
  *((_QWORD *)&v43 + 1) = "Do nothing";
  *(_QWORD *)&v43 = v74;
  *((_QWORD *)&v41 + 1) = 4;
  *(_QWORD *)&v41 = "none";
  sub_22735E0(
    (unsigned int)v87,
    a2,
    (unsigned int)"export",
    (unsigned int)"Do nothing",
    a5,
    a6,
    v41,
    v43,
    10,
    v45,
    v47,
    50,
    v49,
    v51,
    48);
  v67 = "What to do with the summary when running this pass";
  v68 = 50;
  sub_2706490(&unk_4FF97C0, "wholeprogramdevirt-summary-action", &v67, v87, v63);
  if ( (_BYTE *)v87[0] != v88 )
    _libc_free(v87[0], "wholeprogramdevirt-summary-action");
  __cxa_atexit(sub_261AD80, &unk_4FF97C0, &qword_4A427C0);
  qword_4FF96C0 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9710 = 0x100000000LL;
  word_4FF96D0 = 0;
  dword_4FF96CC &= 0x8000u;
  qword_4FF96D8 = 0;
  qword_4FF96E0 = 0;
  dword_4FF96C8 = v6;
  qword_4FF96E8 = 0;
  qword_4FF96F0 = 0;
  qword_4FF96F8 = 0;
  qword_4FF9700 = 0;
  qword_4FF9708 = (__int64)&unk_4FF9718;
  qword_4FF9720 = 0;
  qword_4FF9728 = (__int64)&unk_4FF9740;
  qword_4FF9730 = 1;
  dword_4FF9738 = 0;
  byte_4FF973C = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FF9710;
  v9 = (unsigned int)qword_4FF9710 + 1LL;
  if ( v9 > HIDWORD(qword_4FF9710) )
  {
    sub_C8D5F0((char *)&unk_4FF9718 - 16, &unk_4FF9718, v9, 8);
    v8 = (unsigned int)qword_4FF9710;
  }
  *(_QWORD *)(qword_4FF9708 + 8 * v8) = v7;
  qword_4FF9748 = (__int64)&byte_4FF9758;
  qword_4FF9770 = (__int64)&byte_4FF9780;
  LODWORD(qword_4FF9710) = qword_4FF9710 + 1;
  qword_4FF97B0 = (__int64)sub_BC4D70;
  qword_4FF9768 = (__int64)&unk_49DC130;
  qword_4FF9750 = 0;
  byte_4FF9758 = 0;
  qword_4FF96C0 = (__int64)&unk_49DC010;
  qword_4FF9778 = 0;
  byte_4FF9780 = 0;
  qword_4FF9798 = (__int64)&unk_49DC350;
  byte_4FF9790 = 0;
  qword_4FF97B8 = (__int64)nullsub_92;
  sub_C53080(&qword_4FF96C0, "wholeprogramdevirt-read-summary", 31);
  qword_4FF96F0 = 64;
  qword_4FF96E8 = (__int64)"Read summary from given bitcode or YAML file before running pass";
  LOBYTE(dword_4FF96CC) = dword_4FF96CC & 0x9F | 0x20;
  sub_C53130(&qword_4FF96C0);
  __cxa_atexit(sub_BC5A40, &qword_4FF96C0, &qword_4A427C0);
  qword_4FF95C0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9610 = 0x100000000LL;
  dword_4FF95CC &= 0x8000u;
  word_4FF95D0 = 0;
  qword_4FF95D8 = 0;
  qword_4FF95E0 = 0;
  dword_4FF95C8 = v10;
  qword_4FF95E8 = 0;
  qword_4FF95F0 = 0;
  qword_4FF95F8 = 0;
  qword_4FF9600 = 0;
  qword_4FF9608 = (__int64)&unk_4FF9618;
  qword_4FF9620 = 0;
  qword_4FF9628 = (__int64)&unk_4FF9640;
  qword_4FF9630 = 1;
  dword_4FF9638 = 0;
  byte_4FF963C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FF9610;
  if ( (unsigned __int64)(unsigned int)qword_4FF9610 + 1 > HIDWORD(qword_4FF9610) )
  {
    v53 = v11;
    sub_C8D5F0((char *)&unk_4FF9618 - 16, &unk_4FF9618, (unsigned int)qword_4FF9610 + 1LL, 8);
    v12 = (unsigned int)qword_4FF9610;
    v11 = v53;
  }
  *(_QWORD *)(qword_4FF9608 + 8 * v12) = v11;
  qword_4FF9648 = (__int64)&byte_4FF9658;
  qword_4FF9670 = (__int64)&byte_4FF9680;
  LODWORD(qword_4FF9610) = qword_4FF9610 + 1;
  qword_4FF96B0 = (__int64)sub_BC4D70;
  qword_4FF9668 = (__int64)&unk_49DC130;
  qword_4FF9650 = 0;
  byte_4FF9658 = 0;
  qword_4FF95C0 = (__int64)&unk_49DC010;
  qword_4FF9678 = 0;
  byte_4FF9680 = 0;
  qword_4FF9698 = (__int64)&unk_49DC350;
  byte_4FF9690 = 0;
  qword_4FF96B8 = (__int64)nullsub_92;
  sub_C53080(&qword_4FF95C0, "wholeprogramdevirt-write-summary", 32);
  qword_4FF95F0 = 152;
  qword_4FF95E8 = (__int64)"Write summary to given bitcode or YAML file after running pass. Output file format is deduced"
                           " from extension: *.bc means writing bitcode, otherwise YAML";
  LOBYTE(dword_4FF95CC) = dword_4FF95CC & 0x9F | 0x20;
  sub_C53130(&qword_4FF95C0);
  __cxa_atexit(sub_BC5A40, &qword_4FF95C0, &qword_4A427C0);
  qword_4FF94E0 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9530 = 0x100000000LL;
  dword_4FF94EC &= 0x8000u;
  word_4FF94F0 = 0;
  qword_4FF94F8 = 0;
  qword_4FF9500 = 0;
  dword_4FF94E8 = v13;
  qword_4FF9508 = 0;
  qword_4FF9510 = 0;
  qword_4FF9518 = 0;
  qword_4FF9520 = 0;
  qword_4FF9528 = (__int64)&unk_4FF9538;
  qword_4FF9540 = 0;
  qword_4FF9548 = (__int64)&unk_4FF9560;
  qword_4FF9550 = 1;
  dword_4FF9558 = 0;
  byte_4FF955C = 1;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_4FF9530;
  v16 = (unsigned int)qword_4FF9530 + 1LL;
  if ( v16 > HIDWORD(qword_4FF9530) )
  {
    sub_C8D5F0((char *)&unk_4FF9538 - 16, &unk_4FF9538, v16, 8);
    v15 = (unsigned int)qword_4FF9530;
  }
  *(_QWORD *)(qword_4FF9528 + 8 * v15) = v14;
  LODWORD(qword_4FF9530) = qword_4FF9530 + 1;
  qword_4FF9568 = 0;
  qword_4FF9570 = (__int64)&unk_49D9728;
  qword_4FF9578 = 0;
  qword_4FF94E0 = (__int64)&unk_49DBF10;
  qword_4FF9580 = (__int64)&unk_49DC290;
  qword_4FF95A0 = (__int64)nullsub_24;
  qword_4FF9598 = (__int64)sub_984050;
  sub_C53080(&qword_4FF94E0, "wholeprogramdevirt-branch-funnel-threshold", 42);
  LODWORD(qword_4FF9568) = 10;
  BYTE4(qword_4FF9578) = 1;
  LODWORD(qword_4FF9578) = 10;
  qword_4FF9510 = 69;
  LOBYTE(dword_4FF94EC) = dword_4FF94EC & 0x9F | 0x20;
  qword_4FF9508 = (__int64)"Maximum number of call targets per call site to enable branch funnels";
  sub_C53130(&qword_4FF94E0);
  __cxa_atexit(sub_984970, &qword_4FF94E0, &qword_4A427C0);
  qword_4FF9400 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9450 = 0x100000000LL;
  dword_4FF940C &= 0x8000u;
  word_4FF9410 = 0;
  qword_4FF9418 = 0;
  qword_4FF9420 = 0;
  dword_4FF9408 = v17;
  qword_4FF9428 = 0;
  qword_4FF9430 = 0;
  qword_4FF9438 = 0;
  qword_4FF9440 = 0;
  qword_4FF9448 = (__int64)&unk_4FF9458;
  qword_4FF9460 = 0;
  qword_4FF9468 = (__int64)&unk_4FF9480;
  qword_4FF9470 = 1;
  dword_4FF9478 = 0;
  byte_4FF947C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_4FF9450;
  v20 = (unsigned int)qword_4FF9450 + 1LL;
  if ( v20 > HIDWORD(qword_4FF9450) )
  {
    sub_C8D5F0((char *)&unk_4FF9458 - 16, &unk_4FF9458, v20, 8);
    v19 = (unsigned int)qword_4FF9450;
  }
  *(_QWORD *)(qword_4FF9448 + 8 * v19) = v18;
  LODWORD(qword_4FF9450) = qword_4FF9450 + 1;
  qword_4FF9488 = 0;
  qword_4FF9490 = (__int64)&unk_49D9748;
  qword_4FF9498 = 0;
  qword_4FF9400 = (__int64)&unk_49DC090;
  qword_4FF94A0 = (__int64)&unk_49DC1D0;
  qword_4FF94C0 = (__int64)nullsub_23;
  qword_4FF94B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9400, "wholeprogramdevirt-print-index-based", 36);
  qword_4FF9430 = 43;
  LOBYTE(dword_4FF940C) = dword_4FF940C & 0x9F | 0x20;
  qword_4FF9428 = (__int64)"Print index-based devirtualization messages";
  sub_C53130(&qword_4FF9400);
  __cxa_atexit(sub_984900, &qword_4FF9400, &qword_4A427C0);
  qword_4FF9320 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9370 = 0x100000000LL;
  dword_4FF932C &= 0x8000u;
  word_4FF9330 = 0;
  qword_4FF9368 = (__int64)&unk_4FF9378;
  qword_4FF9338 = 0;
  dword_4FF9328 = v21;
  qword_4FF9340 = 0;
  qword_4FF9348 = 0;
  qword_4FF9350 = 0;
  qword_4FF9358 = 0;
  qword_4FF9360 = 0;
  qword_4FF9380 = 0;
  qword_4FF9388 = (__int64)&unk_4FF93A0;
  qword_4FF9390 = 1;
  dword_4FF9398 = 0;
  byte_4FF939C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_4FF9370;
  if ( (unsigned __int64)(unsigned int)qword_4FF9370 + 1 > HIDWORD(qword_4FF9370) )
  {
    v54 = v22;
    sub_C8D5F0((char *)&unk_4FF9378 - 16, &unk_4FF9378, (unsigned int)qword_4FF9370 + 1LL, 8);
    v23 = (unsigned int)qword_4FF9370;
    v22 = v54;
  }
  *(_QWORD *)(qword_4FF9368 + 8 * v23) = v22;
  qword_4FF93B0 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF9370) = qword_4FF9370 + 1;
  qword_4FF93A8 = 0;
  qword_4FF9320 = (__int64)&unk_49DC090;
  qword_4FF93C0 = (__int64)&unk_49DC1D0;
  qword_4FF93B8 = 0;
  qword_4FF93E0 = (__int64)nullsub_23;
  qword_4FF93D8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9320, "whole-program-visibility", 24);
  qword_4FF9350 = 31;
  LOBYTE(dword_4FF932C) = dword_4FF932C & 0x9F | 0x20;
  qword_4FF9348 = (__int64)"Enable whole program visibility";
  sub_C53130(&qword_4FF9320);
  __cxa_atexit(sub_984900, &qword_4FF9320, &qword_4A427C0);
  qword_4FF9240 = (__int64)&unk_49DC150;
  v24 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9290 = 0x100000000LL;
  dword_4FF924C &= 0x8000u;
  qword_4FF9288 = (__int64)&unk_4FF9298;
  word_4FF9250 = 0;
  qword_4FF9258 = 0;
  dword_4FF9248 = v24;
  qword_4FF9260 = 0;
  qword_4FF9268 = 0;
  qword_4FF9270 = 0;
  qword_4FF9278 = 0;
  qword_4FF9280 = 0;
  qword_4FF92A0 = 0;
  qword_4FF92A8 = (__int64)&unk_4FF92C0;
  qword_4FF92B0 = 1;
  dword_4FF92B8 = 0;
  byte_4FF92BC = 1;
  v25 = sub_C57470();
  v26 = (unsigned int)qword_4FF9290;
  if ( (unsigned __int64)(unsigned int)qword_4FF9290 + 1 > HIDWORD(qword_4FF9290) )
  {
    v55 = v25;
    sub_C8D5F0((char *)&unk_4FF9298 - 16, &unk_4FF9298, (unsigned int)qword_4FF9290 + 1LL, 8);
    v26 = (unsigned int)qword_4FF9290;
    v25 = v55;
  }
  *(_QWORD *)(qword_4FF9288 + 8 * v26) = v25;
  qword_4FF92D0 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF9290) = qword_4FF9290 + 1;
  qword_4FF92C8 = 0;
  qword_4FF9240 = (__int64)&unk_49DC090;
  qword_4FF92E0 = (__int64)&unk_49DC1D0;
  qword_4FF92D8 = 0;
  qword_4FF9300 = (__int64)nullsub_23;
  qword_4FF92F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9240, "disable-whole-program-visibility", 32);
  qword_4FF9270 = 61;
  LOBYTE(dword_4FF924C) = dword_4FF924C & 0x9F | 0x20;
  qword_4FF9268 = (__int64)"Disable whole program visibility (overrides enabling options)";
  sub_C53130(&qword_4FF9240);
  __cxa_atexit(sub_984900, &qword_4FF9240, &qword_4A427C0);
  qword_4FF9140 = (__int64)&unk_49DC150;
  v27 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9158 = 0;
  word_4FF9150 = 0;
  qword_4FF9188 = (__int64)&unk_4FF9198;
  qword_4FF9160 = 0;
  dword_4FF914C = dword_4FF914C & 0x8000 | 1;
  qword_4FF9190 = 0x100000000LL;
  dword_4FF9148 = v27;
  qword_4FF9168 = 0;
  qword_4FF9170 = 0;
  qword_4FF9178 = 0;
  qword_4FF9180 = 0;
  qword_4FF91A0 = 0;
  qword_4FF91A8 = (__int64)&unk_4FF91C0;
  qword_4FF91B0 = 1;
  dword_4FF91B8 = 0;
  byte_4FF91BC = 1;
  v28 = sub_C57470();
  v29 = (unsigned int)qword_4FF9190;
  if ( (unsigned __int64)(unsigned int)qword_4FF9190 + 1 > HIDWORD(qword_4FF9190) )
  {
    v56 = v28;
    sub_C8D5F0((char *)&unk_4FF9198 - 16, &unk_4FF9198, (unsigned int)qword_4FF9190 + 1LL, 8);
    v29 = (unsigned int)qword_4FF9190;
    v28 = v56;
  }
  *(_QWORD *)(qword_4FF9188 + 8 * v29) = v28;
  LODWORD(qword_4FF9190) = qword_4FF9190 + 1;
  qword_4FF91C8 = 0;
  qword_4FF9140 = (__int64)&unk_49DAD08;
  qword_4FF91D0 = 0;
  qword_4FF91D8 = 0;
  qword_4FF9218 = (__int64)&unk_49DC350;
  qword_4FF91E0 = 0;
  qword_4FF9238 = (__int64)nullsub_81;
  qword_4FF91E8 = 0;
  qword_4FF9230 = (__int64)sub_BB8600;
  qword_4FF91F0 = 0;
  byte_4FF91F8 = 0;
  qword_4FF9200 = 0;
  qword_4FF9208 = 0;
  qword_4FF9210 = 0;
  sub_C53080(&qword_4FF9140, "wholeprogramdevirt-skip", 23);
  BYTE1(dword_4FF914C) |= 2u;
  qword_4FF9168 = (__int64)"Prevent function(s) from being devirtualized";
  qword_4FF9170 = 44;
  LOBYTE(dword_4FF914C) = dword_4FF914C & 0x9F | 0x20;
  sub_C53130(&qword_4FF9140);
  __cxa_atexit(sub_BB89D0, &qword_4FF9140, &qword_4A427C0);
  qword_4FF9060 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF90B0 = 0x100000000LL;
  dword_4FF906C &= 0x8000u;
  word_4FF9070 = 0;
  qword_4FF90A8 = (__int64)&unk_4FF90B8;
  qword_4FF9078 = 0;
  dword_4FF9068 = v30;
  qword_4FF9080 = 0;
  qword_4FF9088 = 0;
  qword_4FF9090 = 0;
  qword_4FF9098 = 0;
  qword_4FF90A0 = 0;
  qword_4FF90C0 = 0;
  qword_4FF90C8 = (__int64)&unk_4FF90E0;
  qword_4FF90D0 = 1;
  dword_4FF90D8 = 0;
  byte_4FF90DC = 1;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_4FF90B0;
  if ( (unsigned __int64)(unsigned int)qword_4FF90B0 + 1 > HIDWORD(qword_4FF90B0) )
  {
    v57 = v31;
    sub_C8D5F0((char *)&unk_4FF90B8 - 16, &unk_4FF90B8, (unsigned int)qword_4FF90B0 + 1LL, 8);
    v32 = (unsigned int)qword_4FF90B0;
    v31 = v57;
  }
  *(_QWORD *)(qword_4FF90A8 + 8 * v32) = v31;
  qword_4FF90F0 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF90B0) = qword_4FF90B0 + 1;
  qword_4FF90E8 = 0;
  qword_4FF9060 = (__int64)&unk_49DC090;
  qword_4FF9100 = (__int64)&unk_49DC1D0;
  qword_4FF90F8 = 0;
  qword_4FF9120 = (__int64)nullsub_23;
  qword_4FF9118 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9060, "wholeprogramdevirt-keep-unreachable-function", 44);
  qword_4FF9090 = 62;
  qword_4FF9088 = (__int64)"Regard unreachable functions as possible devirtualize targets.";
  LOBYTE(qword_4FF90E8) = 1;
  LOBYTE(dword_4FF906C) = dword_4FF906C & 0x9F | 0x20;
  LOWORD(qword_4FF90F8) = 257;
  sub_C53130(&qword_4FF9060);
  __cxa_atexit(sub_984900, &qword_4FF9060, &qword_4A427C0);
  qword_4FF8F80 = (__int64)&unk_49DC150;
  v33 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF8FFC = 1;
  qword_4FF8FD0 = 0x100000000LL;
  dword_4FF8F8C &= 0x8000u;
  qword_4FF8F98 = 0;
  qword_4FF8FA0 = 0;
  qword_4FF8FA8 = 0;
  dword_4FF8F88 = v33;
  word_4FF8F90 = 0;
  qword_4FF8FB0 = 0;
  qword_4FF8FB8 = 0;
  qword_4FF8FC0 = 0;
  qword_4FF8FC8 = (__int64)&unk_4FF8FD8;
  qword_4FF8FE0 = 0;
  qword_4FF8FE8 = (__int64)&unk_4FF9000;
  qword_4FF8FF0 = 1;
  dword_4FF8FF8 = 0;
  v34 = sub_C57470();
  v35 = (unsigned int)qword_4FF8FD0;
  v36 = (unsigned int)qword_4FF8FD0 + 1LL;
  if ( v36 > HIDWORD(qword_4FF8FD0) )
  {
    sub_C8D5F0((char *)&unk_4FF8FD8 - 16, &unk_4FF8FD8, v36, 8);
    v35 = (unsigned int)qword_4FF8FD0;
  }
  *(_QWORD *)(qword_4FF8FC8 + 8 * v35) = v34;
  LODWORD(qword_4FF8FD0) = qword_4FF8FD0 + 1;
  qword_4FF9008 = 0;
  qword_4FF9010 = (__int64)&unk_49D9728;
  qword_4FF9018 = 0;
  qword_4FF8F80 = (__int64)&unk_49DBF10;
  qword_4FF9020 = (__int64)&unk_49DC290;
  qword_4FF9040 = (__int64)nullsub_24;
  qword_4FF9038 = (__int64)sub_984050;
  sub_C53080(&qword_4FF8F80, "wholeprogramdevirt-cutoff", 25);
  qword_4FF8FB0 = 54;
  qword_4FF8FA8 = (__int64)"Max number of devirtualizations for devirt module pass";
  LODWORD(qword_4FF9008) = 0;
  BYTE4(qword_4FF9018) = 1;
  LODWORD(qword_4FF9018) = 0;
  sub_C53130(&qword_4FF8F80);
  __cxa_atexit(sub_984970, &qword_4FF8F80, &qword_4A427C0);
  v70 = "Fallback to indirect when incorrect";
  v65 = "Trap when incorrect";
  v68 = 8;
  LODWORD(v69) = 2;
  v71 = 35;
  v63[0] = "trap";
  v63[1] = 4;
  *((_QWORD *)&v52 + 1) = "Fallback to indirect when incorrect";
  LODWORD(v64) = 1;
  *(_QWORD *)&v52 = v69;
  v66 = 19;
  *((_QWORD *)&v50 + 1) = 8;
  *(_QWORD *)&v50 = "fallback";
  *((_QWORD *)&v48 + 1) = "Trap when incorrect";
  *(_QWORD *)&v48 = v64;
  *((_QWORD *)&v46 + 1) = 4;
  *(_QWORD *)&v46 = "trap";
  v67 = "fallback";
  v59[3] = 4;
  LODWORD(v60) = 0;
  v61 = "No checking";
  v62 = 11;
  *((_QWORD *)&v44 + 1) = "No checking";
  *(_QWORD *)&v44 = v60;
  *((_QWORD *)&v42 + 1) = 4;
  *(_QWORD *)&v42 = "none";
  sub_22735E0(
    (unsigned int)v87,
    (unsigned int)&qword_4FF8F80,
    (unsigned int)"fallback",
    v37,
    v38,
    v39,
    v42,
    v44,
    11,
    v46,
    v48,
    19,
    v50,
    v52,
    35);
  v59[0] = "Type of checking for incorrect devirtualizations";
  v59[1] = 48;
  v58 = 1;
  sub_2706910(&unk_4FF8D20, "wholeprogramdevirt-check", &v58, v59, v87);
  if ( (_BYTE *)v87[0] != v88 )
    _libc_free(v87[0], "wholeprogramdevirt-check");
  return __cxa_atexit(sub_26F6D70, &unk_4FF8D20, &qword_4A427C0);
}
