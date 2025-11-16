// Function: sub_3149320
// Address: 0x3149320
//
__int64 *sub_3149320()
{
  int v1; // edx
  __int64 *v2; // rbx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  int v7; // edx
  __int64 *v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // edx
  __int64 *v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // edx
  __int64 *v20; // rbx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  int v25; // edx
  __int64 *v26; // rbx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  int v31; // edx
  __int64 *v32; // rbx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // edx
  __int64 *v38; // rbx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  int v43; // edx
  __int64 *v44; // rbx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  int v49; // edx
  __int64 *v50; // rbx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  int v55; // edx
  __int64 *v56; // rbx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  int v61; // edx
  __int64 *v62; // rbx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  const char **v67; // rsi
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 *v72; // rax
  int v73; // edx
  __int64 *v74; // rbx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  int v79; // edx
  __int64 *v80; // rbx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  int v85; // edx
  __int64 *v86; // rbx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rax
  unsigned __int64 v90; // rdx
  int v91; // edx
  __int64 *v92; // rbx
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 v95; // rax
  unsigned __int64 v96; // rdx
  int v97; // edx
  __int64 *v98; // rbx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rax
  unsigned __int64 v102; // rdx
  int v103; // edx
  __int64 *v104; // rbx
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // rax
  unsigned __int64 v108; // rdx
  int v109; // edx
  __int64 *v110; // rbx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rax
  unsigned __int64 v114; // rdx
  int v115; // edx
  __int64 *v116; // rbx
  __int64 v117; // r8
  __int64 v118; // r9
  __int64 v119; // rax
  unsigned __int64 v120; // rdx
  int v121; // [rsp+4h] [rbp-DCh] BYREF
  int *v122; // [rsp+8h] [rbp-D8h] BYREF
  const char *v123[2]; // [rsp+10h] [rbp-D0h] BYREF
  const char *v124; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v125; // [rsp+28h] [rbp-B8h]
  _QWORD v126[2]; // [rsp+30h] [rbp-B0h] BYREF
  int v127; // [rsp+40h] [rbp-A0h]
  const char *v128; // [rsp+48h] [rbp-98h]
  __int64 v129; // [rsp+50h] [rbp-90h]
  const char *v130; // [rsp+58h] [rbp-88h]
  __int64 v131; // [rsp+60h] [rbp-80h]
  int v132; // [rsp+68h] [rbp-78h]
  const char *v133; // [rsp+70h] [rbp-70h]
  __int64 v134; // [rsp+78h] [rbp-68h]
  char *v135; // [rsp+80h] [rbp-60h]
  __int64 v136; // [rsp+88h] [rbp-58h]
  int v137; // [rsp+90h] [rbp-50h]
  const char *v138; // [rsp+98h] [rbp-48h]
  __int64 v139; // [rsp+A0h] [rbp-40h]

  if ( !byte_5033D68 && (unsigned int)sub_2207590((__int64)&byte_5033D68) )
  {
    qword_5033D80 = (__int64)&unk_49DC150;
    v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033DD0 = 0x100000000LL;
    dword_5033D8C &= 0x8000u;
    word_5033D90 = 0;
    qword_5033D98 = 0;
    qword_5033DA0 = 0;
    dword_5033D88 = v1;
    qword_5033DA8 = 0;
    qword_5033DB0 = 0;
    qword_5033DB8 = 0;
    qword_5033DC0 = 0;
    qword_5033DC8 = (__int64)&unk_5033DD8;
    qword_5033DE0 = 0;
    qword_5033DE8 = (__int64)&unk_5033E00;
    qword_5033DF0 = 1;
    dword_5033DF8 = 0;
    byte_5033DFC = 1;
    v2 = sub_C57470();
    v5 = (unsigned int)qword_5033DD0;
    v6 = (unsigned int)qword_5033DD0 + 1LL;
    if ( v6 > HIDWORD(qword_5033DD0) )
    {
      sub_C8D5F0((__int64)&unk_5033DD8 - 16, &unk_5033DD8, v6, 8u, v3, v4);
      v5 = (unsigned int)qword_5033DD0;
    }
    *(_QWORD *)(qword_5033DC8 + 8 * v5) = v2;
    LODWORD(qword_5033DD0) = qword_5033DD0 + 1;
    qword_5033E08 = 0;
    qword_5033E10 = (__int64)&unk_49D9748;
    qword_5033E18 = 0;
    qword_5033D80 = (__int64)&unk_49DC090;
    qword_5033E20 = (__int64)&unk_49DC1D0;
    qword_5033E40 = (__int64)nullsub_23;
    qword_5033E38 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5033D80, (__int64)"mc-relax-all", 12);
    qword_5033DB0 = 72;
    qword_5033DA8 = (__int64)"When used with filetype=obj, relax all fixups in the emitted object file";
    sub_C53130((__int64)&qword_5033D80);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5033D80, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033D68);
  }
  qword_5033ED8 = (__int64)&qword_5033D80;
  if ( !byte_5033C88 && (unsigned int)sub_2207590((__int64)&byte_5033C88) )
  {
    qword_5033CA0 = (__int64)&unk_49DC150;
    v115 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033CF0 = 0x100000000LL;
    dword_5033CAC &= 0x8000u;
    word_5033CB0 = 0;
    qword_5033CB8 = 0;
    qword_5033CC0 = 0;
    dword_5033CA8 = v115;
    qword_5033CC8 = 0;
    qword_5033CD0 = 0;
    qword_5033CD8 = 0;
    qword_5033CE0 = 0;
    qword_5033CE8 = (__int64)&unk_5033CF8;
    qword_5033D00 = 0;
    qword_5033D08 = (__int64)&unk_5033D20;
    qword_5033D10 = 1;
    dword_5033D18 = 0;
    byte_5033D1C = 1;
    v116 = sub_C57470();
    v119 = (unsigned int)qword_5033CF0;
    v120 = (unsigned int)qword_5033CF0 + 1LL;
    if ( v120 > HIDWORD(qword_5033CF0) )
    {
      sub_C8D5F0((__int64)&unk_5033CF8 - 16, &unk_5033CF8, v120, 8u, v117, v118);
      v119 = (unsigned int)qword_5033CF0;
    }
    *(_QWORD *)(qword_5033CE8 + 8 * v119) = v116;
    LODWORD(qword_5033CF0) = qword_5033CF0 + 1;
    qword_5033D28 = 0;
    qword_5033D30 = (__int64)&unk_49D9748;
    qword_5033D38 = 0;
    qword_5033CA0 = (__int64)&unk_49DC090;
    qword_5033D40 = (__int64)&unk_49DC1D0;
    qword_5033D60 = (__int64)nullsub_23;
    qword_5033D58 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5033CA0, (__int64)"incremental-linker-compatible", 29);
    qword_5033CD0 = 93;
    qword_5033CC8 = (__int64)"When used with filetype=obj, emit an object file which can be used with an incremental linker";
    sub_C53130((__int64)&qword_5033CA0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5033CA0, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033C88);
  }
  qword_5033ED0 = (__int64)&qword_5033CA0;
  if ( !byte_5033BA8 && (unsigned int)sub_2207590((__int64)&byte_5033BA8) )
  {
    qword_5033BC0 = (__int64)&unk_49DC150;
    v109 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033C10 = 0x100000000LL;
    dword_5033BCC &= 0x8000u;
    word_5033BD0 = 0;
    qword_5033BD8 = 0;
    qword_5033BE0 = 0;
    dword_5033BC8 = v109;
    qword_5033BE8 = 0;
    qword_5033BF0 = 0;
    qword_5033BF8 = 0;
    qword_5033C00 = 0;
    qword_5033C08 = (__int64)&unk_5033C18;
    qword_5033C20 = 0;
    qword_5033C28 = (__int64)&unk_5033C40;
    qword_5033C30 = 1;
    dword_5033C38 = 0;
    byte_5033C3C = 1;
    v110 = sub_C57470();
    v113 = (unsigned int)qword_5033C10;
    v114 = (unsigned int)qword_5033C10 + 1LL;
    if ( v114 > HIDWORD(qword_5033C10) )
    {
      sub_C8D5F0((__int64)&unk_5033C18 - 16, &unk_5033C18, v114, 8u, v111, v112);
      v113 = (unsigned int)qword_5033C10;
    }
    *(_QWORD *)(qword_5033C08 + 8 * v113) = v110;
    LODWORD(qword_5033C10) = qword_5033C10 + 1;
    qword_5033C48 = 0;
    qword_5033C50 = (__int64)&unk_49D9748;
    qword_5033C58 = 0;
    qword_5033BC0 = (__int64)&unk_49DC090;
    qword_5033C60 = (__int64)&unk_49DC1D0;
    qword_5033C80 = (__int64)nullsub_23;
    qword_5033C78 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5033BC0, (__int64)"fdpic", 5);
    qword_5033BF0 = 17;
    qword_5033BE8 = (__int64)"Use the FDPIC ABI";
    sub_C53130((__int64)&qword_5033BC0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5033BC0, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033BA8);
  }
  qword_5033EC8 = (__int64)&qword_5033BC0;
  if ( !byte_5033AC8 && (unsigned int)sub_2207590((__int64)&byte_5033AC8) )
  {
    qword_5033AE0 = (__int64)&unk_49DC150;
    v103 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033B30 = 0x100000000LL;
    word_5033AF0 = 0;
    dword_5033AEC &= 0x8000u;
    qword_5033AF8 = 0;
    qword_5033B00 = 0;
    dword_5033AE8 = v103;
    qword_5033B08 = 0;
    qword_5033B10 = 0;
    qword_5033B18 = 0;
    qword_5033B20 = 0;
    qword_5033B28 = (__int64)&unk_5033B38;
    qword_5033B40 = 0;
    qword_5033B48 = (__int64)&unk_5033B60;
    qword_5033B50 = 1;
    dword_5033B58 = 0;
    byte_5033B5C = 1;
    v104 = sub_C57470();
    v107 = (unsigned int)qword_5033B30;
    v108 = (unsigned int)qword_5033B30 + 1LL;
    if ( v108 > HIDWORD(qword_5033B30) )
    {
      sub_C8D5F0((__int64)&unk_5033B38 - 16, &unk_5033B38, v108, 8u, v105, v106);
      v107 = (unsigned int)qword_5033B30;
    }
    *(_QWORD *)(qword_5033B28 + 8 * v107) = v104;
    LODWORD(qword_5033B30) = qword_5033B30 + 1;
    qword_5033B68 = 0;
    qword_5033B70 = (__int64)&unk_49DA090;
    qword_5033B78 = 0;
    qword_5033AE0 = (__int64)&unk_49DBF90;
    qword_5033B80 = (__int64)&unk_49DC230;
    qword_5033BA0 = (__int64)nullsub_58;
    qword_5033B98 = (__int64)sub_B2B5F0;
    sub_C53080((__int64)&qword_5033AE0, (__int64)"dwarf-version", 13);
    qword_5033B10 = 13;
    qword_5033B08 = (__int64)"Dwarf version";
    LODWORD(qword_5033B68) = 0;
    BYTE4(qword_5033B78) = 1;
    LODWORD(qword_5033B78) = 0;
    sub_C53130((__int64)&qword_5033AE0);
    __cxa_atexit((void (*)(void *))sub_B2B680, &qword_5033AE0, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033AC8);
  }
  qword_5033EC0 = (__int64)&qword_5033AE0;
  if ( !byte_50339E0 && (unsigned int)sub_2207590((__int64)&byte_50339E0) )
  {
    qword_5033A00 = (__int64)&unk_49DC150;
    v97 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    byte_5033A7C = 1;
    qword_5033A50 = 0x100000000LL;
    dword_5033A0C &= 0x8000u;
    qword_5033A18 = 0;
    qword_5033A20 = 0;
    qword_5033A28 = 0;
    dword_5033A08 = v97;
    word_5033A10 = 0;
    qword_5033A30 = 0;
    qword_5033A38 = 0;
    qword_5033A40 = 0;
    qword_5033A48 = (__int64)&unk_5033A58;
    qword_5033A60 = 0;
    qword_5033A68 = (__int64)&unk_5033A80;
    qword_5033A70 = 1;
    dword_5033A78 = 0;
    v98 = sub_C57470();
    v101 = (unsigned int)qword_5033A50;
    v102 = (unsigned int)qword_5033A50 + 1LL;
    if ( v102 > HIDWORD(qword_5033A50) )
    {
      sub_C8D5F0((__int64)&unk_5033A58 - 16, &unk_5033A58, v102, 8u, v99, v100);
      v101 = (unsigned int)qword_5033A50;
    }
    *(_QWORD *)(qword_5033A48 + 8 * v101) = v98;
    LODWORD(qword_5033A50) = qword_5033A50 + 1;
    qword_5033A88 = 0;
    qword_5033A90 = (__int64)&unk_49D9748;
    qword_5033A98 = 0;
    qword_5033A00 = (__int64)&unk_49DC090;
    qword_5033AA0 = (__int64)&unk_49DC1D0;
    qword_5033AC0 = (__int64)nullsub_23;
    qword_5033AB8 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5033A00, (__int64)"dwarf64", 7);
    qword_5033A30 = 50;
    qword_5033A28 = (__int64)"Generate debugging info in the 64-bit DWARF format";
    sub_C53130((__int64)&qword_5033A00);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5033A00, &qword_4A427C0);
    sub_2207640((__int64)&byte_50339E0);
  }
  qword_5033EB8 = (__int64)&qword_5033A00;
  if ( !byte_5033768 )
  {
    if ( (unsigned int)sub_2207590((__int64)&byte_5033768) )
    {
      v126[1] = 6;
      v126[0] = "always";
      v128 = "Always emit EH frame entries";
      v130 = "no-compact-unwind";
      v133 = "Only emit EH frame entries when compact unwind is not available";
      v135 = "default";
      v138 = "Use target platform default";
      v125 = 0x400000003LL;
      v122 = &v121;
      v123[0] = "Whether to emit DWARF EH frame entries.";
      v124 = (const char *)v126;
      v127 = 0;
      v129 = 28;
      v131 = 17;
      v132 = 1;
      v134 = 63;
      v136 = 7;
      v137 = 2;
      v139 = 27;
      v121 = 2;
      v123[1] = (const char *)39;
      sub_3148FB0((__int64)&unk_5033780, "emit-dwarf-unwind", (__int64 *)v123, &v122, (__int64 *)&v124);
      __cxa_atexit((void (*)(void *))sub_3148670, &unk_5033780, &qword_4A427C0);
      sub_2207640((__int64)&byte_5033768);
      if ( v124 != (const char *)v126 )
        _libc_free((unsigned __int64)v124);
    }
  }
  qword_5033EB0 = (__int64)&unk_5033780;
  if ( !byte_5033688 && (unsigned int)sub_2207590((__int64)&byte_5033688) )
  {
    qword_50336A0 = (__int64)&unk_49DC150;
    v91 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    dword_50336AC &= 0x8000u;
    word_50336B0 = 0;
    qword_50336F0 = 0x100000000LL;
    qword_50336B8 = 0;
    qword_50336C0 = 0;
    qword_50336C8 = 0;
    dword_50336A8 = v91;
    qword_50336D0 = 0;
    qword_50336D8 = 0;
    qword_50336E0 = 0;
    qword_50336E8 = (__int64)&unk_50336F8;
    qword_5033700 = 0;
    qword_5033708 = (__int64)&unk_5033720;
    qword_5033710 = 1;
    dword_5033718 = 0;
    byte_503371C = 1;
    v92 = sub_C57470();
    v95 = (unsigned int)qword_50336F0;
    v96 = (unsigned int)qword_50336F0 + 1LL;
    if ( v96 > HIDWORD(qword_50336F0) )
    {
      sub_C8D5F0((__int64)&unk_50336F8 - 16, &unk_50336F8, v96, 8u, v93, v94);
      v95 = (unsigned int)qword_50336F0;
    }
    *(_QWORD *)(qword_50336E8 + 8 * v95) = v92;
    LODWORD(qword_50336F0) = qword_50336F0 + 1;
    qword_5033728 = 0;
    qword_5033730 = (__int64)&unk_49D9748;
    qword_5033738 = 0;
    qword_50336A0 = (__int64)&unk_49DC090;
    qword_5033740 = (__int64)&unk_49DC1D0;
    qword_5033760 = (__int64)nullsub_23;
    qword_5033758 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_50336A0, (__int64)"emit-compact-unwind-non-canonical", 33);
    qword_50336D0 = 64;
    qword_50336C8 = (__int64)"Whether to try to emit Compact Unwind for non canonical entries.";
    LOWORD(qword_5033738) = 256;
    LOBYTE(qword_5033728) = 0;
    sub_C53130((__int64)&qword_50336A0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_50336A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033688);
  }
  qword_5033EA8 = (__int64)&qword_50336A0;
  if ( !byte_50335A8 && (unsigned int)sub_2207590((__int64)&byte_50335A8) )
  {
    qword_50335C0 = (__int64)&unk_49DC150;
    v85 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    dword_50335CC &= 0x8000u;
    word_50335D0 = 0;
    qword_5033610 = 0x100000000LL;
    qword_50335D8 = 0;
    qword_50335E0 = 0;
    qword_50335E8 = 0;
    dword_50335C8 = v85;
    qword_50335F0 = 0;
    qword_50335F8 = 0;
    qword_5033600 = 0;
    qword_5033608 = (__int64)&unk_5033618;
    qword_5033620 = 0;
    qword_5033628 = (__int64)&unk_5033640;
    qword_5033630 = 1;
    dword_5033638 = 0;
    byte_503363C = 1;
    v86 = sub_C57470();
    v89 = (unsigned int)qword_5033610;
    v90 = (unsigned int)qword_5033610 + 1LL;
    if ( v90 > HIDWORD(qword_5033610) )
    {
      sub_C8D5F0((__int64)&unk_5033618 - 16, &unk_5033618, v90, 8u, v87, v88);
      v89 = (unsigned int)qword_5033610;
    }
    *(_QWORD *)(qword_5033608 + 8 * v89) = v86;
    LODWORD(qword_5033610) = qword_5033610 + 1;
    qword_5033648 = 0;
    qword_5033650 = (__int64)&unk_49D9748;
    qword_5033658 = 0;
    qword_50335C0 = (__int64)&unk_49DC090;
    qword_5033660 = (__int64)&unk_49DC1D0;
    qword_5033680 = (__int64)nullsub_23;
    qword_5033678 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_50335C0, (__int64)"asm-show-inst", 13);
    qword_50335F0 = 57;
    qword_50335E8 = (__int64)"Emit internal instruction representation to assembly file";
    sub_C53130((__int64)&qword_50335C0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_50335C0, &qword_4A427C0);
    sub_2207640((__int64)&byte_50335A8);
  }
  qword_5033EA0 = (__int64)&qword_50335C0;
  if ( !byte_50334C8 && (unsigned int)sub_2207590((__int64)&byte_50334C8) )
  {
    qword_50334E0 = (__int64)&unk_49DC150;
    v79 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    dword_50334EC &= 0x8000u;
    word_50334F0 = 0;
    qword_5033530 = 0x100000000LL;
    qword_50334F8 = 0;
    qword_5033500 = 0;
    qword_5033508 = 0;
    dword_50334E8 = v79;
    qword_5033510 = 0;
    qword_5033518 = 0;
    qword_5033520 = 0;
    qword_5033528 = (__int64)&unk_5033538;
    qword_5033540 = 0;
    qword_5033548 = (__int64)&unk_5033560;
    qword_5033550 = 1;
    dword_5033558 = 0;
    byte_503355C = 1;
    v80 = sub_C57470();
    v83 = (unsigned int)qword_5033530;
    v84 = (unsigned int)qword_5033530 + 1LL;
    if ( v84 > HIDWORD(qword_5033530) )
    {
      sub_C8D5F0((__int64)&unk_5033538 - 16, &unk_5033538, v84, 8u, v81, v82);
      v83 = (unsigned int)qword_5033530;
    }
    *(_QWORD *)(qword_5033528 + 8 * v83) = v80;
    LODWORD(qword_5033530) = qword_5033530 + 1;
    qword_5033568 = 0;
    qword_5033570 = (__int64)&unk_49D9748;
    qword_5033578 = 0;
    qword_50334E0 = (__int64)&unk_49DC090;
    qword_5033580 = (__int64)&unk_49DC1D0;
    qword_50335A0 = (__int64)nullsub_23;
    qword_5033598 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_50334E0, (__int64)"fatal-warnings", 14);
    qword_5033510 = 24;
    qword_5033508 = (__int64)"Treat warnings as errors";
    sub_C53130((__int64)&qword_50334E0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_50334E0, &qword_4A427C0);
    sub_2207640((__int64)&byte_50334C8);
  }
  qword_5033E98 = (__int64)&qword_50334E0;
  if ( !byte_50333F0 && (unsigned int)sub_2207590((__int64)&byte_50333F0) )
  {
    qword_5033400 = (__int64)&unk_49DC150;
    v73 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    dword_503340C &= 0x8000u;
    word_5033410 = 0;
    qword_5033450 = 0x100000000LL;
    qword_5033418 = 0;
    qword_5033420 = 0;
    qword_5033428 = 0;
    dword_5033408 = v73;
    qword_5033430 = 0;
    qword_5033438 = 0;
    qword_5033440 = 0;
    qword_5033448 = (__int64)&unk_5033458;
    qword_5033460 = 0;
    qword_5033468 = (__int64)&unk_5033480;
    qword_5033470 = 1;
    dword_5033478 = 0;
    byte_503347C = 1;
    v74 = sub_C57470();
    v77 = (unsigned int)qword_5033450;
    v78 = (unsigned int)qword_5033450 + 1LL;
    if ( v78 > HIDWORD(qword_5033450) )
    {
      sub_C8D5F0((__int64)&unk_5033458 - 16, &unk_5033458, v78, 8u, v75, v76);
      v77 = (unsigned int)qword_5033450;
    }
    *(_QWORD *)(qword_5033448 + 8 * v77) = v74;
    LODWORD(qword_5033450) = qword_5033450 + 1;
    qword_5033488 = 0;
    qword_5033490 = (__int64)&unk_49D9748;
    qword_5033498 = 0;
    qword_5033400 = (__int64)&unk_49DC090;
    qword_50334A0 = (__int64)&unk_49DC1D0;
    qword_50334C0 = (__int64)nullsub_23;
    qword_50334B8 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5033400, (__int64)"no-warn", 7);
    qword_5033430 = 21;
    qword_5033428 = (__int64)"Suppress all warnings";
    sub_C53130((__int64)&qword_5033400);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5033400, &qword_4A427C0);
    sub_2207640((__int64)&byte_50333F0);
  }
  if ( !byte_5033348 && (unsigned int)sub_2207590((__int64)&byte_5033348) )
  {
    qword_5033360 = (__int64)&unk_49DC150;
    v61 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033378 = 0;
    word_5033370 = 0;
    qword_5033380 = 0;
    qword_5033388 = 0;
    dword_503336C = dword_503336C & 0x8000 | 0x20;
    qword_50333B0 = 0x100000000LL;
    dword_5033368 = v61;
    qword_5033390 = 0;
    qword_5033398 = 0;
    qword_50333A0 = 0;
    qword_50333A8 = (__int64)&unk_50333B8;
    qword_50333C0 = 0;
    qword_50333C8 = (__int64)&unk_50333E0;
    qword_50333D0 = 1;
    dword_50333D8 = 0;
    byte_50333DC = 1;
    v62 = sub_C57470();
    v65 = (unsigned int)qword_50333B0;
    v66 = (unsigned int)qword_50333B0 + 1LL;
    if ( v66 > HIDWORD(qword_50333B0) )
    {
      sub_C8D5F0((__int64)&unk_50333B8 - 16, &unk_50333B8, v66, 8u, v63, v64);
      v65 = (unsigned int)qword_50333B0;
    }
    v67 = (const char **)"W";
    *(_QWORD *)(qword_50333A8 + 8 * v65) = v62;
    LODWORD(qword_50333B0) = qword_50333B0 + 1;
    qword_50333E8 = 0;
    qword_5033360 = (__int64)&unk_49DC380;
    sub_C53080((__int64)&qword_5033360, (__int64)"W", 1);
    qword_5033390 = 19;
    qword_5033388 = (__int64)"Alias for --no-warn";
    if ( qword_50333E8 )
    {
      v72 = sub_CEADF0();
      v67 = &v124;
      v124 = "cl::alias must only have one cl::aliasopt(...) specified!";
      LOWORD(v127) = 259;
      sub_C53280((__int64)&qword_5033360, (__int64)&v124, 0, 0, (__int64)v72);
    }
    qword_50333E8 = (__int64)&qword_5033400;
    sub_C53EE0((__int64)&qword_5033360, v67, v68, v69, v70, v71);
    __cxa_atexit((void (*)(void *))sub_C4FC50, &qword_5033360, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033348);
  }
  qword_5033E90 = (__int64)&qword_5033400;
  if ( !byte_5033268 && (unsigned int)sub_2207590((__int64)&byte_5033268) )
  {
    qword_5033280 = (__int64)&unk_49DC150;
    v55 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_50332D0 = 0x100000000LL;
    word_5033290 = 0;
    dword_503328C &= 0x8000u;
    qword_5033298 = 0;
    qword_50332A0 = 0;
    dword_5033288 = v55;
    qword_50332A8 = 0;
    qword_50332B0 = 0;
    qword_50332B8 = 0;
    qword_50332C0 = 0;
    qword_50332C8 = (__int64)&unk_50332D8;
    qword_50332E0 = 0;
    qword_50332E8 = (__int64)&unk_5033300;
    qword_50332F0 = 1;
    dword_50332F8 = 0;
    byte_50332FC = 1;
    v56 = sub_C57470();
    v59 = (unsigned int)qword_50332D0;
    v60 = (unsigned int)qword_50332D0 + 1LL;
    if ( v60 > HIDWORD(qword_50332D0) )
    {
      sub_C8D5F0((__int64)&unk_50332D8 - 16, &unk_50332D8, v60, 8u, v57, v58);
      v59 = (unsigned int)qword_50332D0;
    }
    *(_QWORD *)(qword_50332C8 + 8 * v59) = v56;
    LODWORD(qword_50332D0) = qword_50332D0 + 1;
    qword_5033308 = 0;
    qword_5033310 = (__int64)&unk_49D9748;
    qword_5033318 = 0;
    qword_5033280 = (__int64)&unk_49DC090;
    qword_5033320 = (__int64)&unk_49DC1D0;
    qword_5033340 = (__int64)nullsub_23;
    qword_5033338 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5033280, (__int64)"no-deprecated-warn", 18);
    qword_50332B0 = 32;
    qword_50332A8 = (__int64)"Suppress all deprecated warnings";
    sub_C53130((__int64)&qword_5033280);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5033280, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033268);
  }
  qword_5033E88 = (__int64)&qword_5033280;
  if ( !byte_5033188 && (unsigned int)sub_2207590((__int64)&byte_5033188) )
  {
    qword_50331A0 = (__int64)&unk_49DC150;
    v49 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_50331F0 = 0x100000000LL;
    dword_50331AC &= 0x8000u;
    word_50331B0 = 0;
    qword_50331B8 = 0;
    qword_50331C0 = 0;
    dword_50331A8 = v49;
    qword_50331C8 = 0;
    qword_50331D0 = 0;
    qword_50331D8 = 0;
    qword_50331E0 = 0;
    qword_50331E8 = (__int64)&unk_50331F8;
    qword_5033200 = 0;
    qword_5033208 = (__int64)&unk_5033220;
    qword_5033210 = 1;
    dword_5033218 = 0;
    byte_503321C = 1;
    v50 = sub_C57470();
    v53 = (unsigned int)qword_50331F0;
    v54 = (unsigned int)qword_50331F0 + 1LL;
    if ( v54 > HIDWORD(qword_50331F0) )
    {
      sub_C8D5F0((__int64)&unk_50331F8 - 16, &unk_50331F8, v54, 8u, v51, v52);
      v53 = (unsigned int)qword_50331F0;
    }
    *(_QWORD *)(qword_50331E8 + 8 * v53) = v50;
    LODWORD(qword_50331F0) = qword_50331F0 + 1;
    qword_5033228 = 0;
    qword_5033230 = (__int64)&unk_49D9748;
    qword_5033238 = 0;
    qword_50331A0 = (__int64)&unk_49DC090;
    qword_5033240 = (__int64)&unk_49DC1D0;
    qword_5033260 = (__int64)nullsub_23;
    qword_5033258 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_50331A0, (__int64)"no-type-check", 13);
    qword_50331D0 = 27;
    qword_50331C8 = (__int64)"Suppress type errors (Wasm)";
    sub_C53130((__int64)&qword_50331A0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_50331A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_5033188);
  }
  qword_5033E80 = (__int64)&qword_50331A0;
  if ( !byte_50330A8 && (unsigned int)sub_2207590((__int64)&byte_50330A8) )
  {
    qword_50330C0 = (__int64)&unk_49DC150;
    v43 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033110 = 0x100000000LL;
    dword_50330CC &= 0x8000u;
    word_50330D0 = 0;
    qword_50330D8 = 0;
    qword_50330E0 = 0;
    dword_50330C8 = v43;
    qword_50330E8 = 0;
    qword_50330F0 = 0;
    qword_50330F8 = 0;
    qword_5033100 = 0;
    qword_5033108 = (__int64)&unk_5033118;
    qword_5033120 = 0;
    qword_5033128 = (__int64)&unk_5033140;
    qword_5033130 = 1;
    dword_5033138 = 0;
    byte_503313C = 1;
    v44 = sub_C57470();
    v47 = (unsigned int)qword_5033110;
    v48 = (unsigned int)qword_5033110 + 1LL;
    if ( v48 > HIDWORD(qword_5033110) )
    {
      sub_C8D5F0((__int64)&unk_5033118 - 16, &unk_5033118, v48, 8u, v45, v46);
      v47 = (unsigned int)qword_5033110;
    }
    *(_QWORD *)(qword_5033108 + 8 * v47) = v44;
    LODWORD(qword_5033110) = qword_5033110 + 1;
    qword_5033148 = 0;
    qword_5033150 = (__int64)&unk_49D9748;
    qword_5033158 = 0;
    qword_50330C0 = (__int64)&unk_49DC090;
    qword_5033160 = (__int64)&unk_49DC1D0;
    qword_5033180 = (__int64)nullsub_23;
    qword_5033178 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_50330C0, (__int64)"save-temp-labels", 16);
    qword_50330F0 = 30;
    qword_50330E8 = (__int64)"Don't discard temporary labels";
    sub_C53130((__int64)&qword_50330C0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_50330C0, &qword_4A427C0);
    sub_2207640((__int64)&byte_50330A8);
  }
  qword_5033E78 = (__int64)&qword_50330C0;
  if ( !byte_5032FC8 && (unsigned int)sub_2207590((__int64)&byte_5032FC8) )
  {
    qword_5032FE0 = (__int64)&unk_49DC150;
    v37 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5033030 = 0x100000000LL;
    dword_5032FEC &= 0x8000u;
    word_5032FF0 = 0;
    qword_5032FF8 = 0;
    qword_5033000 = 0;
    dword_5032FE8 = v37;
    qword_5033008 = 0;
    qword_5033010 = 0;
    qword_5033018 = 0;
    qword_5033020 = 0;
    qword_5033028 = (__int64)&unk_5033038;
    qword_5033040 = 0;
    qword_5033048 = (__int64)&unk_5033060;
    qword_5033050 = 1;
    dword_5033058 = 0;
    byte_503305C = 1;
    v38 = sub_C57470();
    v41 = (unsigned int)qword_5033030;
    v42 = (unsigned int)qword_5033030 + 1LL;
    if ( v42 > HIDWORD(qword_5033030) )
    {
      sub_C8D5F0((__int64)&unk_5033038 - 16, &unk_5033038, v42, 8u, v39, v40);
      v41 = (unsigned int)qword_5033030;
    }
    *(_QWORD *)(qword_5033028 + 8 * v41) = v38;
    LODWORD(qword_5033030) = qword_5033030 + 1;
    qword_5033068 = 0;
    qword_5033070 = (__int64)&unk_49D9748;
    qword_5033078 = 0;
    qword_5032FE0 = (__int64)&unk_49DC090;
    qword_5033080 = (__int64)&unk_49DC1D0;
    qword_50330A0 = (__int64)nullsub_23;
    qword_5033098 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5032FE0, (__int64)"crel", 4);
    qword_5033010 = 34;
    qword_5033008 = (__int64)"Use CREL relocation format for ELF";
    sub_C53130((__int64)&qword_5032FE0);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5032FE0, &qword_4A427C0);
    sub_2207640((__int64)&byte_5032FC8);
  }
  qword_5033E70 = (__int64)&qword_5032FE0;
  if ( !byte_5032EE8 && (unsigned int)sub_2207590((__int64)&byte_5032EE8) )
  {
    qword_5032F00 = (__int64)&unk_49DC150;
    v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5032F50 = 0x100000000LL;
    dword_5032F0C &= 0x8000u;
    word_5032F10 = 0;
    qword_5032F18 = 0;
    qword_5032F20 = 0;
    dword_5032F08 = v31;
    qword_5032F28 = 0;
    qword_5032F30 = 0;
    qword_5032F38 = 0;
    qword_5032F40 = 0;
    qword_5032F48 = (__int64)&unk_5032F58;
    qword_5032F60 = 0;
    qword_5032F68 = (__int64)&unk_5032F80;
    qword_5032F70 = 1;
    dword_5032F78 = 0;
    byte_5032F7C = 1;
    v32 = sub_C57470();
    v35 = (unsigned int)qword_5032F50;
    v36 = (unsigned int)qword_5032F50 + 1LL;
    if ( v36 > HIDWORD(qword_5032F50) )
    {
      sub_C8D5F0((__int64)&unk_5032F58 - 16, &unk_5032F58, v36, 8u, v33, v34);
      v35 = (unsigned int)qword_5032F50;
    }
    *(_QWORD *)(qword_5032F48 + 8 * v35) = v32;
    LODWORD(qword_5032F50) = qword_5032F50 + 1;
    qword_5032F88 = 0;
    qword_5032F90 = (__int64)&unk_49D9748;
    qword_5032F98 = 0;
    qword_5032F00 = (__int64)&unk_49DC090;
    qword_5032FA0 = (__int64)&unk_49DC1D0;
    qword_5032FC0 = (__int64)nullsub_23;
    qword_5032FB8 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5032F00, (__int64)"implicit-mapsyms", 16);
    qword_5032F30 = 209;
    qword_5032F28 = (__int64)"Allow mapping symbol at section beginning to be implicit, lowering number of mapping symbol"
                             "s at the expense of some portability. Recommended for projects that can build all their obj"
                             "ect files using this option";
    sub_C53130((__int64)&qword_5032F00);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5032F00, &qword_4A427C0);
    sub_2207640((__int64)&byte_5032EE8);
  }
  qword_5033E68 = (__int64)&qword_5032F00;
  if ( !byte_5032E08 && (unsigned int)sub_2207590((__int64)&byte_5032E08) )
  {
    qword_5032E20 = (__int64)&unk_49DC150;
    v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5032E70 = 0x100000000LL;
    dword_5032E2C &= 0x8000u;
    word_5032E30 = 0;
    qword_5032E38 = 0;
    qword_5032E40 = 0;
    dword_5032E28 = v25;
    qword_5032E48 = 0;
    qword_5032E50 = 0;
    qword_5032E58 = 0;
    qword_5032E60 = 0;
    qword_5032E68 = (__int64)&unk_5032E78;
    qword_5032E80 = 0;
    qword_5032E88 = (__int64)&unk_5032EA0;
    qword_5032E90 = 1;
    dword_5032E98 = 0;
    byte_5032E9C = 1;
    v26 = sub_C57470();
    v29 = (unsigned int)qword_5032E70;
    v30 = (unsigned int)qword_5032E70 + 1LL;
    if ( v30 > HIDWORD(qword_5032E70) )
    {
      sub_C8D5F0((__int64)&unk_5032E78 - 16, &unk_5032E78, v30, 8u, v27, v28);
      v29 = (unsigned int)qword_5032E70;
    }
    *(_QWORD *)(qword_5032E68 + 8 * v29) = v26;
    LODWORD(qword_5032E70) = qword_5032E70 + 1;
    qword_5032EA8 = 0;
    qword_5032EB0 = (__int64)&unk_49D9748;
    qword_5032EB8 = 0;
    qword_5032E20 = (__int64)&unk_49DC090;
    qword_5032EC0 = (__int64)&unk_49DC1D0;
    qword_5032EE0 = (__int64)nullsub_23;
    qword_5032ED8 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5032E20, (__int64)"x86-relax-relocations", 21);
    qword_5032E48 = (__int64)"Emit GOTPCRELX/REX_GOTPCRELX/CODE_4_GOTPCRELX instead of GOTPCREL on x86-64 ELF";
    LOWORD(qword_5032EB8) = 257;
    qword_5032E50 = 79;
    LOBYTE(qword_5032EA8) = 1;
    sub_C53130((__int64)&qword_5032E20);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5032E20, &qword_4A427C0);
    sub_2207640((__int64)&byte_5032E08);
  }
  qword_5033E60 = (__int64)&qword_5032E20;
  if ( !byte_5032D20 && (unsigned int)sub_2207590((__int64)&byte_5032D20) )
  {
    qword_5032D40 = (__int64)&unk_49DC150;
    v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_5032D90 = 0x100000000LL;
    word_5032D50 = 0;
    dword_5032D4C &= 0x8000u;
    qword_5032D58 = 0;
    qword_5032D60 = 0;
    dword_5032D48 = v19;
    qword_5032D68 = 0;
    qword_5032D70 = 0;
    qword_5032D78 = 0;
    qword_5032D80 = 0;
    qword_5032D88 = (__int64)&unk_5032D98;
    qword_5032DA0 = 0;
    qword_5032DA8 = (__int64)&unk_5032DC0;
    qword_5032DB0 = 1;
    dword_5032DB8 = 0;
    byte_5032DBC = 1;
    v20 = sub_C57470();
    v23 = (unsigned int)qword_5032D90;
    v24 = (unsigned int)qword_5032D90 + 1LL;
    if ( v24 > HIDWORD(qword_5032D90) )
    {
      sub_C8D5F0((__int64)&unk_5032D98 - 16, &unk_5032D98, v24, 8u, v21, v22);
      v23 = (unsigned int)qword_5032D90;
    }
    *(_QWORD *)(qword_5032D88 + 8 * v23) = v20;
    LODWORD(qword_5032D90) = qword_5032D90 + 1;
    qword_5032DC8 = 0;
    qword_5032DD0 = (__int64)&unk_49D9748;
    qword_5032DD8 = 0;
    qword_5032D40 = (__int64)&unk_49DC090;
    qword_5032DE0 = (__int64)&unk_49DC1D0;
    qword_5032E00 = (__int64)nullsub_23;
    qword_5032DF8 = (__int64)sub_984030;
    sub_C53080((__int64)&qword_5032D40, (__int64)"x86-sse2avx", 11);
    qword_5032D70 = 73;
    qword_5032D68 = (__int64)"Specify that the assembler should encode SSE instructions with VEX prefix";
    sub_C53130((__int64)&qword_5032D40);
    __cxa_atexit((void (*)(void *))sub_984900, &qword_5032D40, &qword_4A427C0);
    sub_2207640((__int64)&byte_5032D20);
  }
  qword_5033E58 = (__int64)&qword_5032D40;
  if ( !byte_5032C00 && (unsigned int)sub_2207590((__int64)&byte_5032C00) )
  {
    v125 = 52;
    v123[0] = byte_3F871B3;
    v124 = "The name of the ABI to be targeted from the backend.";
    qword_5032C20 = (__int64)&unk_49DC150;
    v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    byte_5032C9C = 1;
    qword_5032C70 = 0x100000000LL;
    dword_5032C2C &= 0x8000u;
    qword_5032C38 = 0;
    qword_5032C40 = 0;
    qword_5032C48 = 0;
    dword_5032C28 = v13;
    word_5032C30 = 0;
    qword_5032C50 = 0;
    qword_5032C58 = 0;
    qword_5032C60 = 0;
    qword_5032C68 = (__int64)&unk_5032C78;
    qword_5032C80 = 0;
    qword_5032C88 = (__int64)&unk_5032CA0;
    qword_5032C90 = 1;
    dword_5032C98 = 0;
    v14 = sub_C57470();
    v17 = (unsigned int)qword_5032C70;
    v18 = (unsigned int)qword_5032C70 + 1LL;
    if ( v18 > HIDWORD(qword_5032C70) )
    {
      sub_C8D5F0((__int64)&unk_5032C78 - 16, &unk_5032C78, v18, 8u, v15, v16);
      v17 = (unsigned int)qword_5032C70;
    }
    *(_QWORD *)(qword_5032C68 + 8 * v17) = v14;
    qword_5032CA8 = (__int64)&byte_5032CB8;
    qword_5032CD0 = (__int64)&byte_5032CE0;
    LODWORD(qword_5032C70) = qword_5032C70 + 1;
    qword_5032CB0 = 0;
    qword_5032CC8 = (__int64)&unk_49DC130;
    byte_5032CB8 = 0;
    byte_5032CE0 = 0;
    qword_5032C20 = (__int64)&unk_49DC010;
    qword_5032CD8 = 0;
    byte_5032CF0 = 0;
    qword_5032CF8 = (__int64)&unk_49DC350;
    qword_5032D18 = (__int64)nullsub_92;
    qword_5032D10 = (__int64)sub_BC4D70;
    sub_3148DC0((__int64)&qword_5032C20, "target-abi", (__int64 *)&v124, v123);
    sub_C53130((__int64)&qword_5032C20);
    __cxa_atexit((void (*)(void *))sub_BC5A40, &qword_5032C20, &qword_4A427C0);
    sub_2207640((__int64)&byte_5032C00);
  }
  qword_5033E50 = (__int64)&qword_5032C20;
  if ( !byte_5032AE8 && (unsigned int)sub_2207590((__int64)&byte_5032AE8) )
  {
    qword_5032B00 = (__int64)&unk_49DC150;
    v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    dword_5032B0C &= 0x8000u;
    word_5032B10 = 0;
    qword_5032B50 = 0x100000000LL;
    qword_5032B18 = 0;
    qword_5032B20 = 0;
    qword_5032B28 = 0;
    dword_5032B08 = v7;
    qword_5032B30 = 0;
    qword_5032B38 = 0;
    qword_5032B40 = 0;
    qword_5032B48 = (__int64)&unk_5032B58;
    qword_5032B60 = 0;
    qword_5032B68 = (__int64)&unk_5032B80;
    qword_5032B70 = 1;
    dword_5032B78 = 0;
    byte_5032B7C = 1;
    v8 = sub_C57470();
    v11 = (unsigned int)qword_5032B50;
    v12 = (unsigned int)qword_5032B50 + 1LL;
    if ( v12 > HIDWORD(qword_5032B50) )
    {
      sub_C8D5F0((__int64)&unk_5032B58 - 16, &unk_5032B58, v12, 8u, v9, v10);
      v11 = (unsigned int)qword_5032B50;
    }
    *(_QWORD *)(qword_5032B48 + 8 * v11) = v8;
    qword_5032B88 = (__int64)&byte_5032B98;
    qword_5032BB0 = (__int64)&byte_5032BC0;
    LODWORD(qword_5032B50) = qword_5032B50 + 1;
    qword_5032B90 = 0;
    qword_5032BA8 = (__int64)&unk_49DC130;
    byte_5032B98 = 0;
    byte_5032BC0 = 0;
    qword_5032B00 = (__int64)&unk_49DC010;
    qword_5032BB8 = 0;
    byte_5032BD0 = 0;
    qword_5032BD8 = (__int64)&unk_49DC350;
    qword_5032BF8 = (__int64)nullsub_92;
    qword_5032BF0 = (__int64)sub_BC4D70;
    sub_C53080((__int64)&qword_5032B00, (__int64)"as-secure-log-file", 18);
    qword_5032B30 = 23;
    qword_5032B28 = (__int64)"As secure log file name";
    LOBYTE(dword_5032B0C) = dword_5032B0C & 0x9F | 0x20;
    sub_C53130((__int64)&qword_5032B00);
    __cxa_atexit((void (*)(void *))sub_BC5A40, &qword_5032B00, &qword_4A427C0);
    sub_2207640((__int64)&byte_5032AE8);
  }
  qword_5033E48 = (__int64)&qword_5032B00;
  return &qword_5032B00;
}
