// Function: ctor_033_0
// Address: 0x48aff0
//
int ctor_033_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  char **v16; // rcx
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // edx
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  int v25; // edx
  __int64 v26; // rbx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  int v29; // edx
  __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  int v33; // edx
  __int64 v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v38; // [rsp+2Ch] [rbp-234h] BYREF
  int v39; // [rsp+30h] [rbp-230h] BYREF
  int v40; // [rsp+34h] [rbp-22Ch] BYREF
  int *v41; // [rsp+38h] [rbp-228h] BYREF
  const char *v42; // [rsp+40h] [rbp-220h] BYREF
  __int64 v43; // [rsp+48h] [rbp-218h]
  _BYTE v44[160]; // [rsp+50h] [rbp-210h] BYREF
  char *v45; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v46; // [rsp+F8h] [rbp-168h]
  char *v47; // [rsp+100h] [rbp-160h] BYREF
  __int64 v48; // [rsp+108h] [rbp-158h]
  __int64 v49; // [rsp+110h] [rbp-150h]
  char *v50; // [rsp+118h] [rbp-148h]
  __int64 v51; // [rsp+120h] [rbp-140h]
  const char *v52; // [rsp+128h] [rbp-138h]
  __int64 v53; // [rsp+130h] [rbp-130h]
  __int64 v54; // [rsp+138h] [rbp-128h]
  char *v55; // [rsp+140h] [rbp-120h]
  __int64 v56; // [rsp+148h] [rbp-118h]
  const char *v57; // [rsp+150h] [rbp-110h]
  __int64 v58; // [rsp+158h] [rbp-108h]
  __int64 v59; // [rsp+160h] [rbp-100h]
  const char *v60; // [rsp+168h] [rbp-F8h]
  __int64 v61; // [rsp+170h] [rbp-F0h]
  const char *v62; // [rsp+178h] [rbp-E8h]
  const char *v63; // [rsp+180h] [rbp-E0h]
  __int64 v64; // [rsp+188h] [rbp-D8h]
  const char *v65; // [rsp+190h] [rbp-D0h]
  __int64 v66; // [rsp+198h] [rbp-C8h]
  int v67; // [rsp+1A0h] [rbp-C0h]
  const char *v68; // [rsp+1A8h] [rbp-B8h]
  __int64 v69; // [rsp+1B0h] [rbp-B0h]
  const char *v70; // [rsp+1B8h] [rbp-A8h]
  __int64 v71; // [rsp+1C0h] [rbp-A0h]
  int v72; // [rsp+1C8h] [rbp-98h]
  const char *v73; // [rsp+1D0h] [rbp-90h]
  __int64 v74; // [rsp+1D8h] [rbp-88h]
  const char *v75; // [rsp+1E0h] [rbp-80h]
  __int64 v76; // [rsp+1E8h] [rbp-78h]
  int v77; // [rsp+1F0h] [rbp-70h]
  const char *v78; // [rsp+1F8h] [rbp-68h]
  __int64 v79; // [rsp+200h] [rbp-60h]
  const char *v80; // [rsp+208h] [rbp-58h]
  __int64 v81; // [rsp+210h] [rbp-50h]
  int v82; // [rsp+218h] [rbp-48h]
  const char *v83; // [rsp+220h] [rbp-40h]
  char *v84; // [rsp+228h] [rbp-38h]

  qword_4F834E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  word_4F834F0 = 0;
  qword_4F834F8 = 0;
  qword_4F83500 = 0;
  dword_4F834EC = dword_4F834EC & 0x8000 | 1;
  qword_4F83530 = 0x100000000LL;
  dword_4F834E8 = v0;
  qword_4F83508 = 0;
  qword_4F83510 = 0;
  qword_4F83518 = 0;
  qword_4F83520 = 0;
  qword_4F83528 = (__int64)&unk_4F83538;
  qword_4F83540 = 0;
  qword_4F83548 = (__int64)&unk_4F83560;
  qword_4F83550 = 1;
  dword_4F83558 = 0;
  byte_4F8355C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F83530;
  v3 = (unsigned int)qword_4F83530 + 1LL;
  if ( v3 > HIDWORD(qword_4F83530) )
  {
    sub_C8D5F0((char *)&unk_4F83538 - 16, &unk_4F83538, v3, 8);
    v2 = (unsigned int)qword_4F83530;
  }
  *(_QWORD *)(qword_4F83528 + 8 * v2) = v1;
  LODWORD(qword_4F83530) = qword_4F83530 + 1;
  qword_4F83568 = 0;
  qword_4F834E0 = (__int64)&unk_49DAD08;
  qword_4F83570 = 0;
  qword_4F83578 = 0;
  qword_4F835B8 = (__int64)&unk_49DC350;
  qword_4F83580 = 0;
  qword_4F835D8 = (__int64)nullsub_81;
  qword_4F83588 = 0;
  qword_4F835D0 = (__int64)sub_BB8600;
  qword_4F83590 = 0;
  byte_4F83598 = 0;
  qword_4F835A0 = 0;
  qword_4F835A8 = 0;
  qword_4F835B0 = 0;
  sub_C53080(&qword_4F834E0, "print-before", 12);
  BYTE1(dword_4F834EC) |= 2u;
  qword_4F83508 = (__int64)"Print IR before specified passes";
  qword_4F83510 = 32;
  LOBYTE(dword_4F834EC) = dword_4F834EC & 0x9F | 0x20;
  sub_C53130(&qword_4F834E0);
  __cxa_atexit(sub_BB89D0, &qword_4F834E0, &qword_4A427C0);
  qword_4F833E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  word_4F833F0 = 0;
  qword_4F833F8 = 0;
  qword_4F83400 = 0;
  dword_4F833EC = dword_4F833EC & 0x8000 | 1;
  qword_4F83430 = 0x100000000LL;
  dword_4F833E8 = v4;
  qword_4F83408 = 0;
  qword_4F83410 = 0;
  qword_4F83418 = 0;
  qword_4F83420 = 0;
  qword_4F83428 = (__int64)&unk_4F83438;
  qword_4F83440 = 0;
  qword_4F83448 = (__int64)&unk_4F83460;
  qword_4F83450 = 1;
  dword_4F83458 = 0;
  byte_4F8345C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F83430;
  v7 = (unsigned int)qword_4F83430 + 1LL;
  if ( v7 > HIDWORD(qword_4F83430) )
  {
    sub_C8D5F0((char *)&unk_4F83438 - 16, &unk_4F83438, v7, 8);
    v6 = (unsigned int)qword_4F83430;
  }
  *(_QWORD *)(qword_4F83428 + 8 * v6) = v5;
  LODWORD(qword_4F83430) = qword_4F83430 + 1;
  qword_4F83468 = 0;
  qword_4F833E0 = (__int64)&unk_49DAD08;
  qword_4F83470 = 0;
  qword_4F83478 = 0;
  qword_4F834B8 = (__int64)&unk_49DC350;
  qword_4F83480 = 0;
  qword_4F834D8 = (__int64)nullsub_81;
  qword_4F83488 = 0;
  qword_4F834D0 = (__int64)sub_BB8600;
  qword_4F83490 = 0;
  byte_4F83498 = 0;
  qword_4F834A0 = 0;
  qword_4F834A8 = 0;
  qword_4F834B0 = 0;
  sub_C53080(&qword_4F833E0, "print-after", 11);
  BYTE1(dword_4F833EC) |= 2u;
  qword_4F83408 = (__int64)"Print IR after specified passes";
  qword_4F83410 = 31;
  LOBYTE(dword_4F833EC) = dword_4F833EC & 0x9F | 0x20;
  sub_C53130(&qword_4F833E0);
  __cxa_atexit(sub_BB89D0, &qword_4F833E0, &qword_4A427C0);
  qword_4F83300 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F83350 = 0x100000000LL;
  dword_4F8330C &= 0x8000u;
  word_4F83310 = 0;
  qword_4F83318 = 0;
  qword_4F83320 = 0;
  dword_4F83308 = v8;
  qword_4F83328 = 0;
  qword_4F83330 = 0;
  qword_4F83338 = 0;
  qword_4F83340 = 0;
  qword_4F83348 = (__int64)&unk_4F83358;
  qword_4F83360 = 0;
  qword_4F83368 = (__int64)&unk_4F83380;
  qword_4F83370 = 1;
  dword_4F83378 = 0;
  byte_4F8337C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4F83350;
  v11 = (unsigned int)qword_4F83350 + 1LL;
  if ( v11 > HIDWORD(qword_4F83350) )
  {
    sub_C8D5F0((char *)&unk_4F83358 - 16, &unk_4F83358, v11, 8);
    v10 = (unsigned int)qword_4F83350;
  }
  *(_QWORD *)(qword_4F83348 + 8 * v10) = v9;
  LODWORD(qword_4F83350) = qword_4F83350 + 1;
  qword_4F83388 = 0;
  qword_4F83390 = (__int64)&unk_49D9748;
  qword_4F83398 = 0;
  qword_4F83300 = (__int64)&unk_49DC090;
  qword_4F833A0 = (__int64)&unk_49DC1D0;
  qword_4F833C0 = (__int64)nullsub_23;
  qword_4F833B8 = (__int64)sub_984030;
  sub_C53080(&qword_4F83300, "print-before-all", 16);
  qword_4F83328 = (__int64)"Print IR before each pass";
  LOWORD(qword_4F83398) = 256;
  LOBYTE(qword_4F83388) = 0;
  qword_4F83330 = 25;
  LOBYTE(dword_4F8330C) = dword_4F8330C & 0x9F | 0x20;
  sub_C53130(&qword_4F83300);
  __cxa_atexit(sub_984900, &qword_4F83300, &qword_4A427C0);
  qword_4F83220 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F83270 = 0x100000000LL;
  dword_4F8322C &= 0x8000u;
  word_4F83230 = 0;
  qword_4F83238 = 0;
  qword_4F83240 = 0;
  dword_4F83228 = v12;
  qword_4F83248 = 0;
  qword_4F83250 = 0;
  qword_4F83258 = 0;
  qword_4F83260 = 0;
  qword_4F83268 = (__int64)&unk_4F83278;
  qword_4F83280 = 0;
  qword_4F83288 = (__int64)&unk_4F832A0;
  qword_4F83290 = 1;
  dword_4F83298 = 0;
  byte_4F8329C = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4F83270;
  v15 = (unsigned int)qword_4F83270 + 1LL;
  if ( v15 > HIDWORD(qword_4F83270) )
  {
    sub_C8D5F0((char *)&unk_4F83278 - 16, &unk_4F83278, v15, 8);
    v14 = (unsigned int)qword_4F83270;
  }
  *(_QWORD *)(qword_4F83268 + 8 * v14) = v13;
  LODWORD(qword_4F83270) = qword_4F83270 + 1;
  qword_4F832A8 = 0;
  qword_4F832B0 = (__int64)&unk_49D9748;
  qword_4F832B8 = 0;
  qword_4F83220 = (__int64)&unk_49DC090;
  qword_4F832C0 = (__int64)&unk_49DC1D0;
  qword_4F832E0 = (__int64)nullsub_23;
  qword_4F832D8 = (__int64)sub_984030;
  sub_C53080(&qword_4F83220, "print-after-all", 15);
  LOWORD(qword_4F832B8) = 256;
  qword_4F83248 = (__int64)"Print IR after each pass";
  qword_4F83250 = 24;
  LOBYTE(qword_4F832A8) = 0;
  LOBYTE(dword_4F8322C) = dword_4F8322C & 0x9F | 0x20;
  sub_C53130(&qword_4F83220);
  __cxa_atexit(sub_984900, &qword_4F83220, &qword_4A427C0);
  v47 = "regs";
  v50 = "print register pressure";
  v55 = "print function IR size";
  v52 = "fnsize";
  v60 = "print module IR size";
  v57 = "modsize";
  v46 = 0x400000004LL;
  v65 = "(default) print everything";
  v42 = "Print extra information after each pass";
  v45 = (char *)&v47;
  v48 = 4;
  LODWORD(v49) = 1;
  v51 = 23;
  v53 = 6;
  LODWORD(v54) = 2;
  v56 = 22;
  v58 = 7;
  LODWORD(v59) = 4;
  v61 = 20;
  v62 = byte_3F871B3;
  v63 = 0;
  LODWORD(v64) = 255;
  v66 = 26;
  LODWORD(v41) = 1;
  v40 = 1;
  v39 = 1;
  v43 = 39;
  sub_BC6A10(&unk_4F82F80, "extra-print-after-all", &v42, &v39);
  if ( v45 != (char *)&v47 )
    _libc_free(v45, "extra-print-after-all");
  __cxa_atexit(sub_BC56A0, &unk_4F82F80, &qword_4A427C0);
  v45 = "quiet";
  v48 = (__int64)"Run in quiet mode";
  v50 = "diff";
  v53 = (__int64)"Display patch-like changes";
  v55 = "diff-quiet";
  v58 = (__int64)"Display patch-like changes in quiet mode";
  v60 = "cdiff";
  v63 = "Display patch-like changes with color";
  v65 = "cdiff-quiet";
  v68 = "Display patch-like changes in quiet mode with color";
  v70 = "dot-cfg";
  v73 = "Create a website with graphical changes";
  v75 = "dot-cfg-quiet";
  v46 = 5;
  LODWORD(v47) = 2;
  v49 = 17;
  v51 = 4;
  LODWORD(v52) = 3;
  v54 = 26;
  v56 = 10;
  LODWORD(v57) = 4;
  v59 = 40;
  v61 = 5;
  LODWORD(v62) = 5;
  v64 = 37;
  v66 = 11;
  v67 = 6;
  v69 = 51;
  v71 = 7;
  v72 = 7;
  v74 = 39;
  v76 = 13;
  v77 = 8;
  v80 = byte_3F871B3;
  v83 = byte_3F871B3;
  v78 = "Create a website with graphical changes in quiet mode";
  v43 = 0x400000000LL;
  v79 = 53;
  v81 = 0;
  v82 = 1;
  v84 = 0;
  v42 = v44;
  sub_C8D5F0(&v42, v44, 8, 40);
  v16 = (char **)&v42[40 * (unsigned int)v43];
  *v16 = v45;
  v16[39] = v84;
  qmemcpy(
    (void *)((unsigned __int64)(v16 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    (const void *)((char *)&v45 - ((char *)v16 - ((unsigned __int64)(v16 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
    8LL * (((unsigned int)v16 - (((_DWORD)v16 + 8) & 0xFFFFFFF8) + 320) >> 3));
  v40 = 0;
  v39 = 1;
  LODWORD(v43) = v43 + 8;
  v41 = &v40;
  v38 = 1;
  v45 = "Print changed IRs";
  v46 = 17;
  sub_BC6EC0(&unk_4F82D20, "print-changed", &v45, &v38, &v39, &v41, &v42);
  if ( v42 != v44 )
    _libc_free(v42, "print-changed");
  __cxa_atexit(sub_BC5580, &unk_4F82D20, &qword_4A427C0);
  qword_4F82C20 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F82C70 = 0x100000000LL;
  dword_4F82C2C &= 0x8000u;
  word_4F82C30 = 0;
  qword_4F82C38 = 0;
  qword_4F82C40 = 0;
  dword_4F82C28 = v17;
  qword_4F82C48 = 0;
  qword_4F82C50 = 0;
  qword_4F82C58 = 0;
  qword_4F82C60 = 0;
  qword_4F82C68 = (__int64)&unk_4F82C78;
  qword_4F82C80 = 0;
  qword_4F82C88 = (__int64)&unk_4F82CA0;
  qword_4F82C90 = 1;
  dword_4F82C98 = 0;
  byte_4F82C9C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_4F82C70;
  v20 = (unsigned int)qword_4F82C70 + 1LL;
  if ( v20 > HIDWORD(qword_4F82C70) )
  {
    sub_C8D5F0((char *)&unk_4F82C78 - 16, &unk_4F82C78, v20, 8);
    v19 = (unsigned int)qword_4F82C70;
  }
  *(_QWORD *)(qword_4F82C68 + 8 * v19) = v18;
  qword_4F82CA8 = (__int64)&byte_4F82CB8;
  qword_4F82CD0 = (__int64)&byte_4F82CE0;
  LODWORD(qword_4F82C70) = qword_4F82C70 + 1;
  qword_4F82CB0 = 0;
  qword_4F82CC8 = (__int64)&unk_49DC130;
  byte_4F82CB8 = 0;
  byte_4F82CE0 = 0;
  qword_4F82C20 = (__int64)&unk_49DC010;
  qword_4F82CD8 = 0;
  byte_4F82CF0 = 0;
  qword_4F82CF8 = (__int64)&unk_49DC350;
  qword_4F82D18 = (__int64)nullsub_92;
  qword_4F82D10 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4F82C20, "print-changed-diff-path", 23);
  v45 = (char *)&v47;
  strcpy((char *)&v47, "diff");
  LOBYTE(dword_4F82C2C) = dword_4F82C2C & 0x9F | 0x20;
  v46 = 4;
  sub_2240AE0(&qword_4F82CA8, &v45);
  byte_4F82CF0 = 1;
  sub_2240AE0(&qword_4F82CD0, &v45);
  if ( v45 != (char *)&v47 )
    j_j___libc_free_0(v45, v47 + 1);
  qword_4F82C50 = 36;
  qword_4F82C48 = (__int64)"system diff used by change reporters";
  sub_C53130(&qword_4F82C20);
  __cxa_atexit(sub_BC5A40, &qword_4F82C20, &qword_4A427C0);
  qword_4F82B40 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F82B90 = 0x100000000LL;
  dword_4F82B4C &= 0x8000u;
  word_4F82B50 = 0;
  qword_4F82B58 = 0;
  qword_4F82B60 = 0;
  dword_4F82B48 = v21;
  qword_4F82B68 = 0;
  qword_4F82B70 = 0;
  qword_4F82B78 = 0;
  qword_4F82B80 = 0;
  qword_4F82B88 = (__int64)&unk_4F82B98;
  qword_4F82BA0 = 0;
  qword_4F82BA8 = (__int64)&unk_4F82BC0;
  qword_4F82BB0 = 1;
  dword_4F82BB8 = 0;
  byte_4F82BBC = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_4F82B90;
  v24 = (unsigned int)qword_4F82B90 + 1LL;
  if ( v24 > HIDWORD(qword_4F82B90) )
  {
    sub_C8D5F0((char *)&unk_4F82B98 - 16, &unk_4F82B98, v24, 8);
    v23 = (unsigned int)qword_4F82B90;
  }
  *(_QWORD *)(qword_4F82B88 + 8 * v23) = v22;
  LODWORD(qword_4F82B90) = qword_4F82B90 + 1;
  qword_4F82BC8 = 0;
  qword_4F82BD0 = (__int64)&unk_49D9748;
  qword_4F82BD8 = 0;
  qword_4F82B40 = (__int64)&unk_49DC090;
  qword_4F82BE0 = (__int64)&unk_49DC1D0;
  qword_4F82C00 = (__int64)nullsub_23;
  qword_4F82BF8 = (__int64)sub_984030;
  sub_C53080(&qword_4F82B40, "print-module-scope", 18);
  qword_4F82B68 = (__int64)"When printing IR for print-[before|after]{-all} always print a module IR";
  LOWORD(qword_4F82BD8) = 256;
  LOBYTE(qword_4F82BC8) = 0;
  qword_4F82B70 = 72;
  LOBYTE(dword_4F82B4C) = dword_4F82B4C & 0x9F | 0x20;
  sub_C53130(&qword_4F82B40);
  __cxa_atexit(sub_984900, &qword_4F82B40, &qword_4A427C0);
  qword_4F82A60 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F82AB0 = 0x100000000LL;
  dword_4F82A6C &= 0x8000u;
  word_4F82A70 = 0;
  qword_4F82A78 = 0;
  qword_4F82A80 = 0;
  dword_4F82A68 = v25;
  qword_4F82A88 = 0;
  qword_4F82A90 = 0;
  qword_4F82A98 = 0;
  qword_4F82AA0 = 0;
  qword_4F82AA8 = (__int64)&unk_4F82AB8;
  qword_4F82AC0 = 0;
  qword_4F82AC8 = (__int64)&unk_4F82AE0;
  qword_4F82AD0 = 1;
  dword_4F82AD8 = 0;
  byte_4F82ADC = 1;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_4F82AB0;
  v28 = (unsigned int)qword_4F82AB0 + 1LL;
  if ( v28 > HIDWORD(qword_4F82AB0) )
  {
    sub_C8D5F0((char *)&unk_4F82AB8 - 16, &unk_4F82AB8, v28, 8);
    v27 = (unsigned int)qword_4F82AB0;
  }
  *(_QWORD *)(qword_4F82AA8 + 8 * v27) = v26;
  LODWORD(qword_4F82AB0) = qword_4F82AB0 + 1;
  qword_4F82AE8 = 0;
  qword_4F82AF0 = (__int64)&unk_49D9748;
  qword_4F82AF8 = 0;
  qword_4F82A60 = (__int64)&unk_49DC090;
  qword_4F82B00 = (__int64)&unk_49DC1D0;
  qword_4F82B20 = (__int64)nullsub_23;
  qword_4F82B18 = (__int64)sub_984030;
  sub_C53080(&qword_4F82A60, "print-loop-func-scope", 21);
  qword_4F82A88 = (__int64)"When printing IR for print-[before|after]{-all} for a loop pass, always print function IR";
  LOWORD(qword_4F82AF8) = 256;
  LOBYTE(qword_4F82AE8) = 0;
  qword_4F82A90 = 89;
  LOBYTE(dword_4F82A6C) = dword_4F82A6C & 0x9F | 0x20;
  sub_C53130(&qword_4F82A60);
  __cxa_atexit(sub_984900, &qword_4F82A60, &qword_4A427C0);
  qword_4F82960 = (__int64)&unk_49DC150;
  v29 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  word_4F82970 = 0;
  qword_4F82978 = 0;
  qword_4F82980 = 0;
  dword_4F8296C = dword_4F8296C & 0x8000 | 1;
  qword_4F829B0 = 0x100000000LL;
  dword_4F82968 = v29;
  qword_4F82988 = 0;
  qword_4F82990 = 0;
  qword_4F82998 = 0;
  qword_4F829A0 = 0;
  qword_4F829A8 = (__int64)&unk_4F829B8;
  qword_4F829C0 = 0;
  qword_4F829C8 = (__int64)&unk_4F829E0;
  qword_4F829D0 = 1;
  dword_4F829D8 = 0;
  byte_4F829DC = 1;
  v30 = sub_C57470();
  v31 = (unsigned int)qword_4F829B0;
  v32 = (unsigned int)qword_4F829B0 + 1LL;
  if ( v32 > HIDWORD(qword_4F829B0) )
  {
    sub_C8D5F0((char *)&unk_4F829B8 - 16, &unk_4F829B8, v32, 8);
    v31 = (unsigned int)qword_4F829B0;
  }
  *(_QWORD *)(qword_4F829A8 + 8 * v31) = v30;
  LODWORD(qword_4F829B0) = qword_4F829B0 + 1;
  qword_4F829E8 = 0;
  qword_4F82960 = (__int64)&unk_49DAD08;
  qword_4F829F0 = 0;
  qword_4F829F8 = 0;
  qword_4F82A38 = (__int64)&unk_49DC350;
  qword_4F82A00 = 0;
  qword_4F82A58 = (__int64)nullsub_81;
  qword_4F82A08 = 0;
  qword_4F82A50 = (__int64)sub_BB8600;
  qword_4F82A10 = 0;
  byte_4F82A18 = 0;
  qword_4F82A20 = 0;
  qword_4F82A28 = 0;
  qword_4F82A30 = 0;
  sub_C53080(&qword_4F82960, "filter-passes", 13);
  BYTE1(dword_4F8296C) |= 2u;
  qword_4F82998 = (__int64)"pass names";
  qword_4F82988 = (__int64)"Only consider IR changes for passes whose names match the specified value. No-op without -print-changed";
  qword_4F829A0 = 10;
  qword_4F82990 = 103;
  LOBYTE(dword_4F8296C) = dword_4F8296C & 0x9F | 0x20;
  sub_C53130(&qword_4F82960);
  __cxa_atexit(sub_BB89D0, &qword_4F82960, &qword_4A427C0);
  qword_4F82860 = (__int64)&unk_49DC150;
  v33 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F82878 = 0;
  qword_4F82880 = 0;
  qword_4F82888 = 0;
  qword_4F82890 = 0;
  dword_4F8286C = dword_4F8286C & 0x8000 | 1;
  word_4F82870 = 0;
  qword_4F828B0 = 0x100000000LL;
  dword_4F82868 = v33;
  qword_4F82898 = 0;
  qword_4F828A0 = 0;
  qword_4F828A8 = (__int64)&unk_4F828B8;
  qword_4F828C0 = 0;
  qword_4F828C8 = (__int64)&unk_4F828E0;
  qword_4F828D0 = 1;
  dword_4F828D8 = 0;
  byte_4F828DC = 1;
  v34 = sub_C57470();
  v35 = (unsigned int)qword_4F828B0;
  v36 = (unsigned int)qword_4F828B0 + 1LL;
  if ( v36 > HIDWORD(qword_4F828B0) )
  {
    sub_C8D5F0((char *)&unk_4F828B8 - 16, &unk_4F828B8, v36, 8);
    v35 = (unsigned int)qword_4F828B0;
  }
  *(_QWORD *)(qword_4F828A8 + 8 * v35) = v34;
  LODWORD(qword_4F828B0) = qword_4F828B0 + 1;
  qword_4F828E8 = 0;
  qword_4F82860 = (__int64)&unk_49DAD08;
  qword_4F828F0 = 0;
  qword_4F828F8 = 0;
  qword_4F82938 = (__int64)&unk_49DC350;
  qword_4F82900 = 0;
  qword_4F82958 = (__int64)nullsub_81;
  qword_4F82908 = 0;
  qword_4F82950 = (__int64)sub_BB8600;
  qword_4F82910 = 0;
  byte_4F82918 = 0;
  qword_4F82920 = 0;
  qword_4F82928 = 0;
  qword_4F82930 = 0;
  sub_C53080(&qword_4F82860, "filter-print-funcs", 18);
  BYTE1(dword_4F8286C) |= 2u;
  qword_4F82898 = (__int64)"function names";
  qword_4F82888 = (__int64)"Only print IR for functions whose name match this for all print-[before|after][-all] options";
  qword_4F828A0 = 14;
  qword_4F82890 = 92;
  LOBYTE(dword_4F8286C) = dword_4F8286C & 0x9F | 0x20;
  sub_C53130(&qword_4F82860);
  return __cxa_atexit(sub_BB89D0, &qword_4F82860, &qword_4A427C0);
}
