// Function: sub_85C120
// Address: 0x85c120
//
__int64 *__fastcall sub_85C120(
        unsigned int a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        char *a10,
        __int64 *a11,
        __int64 a12,
        unsigned int a13)
{
  int v13; // r10d
  unsigned __int64 v15; // rbx
  __int64 v17; // rdi
  int v18; // edx
  __int64 v19; // r12
  __int64 v20; // rax
  int v21; // r8d
  int v22; // edx
  int v23; // ecx
  __int64 v24; // rax
  int v25; // eax
  int v26; // esi
  __int64 v27; // rdi
  char v28; // al
  bool v29; // al
  char v30; // dl
  bool v31; // sf
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rcx
  int v35; // eax
  __int64 v36; // rax
  __int16 v37; // dx
  __int64 v38; // rdx
  int v39; // eax
  char v40; // r10
  int v41; // edi
  char v42; // dl
  int v43; // edx
  __int64 v44; // rsi
  int v45; // eax
  bool v46; // zf
  int v48; // eax
  char v49; // al
  char v50; // cl
  unsigned __int8 v51; // dl
  __int64 v52; // rdx
  __int64 j; // rax
  bool v54; // dl
  unsigned __int8 v55; // al
  __int64 v56; // rax
  __int64 k; // rax
  __int64 v58; // r12
  __int64 v59; // rax
  int v60; // edi
  __int64 v61; // rax
  int v62; // r9d
  int v63; // edi
  __int64 i; // rax
  int v65; // edi
  __int64 *v66; // rax
  char *v67; // rdx
  int v68; // eax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rdx
  int v72; // eax
  int v73; // edx
  __int64 v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rsi
  char v78; // si
  char v79; // al
  char v80; // si
  __int64 v81; // rax
  char v82; // si
  __int64 v83; // rax
  char v84; // r8
  __int64 *v85; // rax
  __int64 v86; // rdi
  __int64 v87; // rax
  char v88; // dl
  int v89; // edx
  __int64 *v90; // rax
  __int64 *v91; // rax
  __int64 *v92; // rax
  char v93; // cl
  int v94; // r10d
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  _BOOL4 v98; // eax
  _BYTE *v99; // rax
  int v100; // eax
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rax
  char v104; // dl
  __int64 v105; // rdx
  char v106; // [rsp+18h] [rbp-B8h]
  __int64 v107; // [rsp+18h] [rbp-B8h]
  int v108; // [rsp+20h] [rbp-B0h]
  char v109; // [rsp+20h] [rbp-B0h]
  __int64 v110; // [rsp+28h] [rbp-A8h]
  int v111; // [rsp+28h] [rbp-A8h]
  __int64 v112; // [rsp+28h] [rbp-A8h]
  _BOOL4 v113; // [rsp+38h] [rbp-98h]
  __int64 v114; // [rsp+38h] [rbp-98h]
  __int64 v115; // [rsp+38h] [rbp-98h]
  char *v116; // [rsp+48h] [rbp-88h]
  char v117; // [rsp+56h] [rbp-7Ah]
  char v118; // [rsp+57h] [rbp-79h]
  bool v122; // [rsp+88h] [rbp-48h]
  __int64 v123; // [rsp+88h] [rbp-48h]
  __int64 v124; // [rsp+88h] [rbp-48h]
  __int64 v125; // [rsp+88h] [rbp-48h]
  __int64 v126; // [rsp+88h] [rbp-48h]
  __int64 v127; // [rsp+88h] [rbp-48h]
  __int64 v128; // [rsp+88h] [rbp-48h]
  __int64 v129; // [rsp+88h] [rbp-48h]
  __int64 v130; // [rsp+88h] [rbp-48h]
  __int64 v131; // [rsp+88h] [rbp-48h]
  __int64 v132; // [rsp+88h] [rbp-48h]
  __int64 v133; // [rsp+88h] [rbp-48h]
  char v134; // [rsp+88h] [rbp-48h]
  bool v135; // [rsp+90h] [rbp-40h]
  __int64 v136; // [rsp+90h] [rbp-40h]
  __int64 v137; // [rsp+90h] [rbp-40h]
  char v138; // [rsp+90h] [rbp-40h]
  char v139; // [rsp+90h] [rbp-40h]
  char v140; // [rsp+90h] [rbp-40h]
  char v141; // [rsp+90h] [rbp-40h]
  __int64 v142; // [rsp+90h] [rbp-40h]
  __int64 v143; // [rsp+98h] [rbp-38h]
  int v144; // [rsp+98h] [rbp-38h]
  int v145; // [rsp+98h] [rbp-38h]
  int v146; // [rsp+98h] [rbp-38h]
  int v147; // [rsp+98h] [rbp-38h]
  __int64 v148; // [rsp+98h] [rbp-38h]
  __int64 v149; // [rsp+98h] [rbp-38h]

  v13 = a2;
  v15 = a1;
  v17 = qword_4F04C68[0];
  v118 = 0;
  v116 = a10;
  if ( unk_4F04C48 != -1 )
    v118 = *(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1;
  v18 = dword_4F04C64 + 1;
  if ( dword_4F04C64 + 1 == (_DWORD)dword_4D049E0 )
  {
    v58 = dword_4D049E0 + 30;
    v137 = a3;
    v59 = sub_822C60(
            (void *)qword_4F04C68[0],
            776 * (dword_4D049E0 + 30) - 23280,
            776 * (dword_4D049E0 + 30),
            dword_4D049E0,
            a5,
            a6);
    a3 = v137;
    v13 = a2;
    v17 = v59;
    qword_4F04C68[0] = v59;
    dword_4D049E0 = v58;
    v18 = dword_4F04C64 + 1;
  }
  dword_4F04C64 = v18;
  v19 = v17 + 776LL * v18;
  *(_QWORD *)v19 = 0;
  *(_QWORD *)(v19 + 768) = 0;
  memset(
    (void *)((v19 + 8) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v19 - (((_DWORD)v19 + 8) & 0xFFFFFFF8) + 776) >> 3));
  if ( v13 != -1 )
  {
    if ( (unsigned __int8)v15 > 0x11u )
      goto LABEL_12;
    v20 = 164102;
    if ( _bittest64(&v20, v15) )
    {
LABEL_8:
      *(_DWORD *)v19 = v13;
      goto LABEL_9;
    }
  }
  if ( (unsigned __int8)v15 <= 9u )
  {
    v24 = 689;
    if ( _bittest64(&v24, v15) )
      goto LABEL_8;
  }
LABEL_12:
  v143 = a3;
  v25 = sub_880E90();
  a3 = v143;
  *(_DWORD *)v19 = v25;
LABEL_9:
  v21 = a13 & 8;
  v22 = dword_4F07270[0];
  v23 = (a13 >> 20) & 1;
  *(_DWORD *)(v19 + 196) = dword_4F07270[0];
  switch ( (char)v15 )
  {
    case 0:
      v125 = a3;
      a11 = *(__int64 **)(qword_4D03FF0 + 8);
      *((_DWORD *)a11 + 60) = dword_4F04C64;
      sub_7296B0(dword_4F073B8[0]);
      v21 = a13 & 8;
      v27 = 0;
      a3 = v125;
      *(_DWORD *)(v19 + 192) = dword_4F07270[0];
      *(_DWORD *)(v19 + 4) = *(_DWORD *)(v19 + 4) & 0xFFEFF700
                           | ((_WORD)dword_4F04C38 << 11) & 0x800
                           | (((a13 & 0x100000) != 0) << 20);
      goto LABEL_163;
    case 1:
    case 9:
    case 10:
      v60 = dword_4F073B8[0];
      if ( v22 != dword_4F073B8[0] )
      {
        v123 = a3;
        sub_7296B0(dword_4F073B8[0]);
        v60 = dword_4F073B8[0];
        a3 = v123;
        LOBYTE(v23) = (a13 & 0x100000) != 0;
        v21 = a13 & 8;
      }
      *(_DWORD *)(v19 + 192) = v60;
      a11 = 0;
      v27 = 0;
      goto LABEL_141;
    case 2:
    case 15:
      goto LABEL_16;
    case 3:
    case 4:
      v62 = dword_4F073B8[0];
      if ( v22 != dword_4F073B8[0] )
      {
        v124 = a3;
        sub_7296B0(dword_4F073B8[0]);
        a3 = v124;
        LOBYTE(v23) = (a13 & 0x100000) != 0;
        v21 = a13 & 8;
        v62 = dword_4F073B8[0];
      }
      *(_DWORD *)(v19 + 192) = v62;
      if ( (_BYTE)v15 == 3 )
      {
        v131 = a3;
        v140 = v23;
        v146 = v21;
        v91 = (__int64 *)sub_726EB0(3, *(_DWORD *)v19, 0);
        v21 = v146;
        LOBYTE(v23) = v140;
        v27 = 1;
        a11 = v91;
        a3 = v131;
        v91[4] = a5;
        *(_QWORD *)(a5 + 128) = v91;
        v62 = dword_4F073B8[0];
      }
      else
      {
        v27 = 0;
        a11 = *(__int64 **)(a5 + 128);
      }
      *(_DWORD *)(v19 + 192) = v62;
LABEL_141:
      *(_BYTE *)(v19 + 4) = v15;
      *(_BYTE *)(v19 + 5) = (8 * (dword_4F04C38 & 1)) | *(_BYTE *)(v19 + 5) & 0xF7;
      *(_BYTE *)(v19 + 6) = *(_BYTE *)(v19 + 6) & 0xEF | (16 * v23);
      if ( (_BYTE)v15 )
      {
        v26 = dword_4F04C64;
        v29 = 0;
      }
      else
      {
LABEL_163:
        v26 = dword_4F04C64;
        v29 = unk_4D04738 != 0;
      }
      goto LABEL_18;
    case 6:
      v63 = dword_4F073B8[0];
      if ( v22 != dword_4F073B8[0] )
      {
        v126 = a3;
        sub_7296B0(dword_4F073B8[0]);
        v63 = dword_4F073B8[0];
        a3 = v126;
        LOBYTE(v23) = (a13 & 0x100000) != 0;
        v21 = a13 & 8;
      }
      *(_DWORD *)(v19 + 192) = v63;
      for ( i = a3; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      *(_DWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 96LL) = *(_DWORD *)v19;
      a11 = *(__int64 **)(*(_QWORD *)(a3 + 168) + 152LL);
      if ( a11 )
      {
        *((_BYTE *)a11 + 29) &= ~0x20u;
        *((_DWORD *)a11 + 6) = *(_DWORD *)v19;
      }
      else
      {
        v132 = a3;
        v141 = v23;
        v147 = v21;
        v92 = (__int64 *)sub_726EB0(6, *(_DWORD *)v19, 0);
        v21 = v147;
        LOBYTE(v23) = v141;
        a3 = v132;
        a11 = v92;
      }
      v26 = dword_4F04C64;
      v27 = 1;
      *((_DWORD *)a11 + 60) = dword_4F04C64;
      goto LABEL_17;
    case 8:
      a11 = 0;
      if ( dword_4F07590 && (a13 & 0x200) == 0 )
      {
        v129 = a3;
        v85 = (__int64 *)sub_726EB0(8, *(_DWORD *)v19, 0);
        a3 = v129;
        LOBYTE(v23) = (a13 & 0x100000) != 0;
        v21 = a13 & 8;
        a11 = v85;
      }
LABEL_16:
      v26 = dword_4F04C64;
      v27 = 0;
      *(_DWORD *)(v19 + 192) = *(_DWORD *)(v19 - 584);
      goto LABEL_17;
    case 16:
      v65 = dword_4F073B8[0];
      if ( v22 != dword_4F073B8[0] )
      {
        v127 = a3;
        sub_7296B0(dword_4F073B8[0]);
        v65 = dword_4F073B8[0];
        a3 = v127;
        LOBYTE(v23) = (a13 & 0x100000) != 0;
        v21 = a13 & 8;
      }
      *(_DWORD *)(v19 + 192) = v65;
      v128 = a3;
      v138 = v23;
      v144 = v21;
      v66 = (__int64 *)sub_726EB0(16, *(_DWORD *)v19, 0);
      v26 = dword_4F04C64;
      v21 = v144;
      v27 = 1;
      LOBYTE(v23) = v138;
      a3 = v128;
      a11 = v66;
      *((_DWORD *)v66 + 60) = dword_4F04C64;
      goto LABEL_17;
    case 17:
      if ( (a13 & 8) != 0 )
      {
        v27 = 0;
        v26 = dword_4F04C64;
        v67 = (char *)qword_4F072B8 + 16 * *(int *)(a4 + 160);
        v68 = *((_DWORD *)v67 + 2);
        dword_4F07270[0] = v68;
        a11 = *(__int64 **)v67;
        goto LABEL_176;
      }
      if ( (*(_BYTE *)(a4 + 89) & 4) == 0 )
        goto LABEL_320;
      v86 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 32LL);
      if ( !v86 )
        goto LABEL_320;
      if ( (*(_BYTE *)(v86 + 89) & 2) != 0 )
      {
        v133 = a3;
        v87 = sub_72F070(v86);
        v21 = a13 & 8;
        LOBYTE(v23) = (a13 & 0x100000) != 0;
        a3 = v133;
      }
      else
      {
        v87 = *(_QWORD *)(v86 + 40);
      }
      break;
    default:
      v26 = dword_4F04C64;
      v27 = 0;
      a11 = 0;
      *(_DWORD *)(v19 + 192) = *(_DWORD *)(v19 - 584);
      goto LABEL_17;
  }
  while ( 1 )
  {
    if ( !v87 )
    {
LABEL_320:
      v89 = 0;
      goto LABEL_321;
    }
    v88 = *(_BYTE *)(v87 + 28);
    if ( v88 == 17 )
      break;
    if ( v88 != 15 && v88 != 2 )
      goto LABEL_320;
    v87 = *(_QWORD *)(v87 + 16);
  }
  v89 = *((_DWORD *)qword_4F072B8 + 4 * *(int *)(*(_QWORD *)(v87 + 32) + 160LL) + 2);
LABEL_321:
  v130 = a3;
  v139 = v23;
  v145 = v21;
  v90 = sub_729790(*(_DWORD *)v19, a4, v89);
  v26 = dword_4F04C64;
  v27 = 1;
  a11 = v90;
  v21 = v145;
  LOBYTE(v23) = v139;
  *((_DWORD *)v90 + 60) = dword_4F04C64;
  a3 = v130;
  v68 = dword_4F07270[0];
LABEL_176:
  *(_DWORD *)(v19 + 192) = v68;
LABEL_17:
  v28 = *(_BYTE *)(v19 + 5);
  *(_BYTE *)(v19 + 4) = v15;
  *(_BYTE *)(v19 + 5) = (8 * (dword_4F04C38 & 1)) | v28 & 0xF7;
  v29 = 0;
  *(_BYTE *)(v19 + 6) = *(_BYTE *)(v19 + 6) & 0xEF | (16 * v23);
LABEL_18:
  v122 = (_BYTE)v15 == 0;
  v117 = (v15 - 9) & 0xF7;
  *(_QWORD *)(v19 + 7) = *(_QWORD *)(v19 + 7) & 0xFBFBFFDFFF3FFFFDLL
                       | ((unsigned __int64)(HIWORD(a13) & 1) << 50)
                       | ((unsigned __int64)((a13 >> 10) & 1) << 37)
                       | ((unsigned __int64)(a13 & 1) << 23)
                       | (2LL * v29)
                       | ((unsigned __int64)((a13 >> 21) & 1) << 58);
  if ( !(_BYTE)v15 || (_BYTE)v15 == 9 || (v30 = (_BYTE)v15 == 13 || (_BYTE)v15 == 10) != 0 )
  {
    *(_BYTE *)(v19 + 12) = *(_BYTE *)(v19 + 12) & 0xF5 | (2 * (unk_4D0482C & 1));
  }
  else
  {
    v31 = *(char *)(v19 - 763) < 0;
    *(_WORD *)(v19 + 11) = *(_WORD *)(v19 + 11) & 0xF07F
                         | (((*(_BYTE *)(v19 - 764) & 4) != 0) << 10)
                         | (((*(_BYTE *)(v19 - 764) & 2) != 0) << 9)
                         | *(_BYTE *)(v19 - 765) & 0x80
                         | ((*(_BYTE *)(v19 - 764) & 1) << 8)
                         | (((*(_BYTE *)(v19 - 764) & 8) != 0) << 11);
    if ( v31 )
    {
      v30 = 1;
    }
    else if ( a5 )
    {
      v30 = (*(_BYTE *)(a5 + 124) & 0x20) != 0;
    }
    *(_BYTE *)(v19 + 13) = *(_BYTE *)(v19 + 13) & 0x7F | (v30 << 7);
    if ( (unsigned __int8)(v15 - 6) > 1u && (_BYTE)v15 != 17 )
      *(_WORD *)(v19 + 13) = *(_WORD *)(v19 + 13) & 0xFDDF
                           | *(_BYTE *)(v19 - 763) & 0x20
                           | (((*(_BYTE *)(v19 - 762) & 2) != 0) << 9);
  }
  *(_BYTE *)(v19 + 12) = *(_BYTE *)(v19 + 12) & 0x8F | (16 * ((a13 & 0x1000) != 0));
  if ( v26 > 0 && *(int *)(v19 - 576) > 0 )
    *(_DWORD *)(v19 + 200) = 1;
  *(_QWORD *)(v19 + 184) = a11;
  *(_QWORD *)(v19 + 208) = a3;
  *(_QWORD *)(v19 + 216) = a4;
  *(_QWORD *)(v19 + 224) = a5;
  *(_DWORD *)(v19 + 7) = *(_DWORD *)(v19 + 7) & 0xFBFFFFEF | ((v21 != 0) << 26) | (16 * (_BYTE)dword_4F04C3C) & 0x10;
  if ( !(_BYTE)v15 && (*(_BYTE *)(v19 + 10) & 4) != 0 )
  {
    *(_QWORD *)(v19 + 328) = *(_QWORD *)(qword_4F07288 + 256);
    v32 = qword_4D03FF0;
    *(_QWORD *)(v19 + 336) = *(_QWORD *)(qword_4D03FF0 + 136);
    *(_QWORD *)(v32 + 136) = 0;
  }
  v33 = a7;
  *(_QWORD *)(v19 + 384) = 0;
  *(_DWORD *)(v19 + 452) = -1;
  v34 = (int)dword_4F04C5C;
  *(_QWORD *)(v19 + 368) = a7;
  v35 = dword_4F04C44;
  *(_DWORD *)(v19 + 344) = v34;
  *(_DWORD *)(v19 + 348) = v35;
  *(_DWORD *)(v19 + 352) = unk_4F04C48;
  *(_QWORD *)(v19 + 360) = a6;
  *(_QWORD *)(v19 + 376) = a8;
  *(_QWORD *)(v19 + 392) = *(_QWORD *)&dword_4F063F8;
  *(_DWORD *)(v19 + 400) = dword_4F04C58;
  *(_QWORD *)(v19 + 408) = a9;
  *(_DWORD *)(v19 + 448) = unk_4F04C2C;
  *(_DWORD *)(v19 + 472) = dword_4F04C40;
  v36 = qword_4D03C50;
  *(_QWORD *)(v19 + 552) = -1;
  *(_DWORD *)(v19 + 560) = -1;
  *(_QWORD *)(v19 + 480) = v36;
  *(_QWORD *)(v19 + 572) = -1;
  v37 = unk_4F06C5A & 3;
  *(_QWORD *)(v19 + 496) = qword_4F06BC0;
  *(_DWORD *)(v19 + 520) = dword_4F04C34;
  *(_DWORD *)(v19 + 564) = dword_4F04C60;
  LOWORD(v36) = ((unk_4F06C58 & 3) << 8) | (16 * v37) | ((unk_4F06C59 & 3) << 6);
  v38 = *(_WORD *)(v19 + 10) & 0xFC0F;
  *(_WORD *)(v19 + 10) = v38 | v36;
  if ( a7 && (*(_BYTE *)(a7 + 85) & 1) != 0 )
    ++*(_DWORD *)(v19 + 200);
  if ( !a11 )
  {
    if ( (_BYTE)v15 != 10 )
      goto LABEL_42;
LABEL_99:
    *(_DWORD *)(v19 + 552) = 0;
    v136 = a3;
    sub_833810(v27, v33, v38, v34, (__m128i *)&dword_4F077C4, (__int64 *)&dword_4F04C38);
    a3 = v136;
    if ( (a13 & 0x80000) != 0 )
    {
      unk_4F04C48 = -1;
      dword_4F04C44 = -1;
      dword_4F04C40 = -1;
      unk_4F04C2C = -1;
    }
    v39 = dword_4F04C64;
    dword_4F04C60 = dword_4F04C64;
    v40 = v136 != 0;
    v135 = a11 != 0;
    if ( !a3 || !a11 )
    {
      v41 = dword_4F077C4;
      if ( dword_4F077C4 != 2 )
        goto LABEL_264;
      goto LABEL_104;
    }
    goto LABEL_46;
  }
  if ( (_DWORD)v27 )
  {
    v33 = *(_QWORD *)(v19 + 184);
    if ( dword_4F077C4 != 2 )
    {
      v34 = qword_4F04C68[0] + 776 * v34;
      v27 = *(_QWORD *)(v34 + 184);
      *(_QWORD *)(v33 + 16) = v27;
      if ( (*(_BYTE *)(v33 - 8) & 1) != 0 && (*(_BYTE *)(v27 - 8) & 1) == 0 )
      {
        v114 = a3;
        sub_72EDB0(v27, v33, 0x17u, qword_4F04C50);
        a3 = v114;
        *(_QWORD *)(v33 + 16) = 0;
      }
      goto LABEL_39;
    }
    v55 = *(_BYTE *)(v33 + 28);
    if ( v55 == 16 )
      goto LABEL_211;
    if ( v55 > 0x10u )
    {
      if ( v55 == 17 )
      {
        v76 = *(_QWORD *)(v19 + 216);
        v38 = *(_QWORD *)(v76 + 40);
        if ( (*(_BYTE *)(v76 + 89) & 4) != 0 )
        {
          *(_QWORD *)(v33 + 16) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v38 + 32) + 168LL) + 152LL);
          goto LABEL_39;
        }
        if ( v38 && *(_BYTE *)(v38 + 28) == 3 )
        {
          *(_QWORD *)(v33 + 16) = *(_QWORD *)(*(_QWORD *)(v38 + 32) + 128LL);
          goto LABEL_39;
        }
LABEL_123:
        *(_QWORD *)(v33 + 16) = *(_QWORD *)(qword_4F04C68[0] + 184LL);
        goto LABEL_39;
      }
    }
    else
    {
      if ( v55 == 3 )
      {
        v56 = *(_QWORD *)(*(_QWORD *)(v19 + 224) + 40LL);
        if ( !v56 || *(_BYTE *)(v56 + 28) != 3 )
          goto LABEL_123;
LABEL_324:
        *(_QWORD *)(v33 + 16) = *(_QWORD *)(*(_QWORD *)(v56 + 32) + 128LL);
        goto LABEL_39;
      }
      if ( v55 == 6 )
      {
LABEL_211:
        v75 = *(_QWORD *)(v19 + 208);
        v27 = *(unsigned __int8 *)(v75 + 89);
        v56 = *(_QWORD *)(v75 + 40);
        if ( (v27 & 4) != 0 )
        {
          *(_QWORD *)(v33 + 16) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v56 + 32) + 168LL) + 152LL);
          goto LABEL_39;
        }
        if ( !v56 || *(_BYTE *)(v56 + 28) != 3 )
        {
          v27 &= 1u;
          v38 = qword_4F04C68[0];
          if ( (_DWORD)v27 )
          {
            v27 = qword_4F04C68[0] + 776LL * (int)v34;
            v93 = *(_BYTE *)(v27 + 4);
            if ( v93 == 1 )
            {
              v115 = a3;
              v99 = sub_732EF0(v27);
              a3 = v115;
              *(_QWORD *)(v33 + 16) = v99;
            }
            else
            {
              v94 = *(_DWORD *)(v27 + 400);
              v95 = v94;
              if ( ((v93 - 15) & 0xFD) != 0 && v93 != 2 )
              {
                while ( 1 )
                {
                  if ( (_DWORD)v95 == -1 )
                    v95 = *(int *)(v27 + 552);
                  v97 = 776 * v95;
                  v27 = qword_4F04C68[0] + v97;
                  if ( *(_BYTE *)(qword_4F04C68[0] + v97 + 4) == 17 )
                    break;
                  v95 = *(int *)(v27 + 400);
                }
                v96 = qword_4F04C68[0] + v97;
              }
              else
              {
                v96 = qword_4F04C68[0] + 776LL * v94;
              }
              v112 = a3;
              v142 = *(_QWORD *)(v96 + 184);
              v27 = (__int64)sub_732EF0(v27);
              sub_72EDB0(v27, v33, 0x17u, v142);
              a3 = v112;
            }
          }
          else
          {
            *(_QWORD *)(v33 + 16) = *(_QWORD *)(qword_4F04C68[0] + 184LL);
          }
          goto LABEL_39;
        }
        goto LABEL_324;
      }
    }
    sub_721090();
  }
LABEL_39:
  if ( *((_DWORD *)a11 + 60) == -1 )
    *((_DWORD *)a11 + 60) = dword_4F04C64;
  if ( (_BYTE)v15 == 10 )
    goto LABEL_99;
LABEL_42:
  v39 = dword_4F04C64;
  if ( dword_4F04C64 )
    *(_DWORD *)(v19 + 552) = dword_4F04C60;
  dword_4F04C60 = v39;
  v40 = a3 != 0;
  v135 = a11 != 0;
  if ( a3 && a11 )
  {
LABEL_46:
    a11[4] = a3;
    v40 = 1;
    v135 = 1;
  }
  v41 = dword_4F077C4;
  if ( !a6 || (_BYTE)v15 != 9 )
  {
    if ( (_BYTE)v15 == 13 )
    {
      if ( dword_4F077C4 == 2 )
      {
        v113 = 0;
        v43 = 0;
        goto LABEL_229;
      }
LABEL_72:
      *(_DWORD *)(v19 + 5) = *(_DWORD *)(v19 + 5) & 0xFFFDFFFB
                           | *(_BYTE *)(v19 - 771) & 4
                           | (((*(_BYTE *)(v19 - 769) & 2) != 0) << 17);
      goto LABEL_73;
    }
    v41 = dword_4F077C4;
    if ( dword_4F077C4 != 2 )
    {
      if ( (_BYTE)v15 == 6 )
      {
LABEL_277:
        if ( !(_BYTE)v15 )
          goto LABEL_57;
        goto LABEL_72;
      }
      goto LABEL_264;
    }
    if ( (unsigned __int8)v15 > 0xEu )
    {
LABEL_264:
      v113 = 0;
      goto LABEL_50;
    }
LABEL_104:
    if ( ((0x4EA0uLL >> v15) & 1) != 0 )
      goto LABEL_105;
    v113 = 0;
    v41 = 2;
LABEL_50:
    v43 = 0;
    if ( dword_4F04C5C < unk_4F04C48 )
    {
      v44 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368);
      if ( v44 )
      {
        if ( *(_BYTE *)(v44 + 80) != 9 )
          v43 = 1 - v113;
      }
    }
    dword_4F04C5C = v39;
    if ( !(_BYTE)v15 )
    {
      if ( (*(_BYTE *)(v19 + 10) & 4) == 0 )
        goto LABEL_56;
      goto LABEL_69;
    }
    if ( v41 == 2 )
      goto LABEL_106;
    goto LABEL_277;
  }
  v42 = *(_BYTE *)(a6 + 80);
  v113 = v42 == 3;
  if ( dword_4F077C4 != 2 )
    goto LABEL_50;
  v77 = 20128;
  if ( !_bittest64(&v77, v15) )
    goto LABEL_50;
  if ( v42 != 3 )
  {
LABEL_105:
    v113 = 0;
    v43 = 0;
    if ( (_BYTE)v15 )
      goto LABEL_106;
    v43 = 0;
    if ( (*(_BYTE *)(v19 + 10) & 4) == 0 )
      goto LABEL_226;
    v41 = 2;
    v43 = 0;
LABEL_69:
    *(_BYTE *)(v19 + 5) |= 4u;
LABEL_56:
    if ( v41 != 2 )
      goto LABEL_57;
    goto LABEL_226;
  }
  v113 = 1;
  v43 = a13 & 6;
  if ( (a13 & 6) != 0 )
    goto LABEL_50;
LABEL_106:
  if ( (((_BYTE)v15 - 7) & 0xFD) == 0 )
  {
    *(_BYTE *)(v19 + 5) |= 4u;
    if ( !(_BYTE)v15 )
      goto LABEL_108;
    goto LABEL_231;
  }
LABEL_229:
  if ( (unsigned __int8)(v15 - 4) > 1u )
  {
    if ( (_BYTE)v15 == 6 )
    {
      if ( **(_QWORD **)(a3 + 168) )
      {
        v78 = *(_BYTE *)(v19 - 769);
        *(_DWORD *)(v19 + 5) = *(_DWORD *)(v19 + 5) & 0xFFFDFFFB | (((v78 & 2) != 0) << 17) | 4;
        goto LABEL_253;
      }
      goto LABEL_227;
    }
LABEL_226:
    if ( !(_BYTE)v15 )
      goto LABEL_108;
LABEL_227:
    v78 = *(_BYTE *)(v19 - 769);
    *(_DWORD *)(v19 + 5) = *(_BYTE *)(v19 - 771) & 4 | (((v78 & 2) != 0) << 17) | *(_DWORD *)(v19 + 5) & 0xFFFDFFFB;
    goto LABEL_232;
  }
  *(_BYTE *)(v19 + 5) |= 4u;
LABEL_231:
  v78 = *(_BYTE *)(v19 - 769);
  *(_BYTE *)(v19 + 7) = v78 & 2 | *(_BYTE *)(v19 + 7) & 0xFD;
LABEL_232:
  if ( (_BYTE)v15 == 7 )
  {
    v106 = v40;
    v108 = v43;
    v110 = a3;
    *(_BYTE *)(v19 + 8) = (4 * (sub_8D23B0(a3) & 1)) | *(_BYTE *)(v19 + 8) & 0xFB;
    *(_BYTE *)(v19 + 14) = ((a13 & 0x400000) != 0) | *(_BYTE *)(v19 + 14) & 0xFE;
    a3 = v110;
    v43 = v108;
    v40 = v106;
LABEL_234:
    *(_BYTE *)(v19 + 5) = *(_BYTE *)(v19 - 771) & 0x80 | *(_BYTE *)(v19 + 5) & 0x7F;
LABEL_235:
    ++unk_4F04C28;
    if ( dword_4F04C58 != -1 )
    {
      *(_BYTE *)(v19 + 5) |= 8u;
      dword_4F04C38 = 1;
    }
    v79 = *(_BYTE *)(v19 - 770);
    *(_BYTE *)(v19 + 6) = *(_BYTE *)(v19 + 6) & 0xBF | ((*(_BYTE *)(a3 + 178) & 1 | ((v79 & 0x40) != 0)) << 6);
    if ( (_BYTE)v15 == 9 )
    {
LABEL_238:
      v80 = 1;
      if ( unk_4F04C48 == -1 )
        v80 = (unsigned __int8)~*(_BYTE *)(v19 + 9) >> 7;
      v107 = a3;
      v109 = v40;
      v111 = v43;
      *(_BYTE *)(v19 + 10) = v80 | *(_BYTE *)(v19 + 10) & 0xFE;
      unk_4F04C48 = dword_4F04C64;
      sub_85BC50(*(_QWORD **)a9, a8);
      *(_DWORD *)(v19 + 400) = -1;
      v40 = v109;
      dword_4F04C58 = -1;
      a3 = v107;
      qword_4F04C50 = 0;
      *(_BYTE *)(v19 + 6) = *(_BYTE *)(v19 + 6) & 0xD1
                          | (32 * ((a13 & 0x20000) != 0))
                          | (8 * ((a13 & 0x2000) != 0))
                          | (4 * ((a13 & 4) != 0))
                          | (2 * ((a13 & 2) != 0));
      if ( a7 )
      {
        if ( (*(_BYTE *)(v19 + 12) & 0x10) == 0 )
        {
          switch ( *(_BYTE *)(a7 + 80) )
          {
            case 4:
            case 5:
              v81 = *(_QWORD *)(*(_QWORD *)(a7 + 96) + 80LL);
              break;
            case 6:
              v81 = *(_QWORD *)(*(_QWORD *)(a7 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v81 = *(_QWORD *)(*(_QWORD *)(a7 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v81 = *(_QWORD *)(a7 + 88);
              break;
            default:
              BUG();
          }
          v82 = 1;
          if ( (*(_BYTE *)(v81 + 160) & 8) == 0 )
            v82 = *(_BYTE *)(v19 - 769) & 1;
          *(_BYTE *)(v19 + 7) = v82 | *(_BYTE *)(v19 + 7) & 0xFE;
        }
        if ( *(_BYTE *)(a7 + 80) == 9 )
          goto LABEL_249;
      }
      if ( v113 )
      {
LABEL_249:
        *(_QWORD *)(v19 + 640) = *(_QWORD *)(a9 + 56);
        *(_QWORD *)(v19 + 648) = *(_QWORD *)(a9 + 72);
      }
      else
      {
        *(_QWORD *)(v19 + 640) = *(_QWORD *)(a9 + 56);
        *(_QWORD *)(v19 + 648) = *(_QWORD *)(a9 + 72);
        if ( !v111 )
          goto LABEL_92;
      }
      goto LABEL_109;
    }
    v78 = *(_BYTE *)(v19 - 769);
    goto LABEL_262;
  }
LABEL_253:
  if ( (unsigned __int8)(v15 - 8) <= 2u || (_BYTE)v15 == 13 )
  {
    *(_BYTE *)(v19 + 5) |= 0x80u;
    if ( (unsigned __int8)(v15 - 6) <= 1u )
      goto LABEL_235;
    if ( !(_BYTE)v15 )
      goto LABEL_108;
    v79 = *(_BYTE *)(v19 - 770);
    *(_BYTE *)(v19 + 6) |= v79 & 0x40;
    if ( (_BYTE)v15 == 9 )
      goto LABEL_238;
  }
  else
  {
    if ( (_BYTE)v15 == 6 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(a3 + 168) + 109LL) & 0x20) != 0 )
      {
        *(_BYTE *)(v19 + 5) |= 0x80u;
        goto LABEL_235;
      }
      goto LABEL_234;
    }
    if ( !(_BYTE)v15 )
    {
LABEL_108:
      if ( !v43 )
        goto LABEL_73;
      goto LABEL_109;
    }
    *(_BYTE *)(v19 + 5) = *(_BYTE *)(v19 - 771) & 0x80 | *(_BYTE *)(v19 + 5) & 0x7F;
    if ( (unsigned __int8)(v15 - 6) <= 1u )
      goto LABEL_235;
    v79 = *(_BYTE *)(v19 - 770);
    *(_BYTE *)(v19 + 6) |= v79 & 0x40;
  }
  if ( !(_BYTE)v15 || (_BYTE)v15 == 10 )
    goto LABEL_108;
  if ( (_BYTE)v15 != 17 || (*(_DWORD *)(a4 + 192) & 0x8001000) != 0x1000 )
  {
LABEL_262:
    *(_QWORD *)(v19 + 6) = *(_QWORD *)(v19 + 6) & 0xFFEFFFFFFFFFFED1LL
                         | ((unsigned __int64)((*(_BYTE *)(v19 - 764) & 0x10) != 0) << 52)
                         | (32LL * ((v79 & 0x20) != 0))
                         | (2LL * ((v79 & 2) != 0))
                         | (4LL * ((v79 & 4) != 0))
                         | (8LL * ((v79 & 8) != 0))
                         | ((unsigned __int64)(v78 & 1) << 8);
    goto LABEL_108;
  }
  *(_WORD *)(v19 + 6) &= 0xFED1u;
  if ( !v43 )
  {
LABEL_74:
    v48 = dword_4F04C64;
    v46 = dword_4F077C4 == 2;
    *(_DWORD *)(v19 + 400) = dword_4F04C64;
    dword_4F04C58 = v48;
    qword_4F04C50 = a11;
    if ( v46 )
    {
      v49 = v15 - 3;
      v50 = v122 || (_BYTE)v15 == 11;
LABEL_76:
      *(_BYTE *)(v19 + 9) = *(_BYTE *)(v19 + 9) & 0xE1 | *(_BYTE *)(v19 - 767) & 0x1E;
      v51 = v15 - 6;
      goto LABEL_77;
    }
LABEL_58:
    *(_BYTE *)(v19 + 9) = *(_BYTE *)(v19 + 9) & 0xE1 | 6;
    goto LABEL_59;
  }
LABEL_109:
  *(_BYTE *)(v19 + 5) |= 0x10u;
LABEL_73:
  if ( (_BYTE)v15 == 17 )
    goto LABEL_74;
  if ( !(_BYTE)v15 )
  {
LABEL_57:
    v45 = dword_4F04C64;
    v46 = dword_4F077C4 == 2;
    *(_DWORD *)(v19 + 520) = dword_4F04C64;
    dword_4F04C34 = v45;
    if ( v46 )
    {
      v50 = 1;
      *(_BYTE *)(v19 + 9) = *(_BYTE *)(v19 + 9) & 0xE1 | 4;
      v49 = -3;
      goto LABEL_205;
    }
    goto LABEL_58;
  }
LABEL_92:
  v49 = v15 - 3;
  if ( dword_4F077C4 != 2 )
  {
    if ( (unsigned __int8)v49 <= 1u )
    {
      *(_DWORD *)(v19 + 400) = -1;
      dword_4F04C58 = -1;
      qword_4F04C50 = 0;
    }
    goto LABEL_58;
  }
  v51 = v15 - 6;
  if ( (unsigned __int8)(v15 - 6) <= 1u )
  {
    *(_DWORD *)(v19 + 400) = -1;
    dword_4F04C58 = -1;
    qword_4F04C50 = 0;
    goto LABEL_178;
  }
  if ( (unsigned __int8)v49 <= 1u )
  {
    *(_DWORD *)(v19 + 400) = -1;
    v50 = (_BYTE)v15 == 0;
    dword_4F04C58 = -1;
    qword_4F04C50 = 0;
LABEL_179:
    if ( (_BYTE)v15 == 9 )
    {
      *(_BYTE *)(v19 + 9) = *(_BYTE *)(v19 + 9) & 0xE1 | (2 * *(_BYTE *)(a9 + 40)) & 0xE;
      v49 = 6;
LABEL_181:
      if ( (a13 & 0x1000) != 0 )
      {
LABEL_182:
        v69 = 9232;
        if ( !_bittest64(&v69, v15) )
        {
          if ( (_BYTE)v15 != 7 )
            goto LABEL_83;
          if ( (a13 & 0x40000) == 0 && !v50 )
            goto LABEL_292;
        }
LABEL_183:
        dword_4F04C40 = dword_4F04C64;
        goto LABEL_184;
      }
      goto LABEL_78;
    }
    goto LABEL_76;
  }
  v50 = v122 || (_BYTE)v15 == 11;
  if ( !v50 )
  {
LABEL_178:
    v50 = 0;
    goto LABEL_179;
  }
  *(_BYTE *)(v19 + 9) = *(_BYTE *)(v19 + 9) & 0xE1 | 4;
LABEL_77:
  if ( v51 <= 1u )
  {
LABEL_78:
    unk_4F04C2C = dword_4F04C64;
    goto LABEL_79;
  }
LABEL_205:
  switch ( (_BYTE)v15 )
  {
    case 0xA:
      if ( (a13 & 0x80000) == 0 )
        goto LABEL_81;
      goto LABEL_78;
    case 9:
      goto LABEL_181;
    case 0x11:
    case 0xE:
      goto LABEL_78;
  }
LABEL_79:
  if ( (_BYTE)v15 == 3 || !(_BYTE)v15 )
    goto LABEL_183;
LABEL_81:
  if ( (unsigned __int8)v15 <= 0xDu )
    goto LABEL_182;
  if ( (_BYTE)v15 == 17 )
  {
    if ( (*(_BYTE *)(a4 + 206) & 2) != 0 )
      goto LABEL_292;
    goto LABEL_183;
  }
LABEL_83:
  if ( (_BYTE)v15 == 2 || v50 || (a13 & 1) == 0 && (_BYTE)v15 == 9 )
    goto LABEL_183;
  switch ( (_BYTE)v15 )
  {
    case 6:
      v52 = *(_QWORD *)(a3 + 168);
      if ( (*(_BYTE *)(v52 + 109) & 0x20) != 0 )
        goto LABEL_89;
      goto LABEL_183;
    case 8:
      goto LABEL_197;
    case 1:
LABEL_184:
      if ( (_BYTE)v15 != 8 )
        goto LABEL_185;
LABEL_197:
      v73 = dword_4F04C64;
      *(_DWORD *)(v19 + 348) = dword_4F04C64;
      dword_4F04C44 = v73;
      if ( unk_4D043FC )
        *(_BYTE *)(v19 + 7) |= 1u;
      if ( (a13 & 0x800000) == 0 )
      {
        v74 = unk_4F04C18;
        unk_4F04C18 = 0;
        *(_QWORD *)(v19 + 672) = v74;
      }
      goto LABEL_201;
  }
LABEL_292:
  if ( (_BYTE)v15 != 14 && (_BYTE)v15 != 5 )
  {
    if ( (_BYTE)v15 == 9 )
    {
      if ( (a13 & 1) != 0 )
        goto LABEL_188;
      goto LABEL_90;
    }
    if ( (_BYTE)v15 != 16 && (_BYTE)v15 != 7 )
    {
      if ( (_BYTE)v15 == 15 )
        goto LABEL_188;
      if ( (_BYTE)v15 == 17 )
      {
        if ( (*(_BYTE *)(a4 + 206) & 2) != 0 )
          goto LABEL_188;
        goto LABEL_90;
      }
      if ( (_BYTE)v15 != 6 )
      {
LABEL_90:
        dword_4F04C40 = -1;
        goto LABEL_184;
      }
      v52 = *(_QWORD *)(a3 + 168);
LABEL_89:
      if ( (*(_BYTE *)(v52 + 109) & 0x20) != 0 )
        goto LABEL_188;
      goto LABEL_90;
    }
  }
LABEL_185:
  if ( (unsigned __int8)(v15 - 9) <= 1u && (a13 & 1) == 0 )
  {
    *(_DWORD *)(v19 + 348) = -1;
    dword_4F04C44 = -1;
  }
LABEL_188:
  if ( (_BYTE)v15 == 9 )
  {
    v70 = unk_4F04C18;
    unk_4F04C18 = 0;
    *(_QWORD *)(v19 + 672) = v70;
  }
  if ( (a13 & 0x800) == 0 )
  {
    if ( (v15 & 0xFB) == 1 || (_BYTE)v15 == 7 )
    {
      *(_BYTE *)(v19 + 6) = *(_BYTE *)(v19 - 770) & 0x80 | *(_BYTE *)(v19 + 6) & 0x7F;
      if ( (unsigned __int8)v49 <= 2u )
        goto LABEL_194;
    }
    else if ( (unsigned __int8)v49 <= 2u )
    {
      v71 = *(_QWORD *)(*(_QWORD *)a5 + 96LL);
      *(_QWORD *)(v19 + 24) = v71;
      goto LABEL_195;
    }
    goto LABEL_202;
  }
LABEL_201:
  *(_WORD *)(v19 + 6) |= 0x280u;
  if ( (unsigned __int8)v49 <= 2u )
  {
LABEL_194:
    v71 = *(_QWORD *)(*(_QWORD *)a5 + 96LL);
    *(_QWORD *)(v19 + 24) = v71;
    if ( (_BYTE)v15 == 5 )
    {
LABEL_196:
      *(_QWORD *)(v19 + 688) = *(_QWORD *)(v71 + 184);
      *(_QWORD *)(v19 + 696) = *(_QWORD *)(v71 + 192);
      goto LABEL_59;
    }
LABEL_195:
    *(_BYTE *)(v19 + 8) = *(_BYTE *)(v71 + 200) & 2 | *(_BYTE *)(v19 + 8) & 0xFD;
    v72 = dword_4F04C64;
    *(_DWORD *)(v19 + 520) = dword_4F04C64;
    dword_4F04C34 = v72;
    if ( (unsigned __int8)(v15 - 4) > 1u )
      goto LABEL_59;
    goto LABEL_196;
  }
LABEL_202:
  if ( !v117 || (_BYTE)v15 == 13 )
    qword_4D03C50 = 0;
LABEL_59:
  if ( (_BYTE)v15 == 6 )
  {
    for ( j = a3; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    *(_QWORD *)(v19 + 24) = *(_QWORD *)(*(_QWORD *)j + 96LL) + 192LL;
  }
  else if ( (_BYTE)v15 == 7 )
  {
    for ( k = a3; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    *(_QWORD *)(v19 + 168) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)k + 96LL) + 328LL);
  }
  else if ( (_BYTE)v15 )
  {
    if ( a12 )
      *(_QWORD *)(v19 + 24) = a12;
  }
  else
  {
    *(_QWORD *)(v19 + 24) = qword_4D03FF0 + 24;
  }
  if ( !unk_4D03FE8 )
    goto LABEL_66;
  if ( *(int *)(v19 + 200) > 0 )
  {
    dword_4F04C3C = 1;
    *(_BYTE *)(v19 + 7) |= 0x10u;
    goto LABEL_66;
  }
  switch ( (_BYTE)v15 )
  {
    case 8:
      if ( dword_4F07590 )
      {
LABEL_146:
        if ( dword_4F077C4 != 2 )
          return a11;
        v54 = (_BYTE)v15 == 17;
        goto LABEL_148;
      }
LABEL_145:
      dword_4F04C3C = 1;
      *(_BYTE *)(v19 + 7) |= 0x10u;
      goto LABEL_146;
    case 0xD:
      goto LABEL_145;
    case 9:
      if ( !a6 )
        goto LABEL_135;
      if ( (*(_BYTE *)(v19 + 6) & 0xA) != 0 )
      {
        if ( dword_4F07590 )
        {
          if ( v40 && *(char *)(a3 + 177) < 0 )
          {
            if ( (*(_BYTE *)(a3 + 89) & 2) != 0 )
            {
              v149 = a3;
              v101 = sub_72F070(a3);
              a3 = v149;
            }
            else
            {
              v101 = *(_QWORD *)(a3 + 40);
            }
            v102 = *(_QWORD *)a3;
            switch ( *(_BYTE *)(*(_QWORD *)a3 + 80LL) )
            {
              case 4:
              case 5:
                v105 = *(_QWORD *)(*(_QWORD *)(v102 + 96) + 80LL);
                goto LABEL_369;
              case 6:
                v105 = *(_QWORD *)(*(_QWORD *)(v102 + 96) + 32LL);
                goto LABEL_369;
              case 9:
              case 0xA:
                v105 = *(_QWORD *)(*(_QWORD *)(v102 + 96) + 56LL);
                goto LABEL_369;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v105 = *(_QWORD *)(v102 + 88);
LABEL_369:
                if ( !v105 || (*(_BYTE *)(v105 + 160) & 1) == 0 )
                  break;
                goto LABEL_351;
              default:
                break;
            }
            while ( v101 && *(_BYTE *)(v101 + 28) == 6 )
            {
              v103 = *(_QWORD *)(v101 + 32);
              v104 = *(_BYTE *)(v103 + 177);
              if ( (v104 & 0x10) != 0 && (v104 & 0x60) != 0x20 && (*(_BYTE *)(v103 + 178) & 1) == 0 )
                goto LABEL_135;
              v101 = *(_QWORD *)(v103 + 40);
            }
          }
LABEL_351:
          if ( !dword_4F5FD28 )
          {
            v40 = 0;
            dword_4F04C3C = 0;
            goto LABEL_136;
          }
        }
        goto LABEL_135;
      }
      if ( (*(_BYTE *)(qword_4F04C68[0] + 7LL) & 0x10) != 0 )
      {
        v40 = dword_4F04C3C & 1;
        goto LABEL_136;
      }
      if ( !v40 )
      {
LABEL_135:
        v40 = 1;
        dword_4F04C3C = 1;
LABEL_136:
        *(_BYTE *)(v19 + 7) = (16 * v40) | *(_BYTE *)(v19 + 7) & 0xEF;
        if ( dword_4F077C4 != 2 )
          return a11;
        v54 = 0;
        goto LABEL_111;
      }
      if ( unk_4D04238 )
      {
        if ( (*(_BYTE *)(a3 + 178) & 1) != 0 && !v118 )
          goto LABEL_346;
        v134 = v40;
        v148 = a3;
        v100 = sub_8D23B0(a3);
        v40 = v134;
        if ( !v100 )
        {
          a3 = v148;
LABEL_346:
          v98 = (*(_BYTE *)(a3 + 177) & 0x40) != 0;
          v40 = (*(_BYTE *)(a3 + 177) & 0x40) != 0;
LABEL_347:
          dword_4F04C3C = v98;
          goto LABEL_136;
        }
      }
      v98 = 1;
      goto LABEL_347;
  }
LABEL_66:
  if ( dword_4F077C4 != 2 )
    return a11;
  v54 = (_BYTE)v15 == 17;
  if ( !v117 )
    goto LABEL_111;
LABEL_148:
  if ( (unsigned __int8)v15 <= 0xDu )
  {
    v61 = 10434;
    if ( _bittest64(&v61, v15) )
LABEL_111:
      qword_4F06BC0 = *(_QWORD *)(qword_4F04C68[0] + 488LL);
  }
  if ( a10 )
  {
    *(_QWORD *)(v19 + 488) = a10;
    qword_4F06BC0 = a10;
  }
  else
  {
    if ( !(_BYTE)v15 )
    {
      v83 = a11[11];
      qword_4F06BC0 = v83;
      if ( !v83 )
      {
        sub_733780(0x17u, (__int64)a11, 0, 0, 0);
        v83 = qword_4F06BC0;
      }
      *(_QWORD *)(v19 + 488) = v83;
      return a11;
    }
    if ( (_BYTE)v15 != 15 && (_BYTE)v15 != 2 && !v54 )
      return a11;
    v84 = *(_BYTE *)(v19 + 10);
    if ( (v84 & 4) == 0 )
    {
      if ( v135 )
        v116 = (char *)a11[11];
      sub_733780(0x17u, (__int64)a11, v116, 1, (v84 & 4) != 0);
      *(_QWORD *)(v19 + 488) = qword_4F06BC0;
    }
  }
  if ( (_BYTE)v15 == 2 )
  {
    if ( *(char *)(v19 - 769) < 0 )
      *(_BYTE *)(v19 + 7) |= 0x80u;
    return a11;
  }
  if ( (_BYTE)v15 != 15 )
    return a11;
  return (__int64 *)sub_732EF0(v19);
}
