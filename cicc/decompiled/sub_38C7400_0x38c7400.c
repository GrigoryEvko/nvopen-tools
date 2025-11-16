// Function: sub_38C7400
// Address: 0x38c7400
//
void __fastcall sub_38C7400(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rbx
  unsigned __int8 v13; // al
  __int64 v14; // r15
  int v15; // r13d
  __int64 v16; // rdi
  __int64 v17; // rax
  int v18; // r9d
  unsigned int v19; // eax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  unsigned int v26; // eax
  __int64 v27; // r8
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rsi
  char v31; // al
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rax
  int v40; // r10d
  unsigned int v41; // r10d
  unsigned int v42; // r12d
  char v43; // cl
  __int64 *v44; // r8
  __int64 *v45; // r14
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 *v50; // r8
  __int64 v51; // rsi
  __int64 *v52; // rbx
  __int64 v53; // r14
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 *v56; // rbx
  __int64 v57; // r13
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 *v62; // r12
  char v63; // al
  _QWORD *v64; // rbx
  __int64 *v65; // rbx
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // r12
  __int64 v69; // rax
  int v70; // esi
  int v71; // eax
  __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // rax
  __int64 *v75; // r13
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // r14
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rsi
  int v82; // r13d
  int v83; // r8d
  int v84; // r9d
  __int64 v85; // rax
  int v86; // esi
  unsigned int v87; // eax
  __int64 *v88; // r8
  int v89; // esi
  unsigned int v90; // r14d
  __int64 *v91; // r13
  unsigned int v92; // eax
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r12
  __int64 v96; // rsi
  __int64 v97; // rax
  _BYTE *v98; // rdx
  unsigned int v99; // edx
  __int64 v100; // rax
  __int64 v101; // rax
  unsigned __int64 v102; // rdi
  __int64 v103; // [rsp+0h] [rbp-130h]
  __int64 v104; // [rsp+8h] [rbp-128h]
  __int64 v105; // [rsp+8h] [rbp-128h]
  __int64 v106; // [rsp+10h] [rbp-120h]
  __int64 v107; // [rsp+10h] [rbp-120h]
  __int64 v108; // [rsp+18h] [rbp-118h]
  unsigned int v109; // [rsp+20h] [rbp-110h]
  char v110; // [rsp+20h] [rbp-110h]
  int v112; // [rsp+28h] [rbp-108h]
  __int64 v113; // [rsp+28h] [rbp-108h]
  __int64 v114; // [rsp+28h] [rbp-108h]
  __int64 v115; // [rsp+28h] [rbp-108h]
  __int64 v116; // [rsp+30h] [rbp-100h]
  __int64 v117; // [rsp+30h] [rbp-100h]
  __int64 v118; // [rsp+38h] [rbp-F8h]
  __int64 v119; // [rsp+38h] [rbp-F8h]
  __int64 v120; // [rsp+38h] [rbp-F8h]
  __int64 v121; // [rsp+38h] [rbp-F8h]
  __int64 v122; // [rsp+38h] [rbp-F8h]
  char v123; // [rsp+40h] [rbp-F0h]
  __int64 v124; // [rsp+40h] [rbp-F0h]
  __int64 v125; // [rsp+40h] [rbp-F0h]
  __int64 v126; // [rsp+40h] [rbp-F0h]
  __int64 v127; // [rsp+48h] [rbp-E8h]
  __int64 v128; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v129; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int8 v130; // [rsp+68h] [rbp-C8h]
  __int64 *v131; // [rsp+70h] [rbp-C0h]
  __m128i v132; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v133; // [rsp+90h] [rbp-A0h]
  __int64 v134; // [rsp+B0h] [rbp-80h]
  _BYTE *v135; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v136; // [rsp+C8h] [rbp-68h]
  _BYTE v137[16]; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v138; // [rsp+E0h] [rbp-50h] BYREF
  unsigned __int64 v139; // [rsp+E8h] [rbp-48h]
  __int64 v140; // [rsp+F0h] [rbp-40h]
  unsigned int v141; // [rsp+F8h] [rbp-38h]

  sub_38DCD40();
  v5 = a1[1];
  v130 = a3;
  v129 = 0;
  v6 = a1[4];
  v7 = *(_QWORD *)(v5 + 32);
  v8 = a1[3];
  v104 = v5;
  v9 = *(_QWORD *)(v5 + 16);
  v131 = a1;
  v108 = v7;
  v106 = v9;
  v10 = v7;
  v127 = v6;
  v11 = *(unsigned __int8 *)(v7 + 2) ^ 1u;
  v123 = v11;
  if ( a3 )
  {
    v11 = v10;
    if ( *(_QWORD *)(v10 + 64) && v6 != v8 )
    {
      v103 = v8;
      v12 = v8;
      v13 = 0;
      v14 = v11;
      while ( 1 )
      {
        v15 = *(_DWORD *)(v12 + 68);
        if ( !v15 )
          goto LABEL_6;
        if ( v13 )
        {
          v123 |= *(_DWORD *)(v14 + 20) == v15;
          v16 = v131[1];
          v17 = *(_QWORD *)(v16 + 32);
LABEL_10:
          v18 = v15;
          if ( *(_DWORD *)(v17 + 20) != v15 && *(_QWORD *)(v12 + 24) )
            v18 = v15 | 0x40000000;
          v112 = *(_DWORD *)(v17 + 20);
          v109 = v18;
          v19 = sub_38C54A0(v16, *(_DWORD *)(v17 + 12));
          sub_38DDC80(v20, *(_QWORD *)v12, v19);
          v116 = *(_QWORD *)v12;
          v118 = sub_38CF310(*(_QWORD *)(v12 + 8), 0, v131[1], 0);
          v21 = sub_38CF310(v116, 0, v131[1], 0);
          v119 = sub_38CB1F0(17, v118, v21, v131[1], 0);
          v22 = sub_38CB470(0, v131[1]);
          v23 = sub_38CB1F0(17, v119, v22, v131[1], 0);
          sub_38C4F40(v131, v23, 4u);
          (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, v109, 4);
          v24 = *(unsigned int *)(*(_QWORD *)(v131[1] + 16) + 8LL);
          if ( v112 == v15 )
          {
            (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, 0, v24);
            v29 = (unsigned int)sub_38C54A0(v131[1], *(_DWORD *)(v12 + 64));
LABEL_53:
            (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v27 + 424LL))(v27, 0, v29);
            v13 = a3;
            goto LABEL_6;
          }
          v25 = *(_QWORD *)(v12 + 16);
          if ( v25 )
            sub_38DDC80(v131, v25, v24);
          else
            (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, 0, v24);
          v26 = sub_38C54A0(v131[1], *(_DWORD *)(v12 + 64));
          v28 = *(_QWORD *)(v12 + 24);
          v29 = v26;
          if ( !v28 )
            goto LABEL_53;
          v12 += 80;
          sub_38DDC80(v27, v28, v26);
          v13 = a3;
          if ( v127 == v12 )
          {
LABEL_18:
            v8 = v103;
            break;
          }
        }
        else
        {
          (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(v14 + 64), 0);
          (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64, _QWORD))(*a1 + 512))(
            a1,
            *(unsigned int *)(v106 + 8),
            0,
            1,
            0);
          v15 = *(_DWORD *)(v12 + 68);
          v123 |= *(_DWORD *)(v14 + 20) == v15;
          v13 = a3;
          if ( v15 )
          {
            v16 = v131[1];
            v17 = *(_QWORD *)(v16 + 32);
            goto LABEL_10;
          }
LABEL_6:
          v12 += 80;
          if ( v127 == v12 )
            goto LABEL_18;
        }
      }
    }
    if ( !v123 )
      return;
    v30 = *(_QWORD *)(v108 + 408);
  }
  else
  {
    if ( !(_BYTE)v11 )
      return;
    v30 = *(_QWORD *)(v108 + 104);
  }
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64))(*a1 + 160))(a1, v30, 0, v11);
  v107 = sub_38BFA60(v104, 1);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v107, 0);
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v31 = *(_BYTE *)(v108 + 3);
  v128 = 0;
  v110 = v31;
  if ( v127 == v8 )
  {
    v102 = 0;
    goto LABEL_47;
  }
  do
  {
    v8 += 80;
    if ( v110 && *(_DWORD *)(v8 - 12) != *(_DWORD *)(v108 + 20) )
      continue;
    v132.m128i_i64[0] = *(_QWORD *)(v8 - 64);
    v132.m128i_i64[1] = *(_QWORD *)(v8 - 20);
    LOWORD(v133) = *(_WORD *)(v8 - 8);
    HIDWORD(v133) = *(_DWORD *)(v8 - 4);
    if ( !a3 )
    {
      v52 = &v128;
      v126 = v128;
      goto LABEL_42;
    }
    v63 = sub_38C7080((__int64)&v138, (__int64)&v132, &v135);
    v64 = v135;
    if ( !v63 )
    {
      v70 = v141;
      ++v138;
      v71 = v140 + 1;
      if ( 4 * ((int)v140 + 1) >= 3 * v141 )
      {
        v70 = 2 * v141;
      }
      else if ( v141 - HIDWORD(v140) - v71 > v141 >> 3 )
      {
        goto LABEL_56;
      }
      sub_38C71D0((__int64)&v138, v70);
      sub_38C7080((__int64)&v138, (__int64)&v132, &v135);
      v64 = v135;
      v71 = v140 + 1;
LABEL_56:
      LODWORD(v140) = v71;
      if ( *v64
        || v64[1] != 0xFFFFFFFF00000000LL
        || (HIDWORD(v134) = 0x7FFFFFFF, v100 = v64[2], LOWORD(v134) = 0, ((v134 ^ v100) & 0xFFFFFFFF0000FFFFLL) != 0) )
      {
        --HIDWORD(v140);
      }
      v52 = v64 + 3;
      *(__m128i *)(v52 - 3) = _mm_loadu_si128(&v132);
      v72 = v133;
      *v52 = 0;
      *(v52 - 1) = v72;
      goto LABEL_59;
    }
    v52 = (__int64 *)(v135 + 24);
    v126 = *((_QWORD *)v135 + 3);
LABEL_42:
    if ( v126 )
      goto LABEL_43;
LABEL_59:
    v73 = v131[1];
    v105 = *(_QWORD *)(v73 + 24);
    v117 = *(_QWORD *)(v73 + 32);
    v126 = sub_38BFA60(v73, 1);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v131 + 176))(v131, v126, 0);
    v74 = sub_38BFA60(v73, 1);
    v75 = v131;
    v115 = v74;
    v76 = sub_38CF310(v74, 0, v131[1], 0);
    v77 = sub_38CF310(v126, 0, v75[1], 0);
    v78 = sub_38CB1F0(17, v76, v77, v75[1], 0);
    v79 = sub_38CB470(4, v75[1]);
    v80 = sub_38CB1F0(17, v78, v79, v75[1], 0);
    sub_38C4F40(v131, v80, 4u);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, -(v130 ^ 1), 4);
    if ( v130 )
    {
      v81 = 1;
      v82 = 1;
    }
    else
    {
      v81 = dword_452E010[*(unsigned __int16 *)(v73 + 1160) - 2];
      v82 = dword_452E010[*(unsigned __int16 *)(v73 + 1160) - 2];
    }
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v131 + 424))(v131, v81, 1);
    v135 = v137;
    v136 = 0x800000000LL;
    if ( v130 )
    {
      v137[0] = 122;
      LODWORD(v136) = 1;
      if ( *(_QWORD *)(v8 - 64) )
      {
        v137[1] = 80;
        v97 = 2;
        LODWORD(v136) = 2;
        if ( !*(_QWORD *)(v8 - 56) )
          goto LABEL_102;
      }
      else
      {
        v97 = 1;
        if ( !*(_QWORD *)(v8 - 56) )
        {
          v98 = v137;
          goto LABEL_93;
        }
      }
      v137[v97] = 76;
      v97 = (unsigned int)(v136 + 1);
      LODWORD(v136) = v97;
      if ( HIDWORD(v136) == (_DWORD)v97 )
      {
        sub_16CD150((__int64)&v135, v137, v97 + 1, 1, v83, v84);
        v98 = v135;
        v97 = (unsigned int)v136;
LABEL_93:
        v98[v97] = 82;
        v99 = v136 + 1;
        LODWORD(v136) = v136 + 1;
        if ( *(_BYTE *)(v8 - 8) )
        {
          v101 = v99;
          if ( v99 == HIDWORD(v136) )
          {
            sub_16CD150((__int64)&v135, v137, v99 + 1LL, 1, v83, v84);
            v101 = (unsigned int)v136;
          }
          v135[v101] = 83;
          LODWORD(v136) = v136 + 1;
        }
        (*(void (__fastcall **)(__int64 *, _BYTE *))(*v131 + 400))(v131, v135);
        goto LABEL_62;
      }
LABEL_102:
      v98 = v135;
      goto LABEL_93;
    }
LABEL_62:
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, 0, 1);
    if ( v82 == 4 )
    {
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(
        v131,
        *(unsigned int *)(*(_QWORD *)(v73 + 16) + 8LL),
        1);
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, 0, 1);
    }
    sub_38DCDD0(v131, *(unsigned int *)(*(_QWORD *)(v73 + 16) + 28LL));
    v85 = *(_QWORD *)(v131[1] + 16);
    v86 = *(_DWORD *)(v85 + 12);
    if ( !*(_BYTE *)(v85 + 17) )
      v86 = -v86;
    sub_38DCF20(v131, v86);
    v87 = *(_DWORD *)(v8 - 4);
    if ( v87 == 0x7FFFFFFF )
      v87 = sub_38D70E0(v105, *(unsigned int *)(v105 + 12), v130);
    if ( v82 == 1 )
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, v87, 1);
    else
      sub_38DCDD0(v131, v87);
    if ( v130 )
    {
      v88 = v131;
      v89 = 1;
      if ( *(_QWORD *)(v8 - 64) )
        v89 = sub_38C54A0(v131[1], *(_DWORD *)(v8 - 20)) + 2;
      sub_38DCDD0(v88, v89 - ((unsigned int)(*(_QWORD *)(v8 - 56) == 0) - 1));
      if ( *(_QWORD *)(v8 - 64) )
      {
        (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, *(unsigned int *)(v8 - 20), 1);
        v90 = *(_DWORD *)(v8 - 20);
        v91 = v131;
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64 *))(**(_QWORD **)(v131[1] + 16) + 32LL))(
          *(_QWORD *)(v131[1] + 16),
          *(_QWORD *)(v8 - 64),
          v90,
          v131);
        v92 = sub_38C54A0(v91[1], v90);
        sub_38DDD30(v91, v94, v92, v93);
      }
      if ( *(_QWORD *)(v8 - 56) )
        (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, *(unsigned int *)(v8 - 16), 1);
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v131 + 424))(v131, *(unsigned int *)(v117 + 12), 1);
    }
    v95 = *(_QWORD *)(v73 + 16);
    if ( !*(_BYTE *)(v8 - 7) )
      sub_38C4FC0(
        (__int64)&v129,
        *(unsigned int **)(v95 + 368),
        0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v95 + 376) - *(_QWORD *)(v95 + 368)) >> 4),
        0);
    v96 = 4;
    HIDWORD(v129) = v129;
    if ( !v130 )
      v96 = *(unsigned int *)(v95 + 8);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64, _QWORD))(*v131 + 512))(v131, v96, 0, 1, 0);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v131 + 176))(v131, v115, 0);
    if ( v135 != v137 )
      _libc_free((unsigned __int64)v135);
    *v52 = v126;
LABEL_43:
    v53 = v131[1];
    v54 = sub_38BFA60(v53, 1);
    v55 = sub_38BFA60(v53, 1);
    v56 = v131;
    v57 = v55;
    v58 = v131[1];
    v114 = *(_QWORD *)(v53 + 32);
    LODWORD(v129) = HIDWORD(v129);
    v120 = sub_38CF310(v55, 0, v58, 0);
    v59 = sub_38CF310(v54, 0, v56[1], 0);
    v121 = sub_38CB1F0(17, v120, v59, v56[1], 0);
    v60 = sub_38CB470(0, v56[1]);
    v61 = sub_38CB1F0(17, v121, v60, v56[1], 0);
    sub_38C4F40(v131, v61, 4u);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v131 + 176))(v131, v54, 0);
    v122 = *(_QWORD *)(v53 + 16);
    if ( v130 )
    {
      v65 = v131;
      v66 = sub_38CF310(v54, 0, v131[1], 0);
      v67 = sub_38CF310(v126, 0, v65[1], 0);
      v68 = sub_38CB1F0(17, v66, v67, v65[1], 0);
      v69 = sub_38CB470(0, v65[1]);
      v36 = v65[1];
      v37 = v68;
      v38 = v69;
    }
    else
    {
      v62 = v131;
      if ( *(_BYTE *)(v122 + 356) )
      {
        sub_38DDC80(v131, v126, 4);
        goto LABEL_27;
      }
      v32 = sub_38CF310(v126, 0, v131[1], 0);
      v33 = sub_38CF310(v107, 0, v62[1], 0);
      v34 = sub_38CB1F0(17, v32, v33, v62[1], 0);
      v35 = sub_38CB470(0, v62[1]);
      v36 = v62[1];
      v37 = v34;
      v38 = v35;
    }
    v39 = sub_38CB1F0(17, v37, v38, v36, 0);
    sub_38C4F40(v131, v39, 4u);
LABEL_27:
    LOBYTE(v40) = 0;
    if ( v130 )
      v40 = *(_DWORD *)(v114 + 12);
    v42 = sub_38C54A0(v131[1], v40);
    sub_38C54F0(v44, *(_QWORD *)(v8 - 80), v41, v43);
    v45 = v131;
    v113 = *(_QWORD *)(v8 - 80);
    v124 = sub_38CF310(*(_QWORD *)(v8 - 72), 0, v131[1], 0);
    v46 = sub_38CF310(v113, 0, v45[1], 0);
    v125 = sub_38CB1F0(17, v124, v46, v45[1], 0);
    v47 = sub_38CB470(0, v45[1]);
    v48 = sub_38CB1F0(17, v125, v47, v45[1], 0);
    sub_38C4F40(v131, v48, v42);
    if ( v130 )
    {
      v49 = 0;
      v50 = v131;
      if ( *(_QWORD *)(v8 - 56) )
        v49 = (unsigned int)sub_38C54A0(v131[1], *(_DWORD *)(v8 - 16));
      sub_38DCDD0(v50, v49);
      v51 = *(_QWORD *)(v8 - 56);
      if ( v51 )
        sub_38C54F0(v131, v51, *(unsigned int *)(v8 - 16), 1);
    }
    sub_38C4FC0(
      (__int64)&v129,
      *(unsigned int **)(v8 - 48),
      0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v8 - 40) - *(_QWORD *)(v8 - 48)) >> 4),
      *(_QWORD *)(v8 - 80));
    if ( v127 == v8 )
      v42 = *(_DWORD *)(v122 + 8);
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64, _QWORD))(*v131 + 512))(v131, v42, 0, 1, 0);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v131 + 176))(v131, v57, 0);
  }
  while ( v127 != v8 );
  v102 = v139;
LABEL_47:
  j___libc_free_0(v102);
}
