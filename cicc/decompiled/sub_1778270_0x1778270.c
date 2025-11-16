// Function: sub_1778270
// Address: 0x1778270
//
__int64 __fastcall sub_1778270(_QWORD *a1, __int64 a2)
{
  unsigned int v4; // r14d
  unsigned int v5; // r15d
  _QWORD **v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r14
  __int64 v11; // rax
  char v12; // al
  unsigned int v13; // r14d
  unsigned __int64 v14; // rdx
  const char *v15; // r8
  int v16; // r9d
  size_t v17; // r15
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  const char *v21; // r8
  int v22; // r9d
  size_t v23; // r15
  unsigned __int64 v24; // rax
  char *v25; // rax
  unsigned int i; // eax
  __int64 *v27; // rax
  __int64 v28; // rdi
  unsigned __int8 *v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rbx
  __int64 v33; // r14
  _QWORD *v34; // rax
  _QWORD *v35; // r12
  __int64 v36; // rdi
  unsigned __int64 v37; // rsi
  __int64 v38; // rax
  __int64 *v39; // rsi
  _QWORD *v40; // rdi
  __int64 v41; // rdx
  bool v42; // zf
  __int64 v43; // rsi
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  unsigned __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // r15
  int v49; // r14d
  unsigned __int64 v50; // rdx
  const char *v51; // r8
  int v52; // r9d
  size_t v53; // r15
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  const char *v57; // r8
  size_t v58; // r9
  unsigned __int64 v59; // rax
  char *v60; // rax
  __int64 v61; // r12
  __int64 *v62; // rax
  __int64 v63; // rdi
  unsigned __int8 *v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rbx
  _QWORD *v67; // rax
  _QWORD *v68; // r15
  __int64 v69; // rdi
  unsigned __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rsi
  unsigned __int8 *v74; // rsi
  __int64 v75; // rdi
  unsigned __int8 *v76; // rax
  _QWORD *v77; // rdi
  int v78; // r8d
  int v79; // r9d
  _BYTE *v80; // rdi
  int v81; // r8d
  int v82; // r9d
  _QWORD *v83; // rdi
  int v84; // r8d
  _BYTE *v85; // rdi
  int v86; // r8d
  int v87; // r9d
  __int64 v88; // [rsp+20h] [rbp-1C0h]
  __int64 v89; // [rsp+28h] [rbp-1B8h]
  __int64 v90; // [rsp+28h] [rbp-1B8h]
  __int64 v91; // [rsp+30h] [rbp-1B0h]
  _BYTE *v92; // [rsp+50h] [rbp-190h]
  __int64 v93; // [rsp+58h] [rbp-188h]
  __int64 *v94; // [rsp+58h] [rbp-188h]
  _BYTE *v95; // [rsp+60h] [rbp-180h]
  __int64 v96; // [rsp+60h] [rbp-180h]
  __int64 *v97; // [rsp+68h] [rbp-178h]
  unsigned __int64 v98; // [rsp+68h] [rbp-178h]
  __int64 v99; // [rsp+70h] [rbp-170h]
  _QWORD *v100; // [rsp+78h] [rbp-168h]
  __int64 v101; // [rsp+80h] [rbp-160h]
  __int64 v102; // [rsp+80h] [rbp-160h]
  unsigned int v103; // [rsp+88h] [rbp-158h]
  __int64 v104; // [rsp+88h] [rbp-158h]
  unsigned __int8 *v105; // [rsp+88h] [rbp-158h]
  unsigned __int64 *v106; // [rsp+88h] [rbp-158h]
  __int64 v107; // [rsp+90h] [rbp-150h]
  unsigned __int8 *v108; // [rsp+98h] [rbp-148h]
  unsigned __int64 v109; // [rsp+98h] [rbp-148h]
  const char *v110; // [rsp+98h] [rbp-148h]
  void *src; // [rsp+A0h] [rbp-140h]
  void *srca; // [rsp+A0h] [rbp-140h]
  unsigned __int8 *srcb; // [rsp+A0h] [rbp-140h]
  unsigned __int64 *srce; // [rsp+A0h] [rbp-140h]
  void *srcf; // [rsp+A0h] [rbp-140h]
  char *srcc; // [rsp+A0h] [rbp-140h]
  const char *srcg; // [rsp+A0h] [rbp-140h]
  const char *srch; // [rsp+A0h] [rbp-140h]
  void *srci; // [rsp+A0h] [rbp-140h]
  int srcd; // [rsp+A0h] [rbp-140h]
  const char *srcj; // [rsp+A0h] [rbp-140h]
  __int64 v122; // [rsp+A8h] [rbp-138h]
  unsigned int v123; // [rsp+BCh] [rbp-124h] BYREF
  _QWORD *v124; // [rsp+C0h] [rbp-120h] BYREF
  _QWORD *v125; // [rsp+C8h] [rbp-118h] BYREF
  __int64 *v126[2]; // [rsp+D0h] [rbp-110h] BYREF
  __int64 *v127[2]; // [rsp+E0h] [rbp-100h] BYREF
  __int64 v128[2]; // [rsp+F0h] [rbp-F0h] BYREF
  __int16 v129; // [rsp+100h] [rbp-E0h]
  __int64 v130[2]; // [rsp+110h] [rbp-D0h] BYREF
  __int16 v131; // [rsp+120h] [rbp-C0h]
  _BYTE *v132; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v133; // [rsp+138h] [rbp-A8h]
  _BYTE v134[16]; // [rsp+140h] [rbp-A0h] BYREF
  __int64 **v135; // [rsp+150h] [rbp-90h] BYREF
  __int64 v136; // [rsp+158h] [rbp-88h]
  _QWORD v137[2]; // [rsp+160h] [rbp-80h] BYREF
  _BYTE *v138; // [rsp+170h] [rbp-70h] BYREF
  __int64 v139; // [rsp+178h] [rbp-68h]
  _BYTE v140[16]; // [rsp+180h] [rbp-60h] BYREF
  __int64 *v141; // [rsp+190h] [rbp-50h] BYREF
  __int64 v142; // [rsp+198h] [rbp-48h]
  _QWORD v143[8]; // [rsp+1A0h] [rbp-40h] BYREF

  if ( sub_15F32D0(a2) )
    return 0;
  v5 = *(_WORD *)(a2 + 18) & 1;
  if ( (*(_WORD *)(a2 + 18) & 1) != 0 )
    return 0;
  v7 = **(_QWORD ****)(a2 - 48);
  v122 = *(_QWORD *)(a2 - 48);
  if ( (unsigned int)*((unsigned __int8 *)v7 + 8) - 13 > 1 )
    return 0;
  src = (void *)a1[333];
  v4 = *(unsigned __int16 *)(a2 + 18);
  if ( (unsigned int)sub_15A9FE0((__int64)src, (__int64)v7) < 1 << (v4 >> 1) >> 1 )
    return 0;
  v8 = (__int64)src;
  v9 = (__int64)v7;
  v10 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v9 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v47 = *(_QWORD *)(v9 + 32);
        v9 = *(_QWORD *)(v9 + 24);
        v10 *= v47;
        continue;
      case 1:
        v11 = 16;
        break;
      case 2:
        v11 = 32;
        break;
      case 3:
      case 9:
        v11 = 64;
        break;
      case 4:
        v11 = 80;
        break;
      case 5:
      case 6:
        v11 = 128;
        break;
      case 7:
        v11 = 8 * (unsigned int)sub_15A9520((__int64)src, 0);
        break;
      case 0xB:
        v11 = *(_DWORD *)(v9 + 8) >> 8;
        break;
      case 0xD:
        v11 = 8LL * *(_QWORD *)sub_15A9930((__int64)src, v9);
        break;
      case 0xE:
        v107 = (__int64)src;
        v104 = *(_QWORD *)(v9 + 24);
        srcf = *(void **)(v9 + 32);
        v109 = (unsigned int)sub_15A9FE0(v8, v104);
        v11 = 8 * (_QWORD)srcf * v109 * ((v109 + ((unsigned __int64)(sub_127FA20(v107, v104) + 7) >> 3) - 1) / v109);
        break;
      case 0xF:
        v11 = 8 * (unsigned int)sub_15A9520((__int64)src, *(_DWORD *)(v9 + 8) >> 8);
        break;
    }
    break;
  }
  if ( (unsigned int)dword_4FA27C0 <= (unsigned __int64)(v11 * v10 + 7) >> 3 )
    return 0;
  v12 = *((_BYTE *)v7 + 8);
  if ( v12 != 13 )
  {
    if ( v12 == 14 )
    {
      v100 = v7[4];
      if ( v100 == (_QWORD *)1 )
        goto LABEL_79;
      if ( a1[342] < (unsigned __int64)v100 )
        return v5;
      v48 = a1[333];
      v98 = sub_12BE0A0(v48, (__int64)v7[3]);
      v49 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
      if ( !v49 )
        v49 = sub_15A9FE0(v48, (__int64)v7);
      v51 = sub_1649960(v122);
      v53 = v50;
      v54 = v50;
      v138 = v140;
      v139 = 0x1000000000LL;
      if ( v50 > 0x10 )
      {
        srcj = v51;
        sub_16CD150((__int64)&v138, v140, v50, 1, (int)v51, v52);
        v51 = srcj;
        v85 = &v138[(unsigned int)v139];
      }
      else
      {
        if ( !v50 )
        {
          LODWORD(v139) = 0;
          goto LABEL_59;
        }
        v85 = v140;
      }
      memcpy(v85, v51, v53);
      LODWORD(v139) = v53 + v139;
      v54 = (unsigned int)v139;
      if ( HIDWORD(v139) - (unsigned __int64)(unsigned int)v139 <= 3 )
      {
        sub_16CD150((__int64)&v138, v140, (unsigned int)v139 + 4LL, 1, v86, v87);
        v54 = (unsigned int)v139;
      }
LABEL_59:
      *(_DWORD *)&v138[v54] = 1953260846;
      v55 = *(_QWORD *)(a2 - 24);
      LODWORD(v139) = v139 + 4;
      v92 = (_BYTE *)v55;
      v57 = sub_1649960(v55);
      v58 = v56;
      v59 = v56;
      v142 = 0x1000000000LL;
      v141 = v143;
      if ( v56 > 0x10 )
      {
        v110 = v57;
        srci = (void *)v56;
        sub_16CD150((__int64)&v141, v143, v56, 1, (int)v57, v56);
        v58 = (size_t)srci;
        v57 = v110;
        v83 = (__int64 *)((char *)v141 + (unsigned int)v142);
      }
      else
      {
        if ( !v56 )
        {
          LODWORD(v142) = 0;
          goto LABEL_62;
        }
        v83 = v143;
      }
      srcd = v58;
      memcpy(v83, v57, v58);
      LODWORD(v142) = srcd + v142;
      v59 = (unsigned int)v142;
      if ( HIDWORD(v142) - (unsigned __int64)(unsigned int)v142 <= 6 )
      {
        sub_16CD150((__int64)&v141, v143, (unsigned int)v142 + 7LL, 1, v84, srcd);
        v59 = (unsigned int)v142;
      }
LABEL_62:
      v60 = (char *)v141 + v59;
      *((_WORD *)v60 + 2) = 25441;
      *(_DWORD *)v60 = 1885696558;
      v60[6] = 107;
      LODWORD(v142) = v142 + 7;
      v96 = sub_1643360(*v7);
      v94 = (__int64 *)sub_159C470(v96, 0, 0);
      if ( v100 )
      {
        v90 = (__int64)v7;
        srcc = 0;
        v88 = a2;
        v61 = 0;
        while ( 1 )
        {
          v127[0] = v94;
          v62 = (__int64 *)sub_159C470(v96, v61, 0);
          v63 = a1[1];
          v127[1] = v62;
          LOWORD(v137[0]) = 262;
          v135 = &v141;
          v64 = sub_1709730(v63, v90, v92, v127, 2u, (__int64 *)&v135);
          v65 = a1[1];
          LOWORD(v137[0]) = 262;
          v102 = (__int64)v64;
          LODWORD(v132) = v61;
          v135 = (__int64 **)&v138;
          v105 = sub_1759FE0(v65, v122, (unsigned int *)&v132, 1, (__int64 *)&v135);
          v66 = a1[1];
          v131 = 257;
          v67 = sub_1648A60(64, 2u);
          v68 = v67;
          if ( v67 )
            sub_15F9650((__int64)v67, (__int64)v105, v102, 0, 0);
          v69 = *(_QWORD *)(v66 + 8);
          if ( v69 )
          {
            v106 = *(unsigned __int64 **)(v66 + 16);
            sub_157E9D0(v69 + 40, (__int64)v68);
            v70 = *v106;
            v71 = v68[3] & 7LL;
            v68[4] = v106;
            v70 &= 0xFFFFFFFFFFFFFFF8LL;
            v68[3] = v70 | v71;
            *(_QWORD *)(v70 + 8) = v68 + 3;
            *v106 = *v106 & 7 | (unsigned __int64)(v68 + 3);
          }
          v39 = v130;
          v40 = v68;
          sub_164B780((__int64)v68, v130);
          v42 = *(_QWORD *)(v66 + 80) == 0;
          v125 = v68;
          if ( v42 )
            break;
          (*(void (__fastcall **)(__int64, _QWORD **))(v66 + 88))(v66 + 64, &v125);
          v72 = *(_QWORD *)v66;
          if ( *(_QWORD *)v66 )
          {
            v135 = *(__int64 ***)v66;
            sub_1623A60((__int64)&v135, v72, 2);
            v73 = v68[6];
            if ( v73 )
              sub_161E7C0((__int64)(v68 + 6), v73);
            v74 = (unsigned __int8 *)v135;
            v68[6] = v135;
            if ( v74 )
              sub_1623210((__int64)&v135, v74, (__int64)(v68 + 6));
          }
          ++v61;
          sub_15F9450((__int64)v68, ((unsigned int)srcc | v49) & -((unsigned int)srcc | v49));
          v135 = 0;
          v136 = 0;
          v137[0] = 0;
          sub_14A8180(v88, (__int64 *)&v135, 0);
          sub_1626170((__int64)v68, (__int64 *)&v135);
          srcc += v98;
          if ( (_QWORD *)v61 == v100 )
            goto LABEL_75;
        }
LABEL_98:
        sub_4263D6(v40, v39, v41);
      }
LABEL_75:
      if ( v141 != v143 )
        _libc_free((unsigned __int64)v141);
      v46 = (unsigned __int64)v138;
      if ( v138 != v140 )
        goto LABEL_39;
      return 1;
    }
    return 0;
  }
  if ( *(_BYTE *)(v122 + 16) != 54 || !(unsigned __int8)sub_1776740(*(_QWORD *)(v122 - 24)) )
  {
    v103 = *((_DWORD *)v7 + 3);
    if ( v103 != 1 )
    {
      srca = (void *)a1[333];
      v101 = sub_15A9930((__int64)srca, (__int64)v7);
      if ( (*(_BYTE *)(v101 + 12) & 1) != 0 )
        return v5;
      v13 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
      if ( !v13 )
        v13 = sub_15A9FE0((__int64)srca, (__int64)v7);
      v15 = sub_1649960(v122);
      v17 = v14;
      v18 = v14;
      v132 = v134;
      v133 = 0x1000000000LL;
      if ( v14 > 0x10 )
      {
        srch = v15;
        sub_16CD150((__int64)&v132, v134, v14, 1, (int)v15, v16);
        v15 = srch;
        v80 = &v132[(unsigned int)v133];
      }
      else
      {
        if ( !v14 )
        {
          LODWORD(v133) = 0;
          goto LABEL_20;
        }
        v80 = v134;
      }
      memcpy(v80, v15, v17);
      LODWORD(v133) = v17 + v133;
      v18 = (unsigned int)v133;
      if ( HIDWORD(v133) - (unsigned __int64)(unsigned int)v133 <= 3 )
      {
        sub_16CD150((__int64)&v132, v134, (unsigned int)v133 + 4LL, 1, v81, v82);
        v18 = (unsigned int)v133;
      }
LABEL_20:
      *(_DWORD *)&v132[v18] = 1953260846;
      v19 = *(_QWORD *)(a2 - 24);
      LODWORD(v133) = v133 + 4;
      v95 = (_BYTE *)v19;
      v21 = sub_1649960(v19);
      v23 = v20;
      v24 = v20;
      v135 = (__int64 **)v137;
      v136 = 0x1000000000LL;
      if ( v20 > 0x10 )
      {
        srcg = v21;
        sub_16CD150((__int64)&v135, v137, v20, 1, (int)v21, v22);
        v21 = srcg;
        v77 = (__int64 **)((char *)v135 + (unsigned int)v136);
      }
      else
      {
        if ( !v20 )
        {
          LODWORD(v136) = 0;
          goto LABEL_23;
        }
        v77 = v137;
      }
      memcpy(v77, v21, v23);
      LODWORD(v136) = v23 + v136;
      v24 = (unsigned int)v136;
      if ( HIDWORD(v136) - (unsigned __int64)(unsigned int)v136 <= 6 )
      {
        sub_16CD150((__int64)&v135, v137, (unsigned int)v136 + 7LL, 1, v78, v79);
        v24 = (unsigned int)v136;
      }
LABEL_23:
      v25 = (char *)v135 + v24;
      *(_DWORD *)v25 = 1885696558;
      *((_WORD *)v25 + 2) = 25441;
      v25[6] = 107;
      LODWORD(v136) = v136 + 7;
      v99 = sub_1643350(*v7);
      v123 = 0;
      v97 = (__int64 *)sub_159C470(v99, 0, 0);
      if ( v103 )
      {
        v91 = (__int64)v7;
        v89 = a2;
        v93 = v13;
        for ( i = 0; i < v103; v123 = i )
        {
          v126[0] = v97;
          v27 = (__int64 *)sub_159C470(v99, i, 0);
          v28 = a1[1];
          v126[1] = v27;
          LOWORD(v143[0]) = 262;
          v141 = (__int64 *)&v135;
          v29 = sub_1709730(v28, v91, v95, v126, 2u, (__int64 *)&v141);
          v30 = a1[1];
          v108 = v29;
          LOWORD(v143[0]) = 262;
          v141 = (__int64 *)&v132;
          srcb = sub_1759FE0(v30, v122, &v123, 1, (__int64 *)&v141);
          v31 = *(_QWORD *)(v101 + 8LL * v123 + 16) | v93;
          v129 = 257;
          v32 = a1[1];
          v33 = v31 & -v31;
          v34 = sub_1648A60(64, 2u);
          v35 = v34;
          if ( v34 )
            sub_15F9650((__int64)v34, (__int64)srcb, (__int64)v108, 0, 0);
          v36 = *(_QWORD *)(v32 + 8);
          if ( v36 )
          {
            srce = *(unsigned __int64 **)(v32 + 16);
            sub_157E9D0(v36 + 40, (__int64)v35);
            v37 = *srce;
            v38 = v35[3] & 7LL;
            v35[4] = srce;
            v37 &= 0xFFFFFFFFFFFFFFF8LL;
            v35[3] = v37 | v38;
            *(_QWORD *)(v37 + 8) = v35 + 3;
            *srce = *srce & 7 | (unsigned __int64)(v35 + 3);
          }
          v39 = v128;
          v40 = v35;
          sub_164B780((__int64)v35, v128);
          v42 = *(_QWORD *)(v32 + 80) == 0;
          v124 = v35;
          if ( v42 )
            goto LABEL_98;
          (*(void (__fastcall **)(__int64, _QWORD **))(v32 + 88))(v32 + 64, &v124);
          v43 = *(_QWORD *)v32;
          if ( *(_QWORD *)v32 )
          {
            v141 = *(__int64 **)v32;
            sub_1623A60((__int64)&v141, v43, 2);
            v44 = v35[6];
            if ( v44 )
              sub_161E7C0((__int64)(v35 + 6), v44);
            v45 = (unsigned __int8 *)v141;
            v35[6] = v141;
            if ( v45 )
              sub_1623210((__int64)&v141, v45, (__int64)(v35 + 6));
          }
          sub_15F9450((__int64)v35, v33);
          v141 = 0;
          v142 = 0;
          v143[0] = 0;
          sub_14A8180(v89, (__int64 *)&v141, 0);
          sub_1626170((__int64)v35, (__int64 *)&v141);
          i = v123 + 1;
        }
      }
      if ( v135 != v137 )
        _libc_free((unsigned __int64)v135);
      v46 = (unsigned __int64)v132;
      if ( v132 != v134 )
LABEL_39:
        _libc_free(v46);
      return 1;
    }
LABEL_79:
    v75 = a1[1];
    LOWORD(v143[0]) = 257;
    LODWORD(v138) = 0;
    v76 = sub_1759FE0(v75, v122, (unsigned int *)&v138, 1, (__int64 *)&v141);
    sub_17767C0((__int64)a1, a2, (__int64 **)v76);
    return 1;
  }
  return v5;
}
