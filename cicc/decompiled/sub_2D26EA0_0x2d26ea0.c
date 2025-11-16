// Function: sub_2D26EA0
// Address: 0x2d26ea0
//
__int64 __fastcall sub_2D26EA0(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v9; // rdx
  __int64 v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 *v21; // rsi
  __int64 v22; // r11
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 *v25; // rax
  __int64 v26; // r11
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // r12
  unsigned int v31; // eax
  __int64 *v32; // rdx
  __int64 v33; // r10
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rcx
  unsigned int v53; // r14d
  __int64 v54; // rax
  __int64 v55; // rcx
  unsigned int v56; // edx
  _QWORD *v57; // rbx
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  int v64; // edx
  __int64 v65; // rsi
  unsigned int v66; // ecx
  __int64 *v67; // rbx
  __int64 v68; // r9
  char *v69; // r13
  char *v70; // r15
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  _QWORD *v91; // r13
  __int64 v92; // r9
  __int64 v93; // r8
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  _QWORD *v98; // r15
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rdx
  __int64 v105; // rcx
  int v106; // r8d
  int v107; // esi
  int v108; // eax
  __int64 v109; // rdx
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  __int64 v120; // r9
  int v121; // r10d
  int v122; // r10d
  __int64 *v124; // [rsp+60h] [rbp-2B0h]
  __int64 v125; // [rsp+68h] [rbp-2A8h]
  __int64 v126; // [rsp+78h] [rbp-298h] BYREF
  _BYTE *v127; // [rsp+80h] [rbp-290h] BYREF
  __int64 v128; // [rsp+88h] [rbp-288h]
  _BYTE v129[48]; // [rsp+90h] [rbp-280h] BYREF
  char *v130; // [rsp+C0h] [rbp-250h] BYREF
  int v131; // [rsp+C8h] [rbp-248h]
  int v132; // [rsp+100h] [rbp-210h]
  char *v133; // [rsp+108h] [rbp-208h] BYREF
  unsigned int v134; // [rsp+110h] [rbp-200h]
  char *v135; // [rsp+148h] [rbp-1C8h] BYREF
  unsigned int v136; // [rsp+150h] [rbp-1C0h]
  char *v137; // [rsp+188h] [rbp-188h] BYREF
  int v138; // [rsp+190h] [rbp-180h]
  char *v139[40]; // [rsp+1D0h] [rbp-140h] BYREF

  v7 = *(_QWORD *)(a2 + 16);
  v127 = v129;
  v128 = 0x600000000LL;
  if ( v7 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v7 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_38;
    }
    v10 = *(_QWORD *)(v9 + 40);
    if ( !*(_BYTE *)(a3 + 28) )
      goto LABEL_14;
LABEL_4:
    v11 = *(_QWORD **)(a3 + 8);
    v12 = &v11[*(unsigned int *)(a3 + 20)];
    if ( v11 != v12 )
    {
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_11;
      }
LABEL_8:
      v13 = (unsigned int)v128;
      v14 = (unsigned int)v128 + 1LL;
      if ( v14 > HIDWORD(v128) )
      {
        sub_C8D5F0((__int64)&v127, v129, v14, 8u, a5, a6);
        v13 = (unsigned int)v128;
      }
      *(_QWORD *)&v127[8 * v13] = v10;
      LODWORD(v128) = v128 + 1;
    }
LABEL_11:
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        break;
      while ( 1 )
      {
        v15 = *(_QWORD *)(v7 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v15 - 30) > 0xAu )
          break;
        v10 = *(_QWORD *)(v15 + 40);
        if ( *(_BYTE *)(a3 + 28) )
          goto LABEL_4;
LABEL_14:
        if ( sub_C8CA60(a3, v10) )
          goto LABEL_8;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          goto LABEL_16;
      }
    }
LABEL_16:
    v16 = v128;
    v125 = (__int64)(a1 + 38);
    if ( (_DWORD)v128 )
    {
      v17 = *((_QWORD *)a1 + 24);
      v18 = a1[52];
      if ( (_DWORD)v128 != 1 )
      {
        v19 = *(_QWORD *)v127;
        if ( (_DWORD)v18 )
        {
          a6 = (unsigned int)(v18 - 1);
          v20 = a6 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v21 = (__int64 *)(v17 + 272LL * v20);
          v22 = *v21;
          if ( v19 == *v21 )
          {
LABEL_20:
            v19 = *((_QWORD *)v127 + 1);
            v23 = (__int64)(v21 + 1);
          }
          else
          {
            v107 = 1;
            while ( v22 != -4096 )
            {
              v121 = v107 + 1;
              v20 = a6 & (v20 + v107);
              v21 = (__int64 *)(v17 + 272LL * v20);
              v22 = *v21;
              if ( v19 == *v21 )
                goto LABEL_20;
              v107 = v121;
            }
            v19 = *((_QWORD *)v127 + 1);
            v23 = v17 + 272LL * (unsigned int)v18 + 8;
          }
          LODWORD(v24) = a6 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v25 = (__int64 *)(v17 + 272LL * (unsigned int)v24);
          v26 = *v25;
          if ( *v25 == v19 )
          {
LABEL_22:
            v27 = (__int64)(v25 + 1);
          }
          else
          {
            v108 = 1;
            while ( v26 != -4096 )
            {
              v122 = v108 + 1;
              v24 = (unsigned int)a6 & ((_DWORD)v24 + v108);
              v25 = (__int64 *)(v17 + 272 * v24);
              v26 = *v25;
              if ( *v25 == v19 )
                goto LABEL_22;
              v108 = v122;
            }
            v27 = v17 + 272 * v18 + 8;
          }
        }
        else
        {
          v23 = v17 + 8;
          v27 = v17 + 8;
        }
        sub_2D25000((__int64)&v130, v23, v27, *a1, v19, a6);
        v30 = (__int64 *)(v127 + 16);
        v124 = (__int64 *)&v127[8 * (unsigned int)v128];
        if ( v124 != (__int64 *)(v127 + 16) )
        {
          while ( 1 )
          {
            v50 = a1[52];
            v51 = *v30;
            v52 = *((_QWORD *)a1 + 24);
            if ( !(_DWORD)v50 )
              goto LABEL_36;
            v28 = (unsigned int)(v50 - 1);
            v31 = v28 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
            v29 = v31;
            v32 = (__int64 *)(v52 + 272LL * v31);
            v33 = *v32;
            if ( v51 != *v32 )
              break;
LABEL_26:
            sub_2D25000((__int64)v139, (__int64)&v130, (__int64)(v32 + 1), *a1, v28, v29);
            sub_2D23900((__int64)&v130, v139, v34, v35, v36, v37);
            v132 = (int)v139[8];
            sub_2D235D0((__int64)&v133, &v139[9], v38, v39, v40, v41);
            sub_2D235D0((__int64)&v135, &v139[17], v42, v43, v44, v45);
            sub_2D23470((__int64)&v137, &v139[25], v46, v47, v48, v49);
            if ( (char **)v139[25] != &v139[27] )
              _libc_free((unsigned __int64)v139[25]);
            if ( (char **)v139[17] != &v139[19] )
              _libc_free((unsigned __int64)v139[17]);
            if ( (char **)v139[9] != &v139[11] )
              _libc_free((unsigned __int64)v139[9]);
            if ( (char **)v139[0] != &v139[2] )
              _libc_free((unsigned __int64)v139[0]);
            if ( v124 == ++v30 )
              goto LABEL_41;
          }
          v64 = 1;
          while ( v33 != -4096 )
          {
            v29 = (unsigned int)(v64 + 1);
            v31 = v28 & (v64 + v31);
            v32 = (__int64 *)(v52 + 272LL * v31);
            v33 = *v32;
            if ( v51 == *v32 )
              goto LABEL_26;
            v64 = v29;
          }
LABEL_36:
          v32 = (__int64 *)(v52 + 272 * v50);
          goto LABEL_26;
        }
LABEL_41:
        v54 = a1[44];
        v55 = *((_QWORD *)a1 + 20);
        if ( (_DWORD)v54 )
        {
          v56 = (v54 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v57 = (_QWORD *)(v55 + 272LL * v56);
          v58 = *v57;
          if ( a2 == *v57 )
          {
LABEL_43:
            if ( v57 != (_QWORD *)(v55 + 272 * v54) )
            {
              v53 = 0;
              if ( (unsigned __int8)sub_2D26D70((__int64)&v130, (__int64)(v57 + 1)) )
              {
LABEL_45:
                sub_2D22970((unsigned __int64 *)&v130);
                goto LABEL_46;
              }
              sub_2D23900((__int64)(v57 + 1), &v130, v59, v60, v61, v62);
              *((_DWORD *)v57 + 18) = v132;
              sub_2D235D0((__int64)(v57 + 10), &v133, v75, v76, v77, v78);
              sub_2D235D0((__int64)(v57 + 18), &v135, v79, v80, v81, v82);
LABEL_59:
              sub_2D23470((__int64)(v57 + 26), &v137, v83, v84, v85, v86);
LABEL_60:
              v53 = 1;
              goto LABEL_45;
            }
          }
          else
          {
            v106 = 1;
            while ( v58 != -4096 )
            {
              v56 = (v54 - 1) & (v106 + v56);
              v57 = (_QWORD *)(v55 + 272LL * v56);
              v58 = *v57;
              if ( a2 == *v57 )
                goto LABEL_43;
              ++v106;
            }
          }
        }
        v126 = a2;
        if ( (unsigned __int8)sub_2D227E0(v125, &v126, v139) )
          goto LABEL_60;
        v57 = sub_2D26220(v125, &v126, (_QWORD *)v139[0]);
        *v57 = v126;
        v57[1] = v57 + 3;
        v57[2] = 0x600000000LL;
        if ( v131 )
          sub_2D23900((__int64)(v57 + 1), &v130, v104, v105, v85, v86);
        *((_DWORD *)v57 + 18) = v132;
        v57[10] = v57 + 12;
        v57[11] = 0x200000000LL;
        v84 = v134;
        if ( v134 )
          sub_2D235D0((__int64)(v57 + 10), &v133, v104, v134, v85, v86);
        v57[18] = v57 + 20;
        v57[19] = 0x200000000LL;
        v83 = v136;
        if ( v136 )
          sub_2D235D0((__int64)(v57 + 18), &v135, v136, v84, v85, v86);
        v57[26] = v57 + 28;
        v57[27] = 0xC00000000LL;
        if ( !v138 )
          goto LABEL_60;
        goto LABEL_59;
      }
      v65 = *(_QWORD *)v127;
      if ( (_DWORD)v18 )
      {
        v66 = (v18 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
        v67 = (__int64 *)(v17 + 272LL * v66);
        v68 = *v67;
        if ( v65 == *v67 )
          goto LABEL_55;
        while ( v68 != -4096 )
        {
          v66 = (v18 - 1) & (v16 + v66);
          v67 = (__int64 *)(v17 + 272LL * v66);
          v68 = *v67;
          if ( v65 == *v67 )
            goto LABEL_55;
          ++v16;
        }
      }
      v67 = (__int64 *)(v17 + 272 * v18);
LABEL_55:
      v130 = (char *)a2;
      v53 = sub_2D227E0(v125, (__int64 *)&v130, v139);
      if ( (_BYTE)v53 )
      {
        v69 = v139[0];
        v70 = v139[0] + 8;
        if ( (unsigned __int8)sub_2D26D70((__int64)(v67 + 1), (__int64)(v139[0] + 8)) )
        {
          v53 = 0;
        }
        else
        {
          sub_2D23740((__int64)v70, (__int64)(v67 + 1), v71, v72, v73, v74);
          *((_DWORD *)v69 + 18) = *((_DWORD *)v67 + 18);
          sub_2D23390((__int64)(v69 + 80), (__int64)(v67 + 10), v109, v110, v111, v112);
          sub_2D23390((__int64)(v69 + 144), (__int64)(v67 + 18), v113, v114, v115, v116);
          sub_2D232B0((__int64)(v69 + 208), (__int64)(v67 + 26), v117, v118, v119, v120);
        }
      }
      else
      {
        v91 = sub_2D26220(v125, (__int64 *)&v130, (_QWORD *)v139[0]);
        *v91 = v130;
        v91[1] = v91 + 3;
        v91[2] = 0x600000000LL;
        if ( *((_DWORD *)v67 + 4) )
          sub_2D23740((__int64)(v91 + 1), (__int64)(v67 + 1), v87, v88, v89, v90);
        *((_DWORD *)v91 + 18) = *((_DWORD *)v67 + 18);
        v91[10] = v91 + 12;
        v91[11] = 0x200000000LL;
        v92 = *((unsigned int *)v67 + 22);
        if ( (_DWORD)v92 )
          sub_2D23390((__int64)(v91 + 10), (__int64)(v67 + 10), v87, v88, v89, v92);
        v91[18] = v91 + 20;
        v91[19] = 0x200000000LL;
        v93 = *((unsigned int *)v67 + 38);
        if ( (_DWORD)v93 )
          sub_2D23390((__int64)(v91 + 18), (__int64)(v67 + 18), v87, v88, v93, v92);
        v91[26] = v91 + 28;
        v91[27] = 0xC00000000LL;
        if ( *((_DWORD *)v67 + 54) )
          sub_2D232B0((__int64)(v91 + 26), (__int64)(v67 + 26), v87, v88, v93, v92);
        v53 = 1;
      }
      goto LABEL_46;
    }
  }
  else
  {
LABEL_38:
    v125 = (__int64)(a1 + 38);
  }
  memset(v139, 0, 0x108u);
  v139[0] = (char *)&v139[2];
  v139[9] = (char *)&v139[11];
  v139[10] = (char *)0x200000000LL;
  v139[18] = (char *)0x200000000LL;
  v139[25] = (char *)&v139[27];
  v139[26] = (char *)0xC00000000LL;
  v139[17] = (char *)&v139[19];
  v139[1] = (char *)0x600000000LL;
  v126 = a2;
  if ( (unsigned __int8)sub_2D227E0(v125, &v126, &v130) )
  {
    v53 = 0;
    sub_2D22970((unsigned __int64 *)v139);
  }
  else
  {
    v98 = sub_2D26220(v125, &v126, v130);
    v99 = v126;
    v98[2] = 0x600000000LL;
    *v98 = v99;
    v98[1] = v98 + 3;
    if ( LODWORD(v139[1]) )
      sub_2D23900((__int64)(v98 + 1), v139, v94, v95, v96, v97);
    *((_DWORD *)v98 + 18) = v139[8];
    v98[10] = v98 + 12;
    v98[11] = 0x200000000LL;
    if ( LODWORD(v139[10]) )
      sub_2D235D0((__int64)(v98 + 10), &v139[9], v94, v95, v96, v97);
    v98[18] = v98 + 20;
    v98[19] = 0x200000000LL;
    if ( LODWORD(v139[18]) )
      sub_2D235D0((__int64)(v98 + 18), &v139[17], v94, v95, v96, v97);
    v98[26] = v98 + 28;
    v98[27] = 0xC00000000LL;
    if ( LODWORD(v139[26]) )
      sub_2D23470((__int64)(v98 + 26), &v139[25], v94, v95, v96, v97);
    v53 = 1;
    sub_2D22970((unsigned __int64 *)v139);
    sub_2D24C50((__int64)(v98 + 1), *a1, v100, v101, v102, v103);
  }
LABEL_46:
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  return v53;
}
