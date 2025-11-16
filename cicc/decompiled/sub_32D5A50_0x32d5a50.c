// Function: sub_32D5A50
// Address: 0x32d5a50
//
__int64 __fastcall sub_32D5A50(_QWORD *a1, __int64 a2)
{
  unsigned __int16 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // r12d
  __int64 v12; // r13
  __int64 v13; // r15
  unsigned __int16 *v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  char v21; // r15
  const void **v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned __int16 *v26; // rdx
  int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int128 v32; // rax
  unsigned int v33; // r12d
  bool v34; // zf
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r13
  unsigned int v41; // r15d
  __int64 v42; // rdi
  unsigned __int64 v43; // r13
  int v44; // eax
  __int64 v45; // r13
  __int128 v46; // rax
  int v47; // r9d
  int v49; // eax
  int v50; // r9d
  char v51; // al
  char v52; // al
  __int64 v53; // r14
  __int128 v54; // rax
  int v55; // r9d
  __int16 v56; // ax
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // r12
  int v60; // eax
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rdi
  const void *v65; // rdx
  char v66; // bl
  int v67; // r9d
  __int64 v68; // rax
  __int64 v69; // r12
  __int128 v70; // rax
  int v71; // r9d
  __int64 v72; // rdx
  int v73; // ebx
  int v74; // edx
  __int64 i; // rax
  __int64 v76; // r12
  int v77; // eax
  char v78; // bl
  int v79; // r9d
  __int64 v80; // r15
  char v81; // al
  __int64 v82; // rdi
  unsigned __int64 v83; // rax
  unsigned __int64 v84; // rdx
  char v85; // dl
  char v86; // al
  __int64 v87; // r13
  unsigned __int8 (__fastcall *v88)(__int64, const void **, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, int *, __int64); // rbx
  __int64 v89; // rax
  __int64 v90; // r9
  int v91; // edx
  __int64 j; // rax
  __int64 v93; // r12
  __int64 v94; // rdx
  __int64 v95; // r13
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // r14
  __int16 v99; // bx
  __int64 v100; // rsi
  __int64 v101; // [rsp+20h] [rbp-160h]
  unsigned int v102; // [rsp+20h] [rbp-160h]
  __int64 v103; // [rsp+28h] [rbp-158h]
  __int64 v104; // [rsp+28h] [rbp-158h]
  __int64 v105; // [rsp+28h] [rbp-158h]
  unsigned int v106; // [rsp+28h] [rbp-158h]
  __int64 v107; // [rsp+30h] [rbp-150h]
  unsigned __int8 v108; // [rsp+30h] [rbp-150h]
  int v109; // [rsp+3Ch] [rbp-144h]
  __int64 v110; // [rsp+40h] [rbp-140h]
  __int64 v111; // [rsp+48h] [rbp-138h]
  __int128 v112; // [rsp+50h] [rbp-130h]
  __int128 v113; // [rsp+60h] [rbp-120h]
  unsigned int v114; // [rsp+70h] [rbp-110h]
  int v115; // [rsp+74h] [rbp-10Ch]
  __int64 v117; // [rsp+80h] [rbp-100h]
  unsigned int v118; // [rsp+88h] [rbp-F8h]
  int v119; // [rsp+88h] [rbp-F8h]
  __int128 v120; // [rsp+90h] [rbp-F0h]
  int v121; // [rsp+90h] [rbp-F0h]
  int v122; // [rsp+90h] [rbp-F0h]
  __int64 v123; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v124; // [rsp+A8h] [rbp-D8h]
  __int64 v125; // [rsp+B0h] [rbp-D0h] BYREF
  int v126; // [rsp+B8h] [rbp-C8h]
  int v127; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+C8h] [rbp-B8h]
  __int64 v129; // [rsp+D0h] [rbp-B0h]
  __int64 v130; // [rsp+D8h] [rbp-A8h]
  const void *v131; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v132; // [rsp+E8h] [rbp-98h]
  __int64 v133; // [rsp+F0h] [rbp-90h]
  __int64 v134; // [rsp+F8h] [rbp-88h]
  const void *v135; // [rsp+100h] [rbp-80h] BYREF
  __int64 v136; // [rsp+108h] [rbp-78h]
  __int128 v137; // [rsp+110h] [rbp-70h] BYREF
  __int64 v138; // [rsp+120h] [rbp-60h]
  _OWORD v139[5]; // [rsp+130h] [rbp-50h] BYREF

  v3 = *(unsigned __int16 **)(a2 + 48);
  LODWORD(v4) = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = *(_QWORD *)(a2 + 40);
  v124 = v5;
  LOWORD(v123) = v4;
  v111 = *(_QWORD *)v6;
  v112 = (__int128)_mm_loadu_si128((const __m128i *)v6);
  v109 = *(_DWORD *)(v6 + 8);
  v113 = (__int128)_mm_loadu_si128((const __m128i *)(v6 + 40));
  v110 = *(_QWORD *)(v6 + 40);
  v120 = (__int128)_mm_loadu_si128((const __m128i *)(v6 + 80));
  v114 = *(_DWORD *)(v6 + 48);
  v117 = *(_QWORD *)(v6 + 80);
  v118 = *(_DWORD *)(v6 + 88);
  v115 = *(_DWORD *)(a2 + 24);
  if ( (_WORD)v4 )
  {
    if ( (unsigned __int16)(v4 - 17) > 0xD3u )
    {
      LOWORD(v127) = v4;
      v128 = v5;
      goto LABEL_4;
    }
    LOWORD(v4) = word_4456580[(int)v4 - 1];
    v8 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v123) )
    {
      v128 = v5;
      LOWORD(v127) = 0;
      goto LABEL_9;
    }
    v56 = sub_3009970((__int64)&v123, a2, v37, v38, v39);
    v57 = v4;
    LOWORD(v4) = v56;
    v8 = v57;
  }
  LOWORD(v127) = v4;
  v128 = v8;
  if ( !(_WORD)v4 )
  {
LABEL_9:
    v129 = sub_3007260((__int64)&v127);
    LODWORD(v7) = v129;
    v130 = v9;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
    goto LABEL_172;
  v7 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v4 - 16];
LABEL_10:
  v10 = *(_QWORD *)(a2 + 80);
  v11 = v7;
  v125 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v125, v10, 1);
  v126 = *(_DWORD *)(a2 + 72);
  if ( !(_DWORD)v7 )
  {
    v23 = sub_33DFBC0(v120, *((_QWORD *)&v120 + 1), 0, 0);
    if ( !v23 )
      goto LABEL_39;
    goto LABEL_46;
  }
  v12 = (unsigned int)(v7 - 1);
  if ( ((unsigned int)v12 & (unsigned int)v7) == 0 )
  {
    v13 = *a1;
    v14 = (unsigned __int16 *)(*(_QWORD *)(v117 + 48) + 16LL * v118);
    v15 = *v14;
    v16 = *((_QWORD *)v14 + 1);
    LOWORD(v131) = v15;
    v132 = v16;
    if ( (_WORD)v15 )
    {
      if ( (unsigned __int16)(v15 - 17) > 0xD3u )
      {
        LOWORD(v139[0]) = v15;
        *((_QWORD *)&v139[0] + 1) = v16;
        goto LABEL_59;
      }
      LOWORD(v15) = word_4456580[v15 - 1];
      v58 = 0;
    }
    else
    {
      v103 = v16;
      if ( !sub_30070B0((__int64)&v131) )
      {
        *((_QWORD *)&v139[0] + 1) = v103;
        LOWORD(v139[0]) = 0;
LABEL_17:
        v19 = sub_3007260((__int64)v139);
        v133 = v19;
        v134 = v20;
        goto LABEL_18;
      }
      LOWORD(v15) = sub_3009970((__int64)&v131, v10, v103, v17, v18);
    }
    LOWORD(v139[0]) = v15;
    *((_QWORD *)&v139[0] + 1) = v58;
    if ( !(_WORD)v15 )
      goto LABEL_17;
LABEL_59:
    if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
      goto LABEL_172;
    v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16];
LABEL_18:
    DWORD2(v139[0]) = v19;
    if ( (unsigned int)v19 > 0x40 )
      sub_C43690((__int64)v139, (unsigned int)v12, 0);
    else
      *(_QWORD *)&v139[0] = (unsigned int)v12;
    v21 = sub_33DD210(v13, v120, *((_QWORD *)&v120 + 1), v139, 0);
    if ( DWORD2(v139[0]) > 0x40 && *(_QWORD *)&v139[0] )
      j_j___libc_free_0_0(*(unsigned __int64 *)&v139[0]);
    if ( v21 )
      goto LABEL_55;
  }
  v22 = (const void **)*((_QWORD *)&v120 + 1);
  v23 = sub_33DFBC0(v120, *((_QWORD *)&v120 + 1), 0, 0);
  if ( !v23 )
  {
LABEL_25:
    if ( ((unsigned int)v12 & v11) == 0 )
    {
      v26 = (unsigned __int16 *)(*(_QWORD *)(v117 + 48) + 16LL * v118);
      v27 = *v26;
      v28 = *((_QWORD *)v26 + 1);
      LOWORD(v135) = v27;
      v136 = v28;
      if ( (_WORD)v27 )
      {
        if ( (unsigned __int16)(v27 - 17) > 0xD3u )
        {
          LOWORD(v137) = v27;
          *((_QWORD *)&v137 + 1) = v28;
LABEL_83:
          if ( (_WORD)v27 != 1 && (unsigned __int16)(v27 - 504) > 7u )
          {
            *((_QWORD *)&v32 + 1) = byte_444C4A0;
            *(_QWORD *)&v32 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v27 - 16];
LABEL_30:
            LODWORD(v132) = v32;
            if ( (unsigned int)v32 > 0x40 )
            {
              v22 = (const void **)v12;
              sub_C43690((__int64)&v131, v12, 0);
            }
            else
            {
              v131 = (const void *)v12;
            }
            if ( *(_DWORD *)(v111 + 24) != 51 )
            {
              v22 = (const void **)*((_QWORD *)&v112 + 1);
              if ( !(unsigned __int8)sub_33E0720(v112, *((_QWORD *)&v112 + 1), 1) )
              {
                if ( *(_DWORD *)(v110 + 24) == 51 )
                  goto LABEL_35;
                goto LABEL_107;
              }
            }
            if ( v115 == 195 )
            {
              if ( *(_DWORD *)(v110 + 24) == 51 )
              {
LABEL_133:
                LODWORD(v136) = v132;
                v76 = *a1;
                if ( (unsigned int)v132 > 0x40 )
                {
                  v22 = &v131;
                  sub_C43780((__int64)&v135, &v131);
                }
                else
                {
                  v135 = v131;
                }
                sub_987160((__int64)&v135, (__int64)v22, *((__int64 *)&v32 + 1), v24, v25);
                v77 = v136;
                LODWORD(v136) = 0;
                DWORD2(v137) = v77;
                *(_QWORD *)&v137 = v135;
                v78 = sub_33DD210(v76, v120, *((_QWORD *)&v120 + 1), &v137, 0);
                if ( DWORD2(v137) > 0x40 && (_QWORD)v137 )
                  j_j___libc_free_0_0(v137);
                if ( (unsigned int)v136 > 0x40 && v135 )
                  j_j___libc_free_0_0((unsigned __int64)v135);
                if ( !v78 )
                  goto LABEL_36;
                v68 = sub_3406EB0(*a1, 190, (unsigned int)&v125, v123, v124, v79, v112, v120);
                goto LABEL_100;
              }
LABEL_107:
              v22 = (const void **)*((_QWORD *)&v113 + 1);
              if ( !(unsigned __int8)sub_33E0720(v113, *((_QWORD *)&v113 + 1), 1) )
                goto LABEL_36;
LABEL_35:
              if ( v115 != 195 )
              {
LABEL_36:
                if ( (unsigned int)v132 > 0x40 && v131 )
                  j_j___libc_free_0_0((unsigned __int64)v131);
                goto LABEL_39;
              }
              goto LABEL_133;
            }
            v59 = *a1;
            v60 = v132;
            LODWORD(v136) = v132;
            if ( (unsigned int)v132 > 0x40 )
            {
              sub_C43780((__int64)&v135, &v131);
              v60 = v136;
              if ( (unsigned int)v136 > 0x40 )
              {
                sub_C43D10((__int64)&v135);
                v60 = v136;
                v65 = v135;
LABEL_92:
                DWORD2(v137) = v60;
                *(_QWORD *)&v137 = v65;
                LODWORD(v136) = 0;
                v66 = sub_33DD210(v59, v120, *((_QWORD *)&v120 + 1), &v137, 0);
                if ( DWORD2(v137) > 0x40 && (_QWORD)v137 )
                  j_j___libc_free_0_0(v137);
                if ( (unsigned int)v136 > 0x40 && v135 )
                  j_j___libc_free_0_0((unsigned __int64)v135);
                if ( !v66 )
                {
                  if ( *(_DWORD *)(v110 + 24) != 51 )
                    sub_33E0720(v113, *((_QWORD *)&v113 + 1), 1);
                  goto LABEL_36;
                }
                v68 = sub_3406EB0(*a1, 192, (unsigned int)&v125, v123, v124, v67, v113, v120);
LABEL_100:
                v36 = v68;
                if ( (unsigned int)v132 > 0x40 && v131 )
                  j_j___libc_free_0_0((unsigned __int64)v131);
                goto LABEL_49;
              }
              v61 = (unsigned __int64)v135;
            }
            else
            {
              v61 = (unsigned __int64)v131;
            }
            v62 = v61;
            v63 = 0;
            v64 = ~v62;
            if ( v60 )
              v63 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v60;
            v65 = (const void *)(v64 & v63);
            v135 = v65;
            goto LABEL_92;
          }
LABEL_172:
          BUG();
        }
        LOWORD(v27) = word_4456580[v27 - 1];
        v72 = 0;
      }
      else
      {
        if ( !sub_30070B0((__int64)&v135) )
        {
          v22 = 0;
          *((_QWORD *)&v137 + 1) = v28;
          LOWORD(v137) = 0;
LABEL_29:
          *(_QWORD *)&v32 = sub_3007260((__int64)&v137);
          v139[0] = v32;
          goto LABEL_30;
        }
        LOWORD(v27) = sub_3009970((__int64)&v135, (__int64)v22, v29, v30, v31);
      }
      LOWORD(v137) = v27;
      *((_QWORD *)&v137 + 1) = v72;
      if ( !(_WORD)v27 )
        goto LABEL_29;
      goto LABEL_83;
    }
LABEL_39:
    v33 = (v115 != 195) + 193;
    if ( v110 == v111
      && v114 == v109
      && (unsigned __int8)sub_328A020(a1[1], v33, v123, v124, *((unsigned __int8 *)a1 + 33)) )
    {
      v36 = sub_3406EB0(
              *a1,
              v33,
              (unsigned int)&v125,
              v123,
              v124,
              v50,
              __PAIR128__(v114 | *((_QWORD *)&v112 + 1) & 0xFFFFFFFF00000000LL, v110),
              v120);
    }
    else
    {
      v34 = (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) == 0;
      v35 = 0;
      if ( !v34 )
        v35 = a2;
      v36 = v35;
    }
    goto LABEL_49;
  }
LABEL_46:
  v40 = *(_QWORD *)(v23 + 96);
  v41 = *(_DWORD *)(v40 + 32);
  v42 = v40 + 24;
  v25 = *(_QWORD *)(*(_QWORD *)(v117 + 48) + 16LL * v118 + 8);
  v24 = *(unsigned __int16 *)(*(_QWORD *)(v117 + 48) + 16LL * v118);
  if ( v41 > 0x40 )
  {
    v101 = *(unsigned __int16 *)(*(_QWORD *)(v117 + 48) + 16LL * v118);
    v104 = *(_QWORD *)(*(_QWORD *)(v117 + 48) + 16LL * v118 + 8);
    v49 = sub_C444A0(v42);
    v42 = v40 + 24;
    v25 = v104;
    v24 = v101;
    if ( v41 - v49 > 0x40 )
      goto LABEL_48;
    v43 = **(_QWORD **)(v40 + 24);
    if ( (unsigned int)v7 <= v43 )
      goto LABEL_48;
  }
  else
  {
    v43 = *(_QWORD *)(v40 + 24);
    if ( (unsigned int)v7 <= v43 )
    {
LABEL_48:
      v119 = v24;
      v121 = v25;
      v44 = sub_C459C0(v42, (unsigned int)v7);
      v45 = *a1;
      *(_QWORD *)&v46 = sub_3400BD0(*a1, v44, (unsigned int)&v125, v119, v121, 0, 0);
      v36 = sub_340F900(v45, *(_DWORD *)(a2 + 24), (unsigned int)&v125, v123, v124, v47, v112, v113, v46);
      goto LABEL_49;
    }
  }
  if ( !v43 )
  {
LABEL_55:
    if ( v115 == 195 )
      v36 = v112;
    else
      v36 = v113;
    goto LABEL_49;
  }
  if ( *(_DWORD *)(v111 + 24) == 51
    || (v105 = v24,
        v107 = v25,
        v51 = sub_33E0720(v112, *((_QWORD *)&v112 + 1), 1),
        LODWORD(v25) = v107,
        LODWORD(v24) = v105,
        v51) )
  {
    if ( v115 == 195 )
      LODWORD(v43) = v7 - v43;
    v53 = *a1;
    *(_QWORD *)&v54 = sub_3400BD0(*a1, v43, (unsigned int)&v125, v24, v25, 0, 0);
    v36 = sub_3406EB0(v53, 192, (unsigned int)&v125, v123, v124, v55, v113, v54);
  }
  else
  {
    if ( *(_DWORD *)(v110 + 24) != 51 )
    {
      v22 = (const void **)*((_QWORD *)&v113 + 1);
      v52 = sub_33E0720(v113, *((_QWORD *)&v113 + 1), 1);
      v25 = v107;
      v24 = v105;
      if ( !v52 )
      {
        if ( (((unsigned __int8)v43 | (unsigned __int8)v7) & 7) == 0 )
        {
          if ( (_WORD)v123 )
          {
            if ( (unsigned __int16)(v123 - 17) <= 0xD3u )
              goto LABEL_71;
          }
          else if ( sub_30070B0((__int64)&v123) )
          {
            goto LABEL_71;
          }
          if ( !*(_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40LL))
            && *(_DWORD *)(v111 + 24) == 298
            && *(_DWORD *)(v110 + 24) == 298 )
          {
            if ( (unsigned __int8)sub_3287C60(v111) )
            {
              if ( (unsigned __int8)sub_3287C60(v110) )
              {
                v73 = sub_2EAC1E0(*(_QWORD *)(v111 + 112));
                if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(v110 + 112)) == v73 )
                {
                  v74 = 1;
                  for ( i = *(_QWORD *)(v111 + 56); i; i = *(_QWORD *)(i + 32) )
                  {
                    if ( !*(_DWORD *)(i + 8) )
                    {
                      if ( !v74 )
                        goto LABEL_162;
                      v74 = 0;
                    }
                  }
                  if ( v74 == 1 )
                  {
LABEL_162:
                    v91 = 1;
                    for ( j = *(_QWORD *)(v110 + 56); j; j = *(_QWORD *)(j + 32) )
                    {
                      if ( !*(_DWORD *)(j + 8) )
                      {
                        if ( !v91 )
                          goto LABEL_71;
                        v91 = 0;
                      }
                    }
                    if ( v91 == 1 )
                      goto LABEL_71;
                  }
                  if ( *(_DWORD *)(v110 + 24) == 298 && (*(_BYTE *)(v110 + 33) & 0xC) == 0 )
                  {
                    v22 = (const void **)v111;
                    if ( *(_DWORD *)(v111 + 24) == 298
                      && (*(_BYTE *)(v111 + 33) & 0xC) == 0
                      && (unsigned __int8)sub_33D01F0(*a1, v111, v110, v11 >> 3, 1) )
                    {
                      v131 = *(const void **)(v110 + 80);
                      if ( v131 )
                        sub_325F5D0((__int64 *)&v131);
                      LODWORD(v132) = *(_DWORD *)(v110 + 72);
                      if ( v115 == 195 )
                        v80 = ((v11 - (unsigned int)v43) % v11) >> 3;
                      else
                        v80 = (unsigned int)v43 >> 3;
                      v81 = sub_2EAC4F0(*(_QWORD *)(v110 + 112));
                      v127 = 0;
                      v82 = *(_QWORD *)(v110 + 112);
                      v83 = (v80 | (1LL << v81)) & -(v80 | (1LL << v81));
                      _BitScanReverse64(&v84, v83);
                      v85 = v84 ^ 0x3F;
                      v34 = v83 == 0;
                      v86 = 64;
                      v102 = *(unsigned __int16 *)(v82 + 32);
                      if ( !v34 )
                        v86 = v85;
                      v108 = 63 - v86;
                      v87 = a1[1];
                      v88 = *(unsigned __int8 (__fastcall **)(__int64, const void **, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, int *, __int64))(*(_QWORD *)v87 + 824LL);
                      v106 = sub_2EAC1E0(v82);
                      v89 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
                      v22 = *(const void ***)(*a1 + 64LL);
                      if ( v88(v87, v22, v89, (unsigned int)v123, v124, v106, v108, v102, &v127, v90) && v127 )
                      {
                        v135 = (const void *)v80;
                        LOBYTE(v136) = 0;
                        v93 = sub_3409320(
                                *a1,
                                *(_QWORD *)(*(_QWORD *)(v110 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v110 + 40) + 48LL),
                                v80,
                                v136,
                                (unsigned int)&v131,
                                0);
                        v95 = v94;
                        sub_32B3E80((__int64)a1, v93, 1, 0, v96, v97);
                        v98 = *a1;
                        LOBYTE(v99) = v108;
                        v100 = *(_QWORD *)(v110 + 112);
                        HIBYTE(v99) = 1;
                        v139[0] = _mm_loadu_si128((const __m128i *)(v100 + 40));
                        v139[1] = _mm_loadu_si128((const __m128i *)(v100 + 56));
                        v122 = *(unsigned __int16 *)(v100 + 32);
                        sub_327C6E0((__int64)&v137, (__int64 *)v100, v80);
                        v36 = sub_33F1F00(
                                v98,
                                v123,
                                v124,
                                (unsigned int)&v131,
                                **(_QWORD **)(v110 + 40),
                                *(_QWORD *)(*(_QWORD *)(v110 + 40) + 8LL),
                                v93,
                                v95,
                                v137,
                                v138,
                                v99,
                                v122,
                                (__int64)v139,
                                0);
                        sub_3417D40(*a1, v111, v36, 1);
                        sub_3417D40(*a1, v110, v36, 1);
                        sub_9C6650(&v131);
                        goto LABEL_49;
                      }
                      sub_9C6650(&v131);
                    }
                  }
                }
              }
            }
          }
        }
LABEL_71:
        if ( !v11 )
          goto LABEL_39;
        v12 = v11 - 1;
        goto LABEL_25;
      }
    }
    if ( v115 != 195 )
      LODWORD(v43) = v7 - v43;
    v69 = *a1;
    *(_QWORD *)&v70 = sub_3400BD0(*a1, v43, (unsigned int)&v125, v24, v25, 0, 0);
    v36 = sub_3406EB0(v69, 190, (unsigned int)&v125, v123, v124, v71, v112, v70);
  }
LABEL_49:
  if ( v125 )
    sub_B91220((__int64)&v125, v125);
  return v36;
}
