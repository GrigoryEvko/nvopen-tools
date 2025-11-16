// Function: sub_109E3B0
// Address: 0x109e3b0
//
_BYTE *__fastcall sub_109E3B0(__int64 *a1, __int64 a2, unsigned int a3)
{
  _QWORD *v3; // r15
  _BYTE *v4; // rax
  __int64 v5; // rbx
  unsigned int i; // r14d
  __int64 v7; // r12
  unsigned __int64 v8; // r9
  __int64 *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rdx
  __int64 v14; // r13
  _QWORD **v15; // rdx
  _QWORD *v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // ebx
  _BYTE *v19; // rdi
  __int64 **v20; // r8
  unsigned __int8 ***v21; // rax
  unsigned __int8 **v22; // rcx
  unsigned __int8 *v23; // rdx
  _BYTE *v24; // r12
  __int64 v25; // r10
  __int64 v26; // rbx
  _BYTE *v27; // r13
  char v28; // bl
  __int64 v29; // r14
  char v30; // al
  __int64 v31; // rax
  __int64 v32; // r14
  int v33; // eax
  __int64 v34; // rax
  __int16 v35; // ax
  __int64 *v36; // r13
  __int64 v37; // r15
  char v38; // cl
  void *v39; // r12
  __int64 *v40; // rdi
  __int64 v41; // rsi
  _QWORD *v42; // r12
  _QWORD *v43; // r15
  _QWORD *v44; // rbx
  bool v45; // al
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int8 *v48; // rax
  _BYTE *v49; // rbx
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r11
  __int64 **v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rdx
  unsigned int *v58; // r14
  __int64 v59; // rbx
  __int64 v60; // rdx
  int v61; // ebx
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned int *v64; // r13
  __int64 v65; // rbx
  __int64 v66; // rdx
  int v67; // ebx
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned int *v70; // r13
  __int64 v71; // rbx
  __int64 v72; // rdx
  int v73; // ebx
  __int64 v74; // rax
  __int64 v75; // rdx
  unsigned int *v76; // r13
  __int64 v77; // rbx
  __int64 v78; // rdx
  __int64 v79; // rdx
  __int64 v80; // rax
  _QWORD *v81; // r13
  int v82; // eax
  __int64 v83; // rax
  unsigned int **v84; // r11
  __int64 v85; // rdx
  __int64 v86; // r14
  unsigned int *v87; // rbx
  __int64 v88; // rdx
  __int64 *v89; // rdi
  _BYTE *v90; // rax
  unsigned __int8 v91; // [rsp+0h] [rbp-200h]
  unsigned __int8 v92; // [rsp+2h] [rbp-1FEh]
  unsigned __int8 v93; // [rsp+4h] [rbp-1FCh]
  unsigned __int8 v94; // [rsp+6h] [rbp-1FAh]
  unsigned __int8 v95; // [rsp+8h] [rbp-1F8h]
  unsigned __int8 v96; // [rsp+Ah] [rbp-1F6h]
  unsigned __int8 v97; // [rsp+Ch] [rbp-1F4h]
  unsigned __int8 v98; // [rsp+Eh] [rbp-1F2h]
  __int64 v99; // [rsp+10h] [rbp-1F0h]
  unsigned __int8 v100; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 v101; // [rsp+1Ah] [rbp-1E6h]
  _QWORD *v103; // [rsp+20h] [rbp-1E0h]
  __int64 v105; // [rsp+48h] [rbp-1B8h]
  unsigned int v106; // [rsp+6Ch] [rbp-194h]
  __int64 v107; // [rsp+78h] [rbp-188h]
  __int64 v108; // [rsp+80h] [rbp-180h]
  _BYTE *v109; // [rsp+88h] [rbp-178h]
  int v110; // [rsp+88h] [rbp-178h]
  __int64 v111; // [rsp+88h] [rbp-178h]
  unsigned int **v112; // [rsp+88h] [rbp-178h]
  __int64 v113; // [rsp+90h] [rbp-170h]
  char v114; // [rsp+90h] [rbp-170h]
  __int64 v115; // [rsp+90h] [rbp-170h]
  unsigned __int64 v116; // [rsp+90h] [rbp-170h]
  int v117; // [rsp+90h] [rbp-170h]
  int v118; // [rsp+90h] [rbp-170h]
  char v119; // [rsp+90h] [rbp-170h]
  unsigned int v120; // [rsp+98h] [rbp-168h]
  __int64 **v121; // [rsp+98h] [rbp-168h]
  unsigned int v122; // [rsp+A0h] [rbp-160h]
  char v123; // [rsp+A0h] [rbp-160h]
  __int64 v124; // [rsp+A0h] [rbp-160h]
  __int64 v125; // [rsp+A0h] [rbp-160h]
  __int64 v126; // [rsp+A8h] [rbp-158h]
  __int64 **v127; // [rsp+A8h] [rbp-158h]
  unsigned int v128; // [rsp+B0h] [rbp-150h]
  unsigned int v129[8]; // [rsp+C0h] [rbp-140h] BYREF
  __int16 v130; // [rsp+E0h] [rbp-120h]
  void *v131; // [rsp+F0h] [rbp-110h] BYREF
  _QWORD *v132; // [rsp+F8h] [rbp-108h]
  __int16 v133; // [rsp+110h] [rbp-F0h]
  _BYTE *v134; // [rsp+120h] [rbp-E0h] BYREF
  __int64 v135; // [rsp+128h] [rbp-D8h]
  _BYTE v136[32]; // [rsp+130h] [rbp-D0h] BYREF
  _QWORD v137[15]; // [rsp+150h] [rbp-B0h] BYREF
  _BYTE v138[56]; // [rsp+1C8h] [rbp-38h] BYREF

  v3 = (_QWORD *)a2;
  v106 = *(_DWORD *)(a2 + 8);
  v4 = v137;
  do
  {
    *(_QWORD *)v4 = 0;
    v4 += 40;
    *(v4 - 32) = 0;
    *(v4 - 31) = 0;
    *((_WORD *)v4 - 15) = 0;
  }
  while ( v4 != v138 );
  v134 = v136;
  v135 = 0x400000000LL;
  if ( !v106 )
    goto LABEL_86;
  v5 = 1;
  v120 = 0;
  a2 = (__int64)&v131;
  v108 = v106 - 1 + 2LL;
  for ( i = 0; ; i = v122 )
  {
    v7 = 8 * v5;
    v8 = (unsigned int)v5;
    v9 = *(__int64 **)(*v3 + 8 * v5 - 8);
    if ( !v9 )
    {
      v122 = i;
      v126 = v5 + 1;
      goto LABEL_6;
    }
    v10 = v120;
    v11 = *v9;
    v12 = v120 + 1LL;
    if ( v12 > HIDWORD(v135) )
    {
      v124 = *v9;
      sub_C8D5F0((__int64)&v134, v136, v120 + 1LL, 8u, v12, (unsigned int)v5);
      v10 = (unsigned int)v135;
      v8 = (unsigned int)v5;
      v11 = v124;
      v12 = v120 + 1LL;
    }
    *(_QWORD *)&v134[8 * v10] = v9;
    a2 = (unsigned int)v135;
    v13 = (unsigned int)(v135 + 1);
    LODWORD(v135) = v135 + 1;
    if ( v106 <= (unsigned int)v8 )
    {
      v126 = v5 + 1;
    }
    else
    {
      v126 = v5 + 1;
      v14 = 8 * (v5 + 1 + v106 - 1 - (unsigned int)v8);
      do
      {
        while ( 1 )
        {
          v15 = (_QWORD **)(v7 + *v3);
          v16 = *v15;
          if ( *v15 )
          {
            if ( *v16 == v11 )
              break;
          }
          v7 += 8;
          if ( v14 == v7 )
            goto LABEL_19;
        }
        *v15 = 0;
        v17 = (unsigned int)v135;
        v8 = (unsigned int)v135 + 1LL;
        if ( v8 > HIDWORD(v135) )
        {
          a2 = (__int64)v136;
          v116 = v12;
          v125 = v11;
          sub_C8D5F0((__int64)&v134, v136, (unsigned int)v135 + 1LL, 8u, v12, v8);
          v17 = (unsigned int)v135;
          v12 = v116;
          v11 = v125;
        }
        v7 += 8;
        *(_QWORD *)&v134[8 * v17] = v16;
        LODWORD(v135) = v135 + 1;
      }
      while ( v14 != v7 );
LABEL_19:
      v13 = (unsigned int)v135;
    }
    v18 = v120 + 1;
    if ( v120 + 1 == (_DWORD)v13 )
      break;
    v122 = i + 1;
    v99 = 8LL * v120;
    v105 = 5LL * i;
    v34 = *(_QWORD *)&v134[v99];
    v137[v105] = *(_QWORD *)v34;
    a2 = (__int64)&v137[v105 + 1];
    v107 = a2;
    if ( *(_BYTE *)(v34 + 8) )
    {
      a2 = v34 + 16;
      v117 = v12;
      sub_109E290(v107, (__int64 *)(v34 + 16));
      v13 = (unsigned int)v135;
      LODWORD(v12) = v117;
    }
    else
    {
      v35 = *(_WORD *)(v34 + 10);
      LOBYTE(v137[5 * i + 1]) = 0;
      WORD1(v137[5 * i + 1]) = v35;
    }
    v12 = (unsigned int)v12;
    if ( v18 < (unsigned int)v13 )
    {
      v103 = v3;
      v36 = &v137[v105 + 2];
      v109 = &v138[40 * i + 8];
      while ( 1 )
      {
        v37 = *(_QWORD *)&v134[8 * v12];
        v38 = *(v109 - 120);
        if ( v38 == *(_BYTE *)(v37 + 8) )
        {
          if ( !v38 )
          {
            *((_WORD *)v109 - 59) += *(_WORD *)(v37 + 10);
            goto LABEL_59;
          }
          a2 = v37 + 16;
          v40 = &v137[v105 + 2];
          if ( (void *)*v36 == sub_C33340() )
            goto LABEL_85;
        }
        else
        {
          v114 = *(v109 - 120);
          v39 = sub_C33340();
          if ( v114 )
          {
            sub_109DF40(&v131, v137[v105 + 2], *(__int16 *)(v37 + 10));
            a2 = (__int64)&v131;
            if ( (void *)*v36 == v39 )
              sub_C3D800(v36, (__int64)&v131, 1u);
            else
              sub_C3ADF0((__int64)v36, (__int64)&v131, 1);
            if ( v131 == v39 )
            {
              if ( v132 )
              {
                v41 = 3LL * *(v132 - 1);
                v42 = &v132[v41];
                if ( v132 != &v132[v41] )
                {
                  do
                  {
                    v43 = v42;
                    v42 -= 3;
                    sub_91D830(v42);
                  }
                  while ( v132 != v42 );
                  v41 = 3LL * *(v43 - 4);
                }
                a2 = v41 * 8 + 8;
                j_j_j___libc_free_0_0(v42 - 1);
              }
            }
            else
            {
              sub_C338F0((__int64)&v131);
            }
            goto LABEL_59;
          }
          sub_109CFD0(v107, *(_QWORD *)(v37 + 16));
          a2 = v37 + 16;
          v40 = &v137[v105 + 2];
          if ( (void *)*v36 == v39 )
          {
LABEL_85:
            sub_C3D800(v40, a2, 1u);
            goto LABEL_59;
          }
        }
        sub_C3ADF0((__int64)v40, a2, 1);
LABEL_59:
        v13 = (unsigned int)v135;
        v12 = v18 + 1;
        v18 = v12;
        if ( (unsigned int)v12 >= (unsigned int)v135 )
        {
          v3 = v103;
          break;
        }
      }
    }
    if ( v120 == v13 )
    {
      v120 = v135;
    }
    else
    {
      if ( v120 >= v13 )
      {
        if ( v120 > (unsigned __int64)HIDWORD(v135) )
        {
          a2 = (__int64)v136;
          sub_C8D5F0((__int64)&v134, v136, v120, 8u, v12, v8);
        }
        v51 = &v134[8 * (unsigned int)v135];
        v52 = &v134[v99];
        if ( v51 != (_QWORD *)&v134[v99] )
        {
          do
          {
            if ( v51 )
              *v51 = 0;
            ++v51;
          }
          while ( v52 != v51 );
        }
      }
      LODWORD(v135) = v120;
    }
    if ( LOBYTE(v137[5 * i + 1]) )
    {
      a2 = 40LL * i;
      v44 = &v137[v105 + 2];
      if ( (void *)*v44 == sub_C33340() )
        v44 = (_QWORD *)v137[v105 + 3];
      v45 = (*((_BYTE *)v44 + 20) & 7) == 3;
    }
    else
    {
      v45 = WORD1(v137[5 * i + 1]) == 0;
    }
    if ( !v45 )
    {
      v46 = v120;
      v47 = v120 + 1LL;
      if ( v47 > HIDWORD(v135) )
      {
        a2 = (__int64)v136;
        sub_C8D5F0((__int64)&v134, v136, v47, 8u, v12, v8);
        v46 = (unsigned int)v135;
      }
      *(_QWORD *)&v134[8 * v46] = &v137[v105];
      v120 = v135 + 1;
      LODWORD(v135) = v135 + 1;
    }
LABEL_6:
    v5 = v126;
    if ( v126 == v108 )
      goto LABEL_22;
LABEL_7:
    ;
  }
  v120 = v13;
  v5 = v126;
  v122 = i;
  if ( v126 != v108 )
    goto LABEL_7;
LABEL_22:
  if ( !v120 )
  {
LABEL_86:
    v48 = sub_AD8DD0(*(_QWORD *)(a1[1] + 8), 0.0);
    v19 = v134;
    v24 = v48;
    goto LABEL_87;
  }
  v19 = v134;
  a2 = v120 - 1;
  v20 = (__int64 **)&v134[8 * v120];
  v127 = (__int64 **)v134;
  v21 = (unsigned __int8 ***)v134;
  v121 = v20;
  do
  {
    while ( 1 )
    {
      v22 = *v21;
      v23 = **v21;
      if ( v23 )
      {
        if ( (unsigned int)*v23 - 12 > 1 && (*((_BYTE *)v22 + 8) || ((*((_WORD *)v22 + 5) + 1) & 0xFFFD) != 0) )
          break;
      }
      if ( v20 == (__int64 **)++v21 )
        goto LABEL_30;
    }
    ++v21;
    a2 = (unsigned int)(a2 + 1);
  }
  while ( v20 != (__int64 **)v21 );
LABEL_30:
  if ( a3 < (unsigned int)a2 )
  {
    v24 = 0;
    goto LABEL_87;
  }
  v123 = 0;
  v24 = 0;
  do
  {
    a2 = (__int64)*v127;
    v32 = **v127;
    v28 = *((_BYTE *)*v127 + 8);
    if ( !v32 )
    {
      v55 = *(__int64 ***)(a1[1] + 8);
      if ( v28 )
      {
        a2 += 16;
        v28 = 0;
        v27 = (_BYTE *)sub_AC8EA0(*v55, (__int64 *)a2);
      }
      else
      {
        v27 = sub_AD8DD0((__int64)v55, (double)*(__int16 *)(a2 + 10));
      }
      goto LABEL_37;
    }
    if ( v28 )
    {
      v25 = sub_AC8EA0(**(__int64 ***)(a1[1] + 8), (__int64 *)(a2 + 16));
LABEL_33:
      v26 = *a1;
      v130 = 257;
      if ( *(_BYTE *)(v26 + 108) )
      {
        a2 = 108;
        v27 = (_BYTE *)sub_B35400(v26, 0x6Cu, v32, v25, v128, (__int64)v129, 0, v100, v101);
      }
      else
      {
        v113 = v25;
        a2 = 18;
        v27 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v26 + 80)
                                                                                            + 40LL))(
                         *(_QWORD *)(v26 + 80),
                         18,
                         v32,
                         v25,
                         *(unsigned int *)(v26 + 104));
        if ( !v27 )
        {
          v110 = *(_DWORD *)(v26 + 104);
          v133 = 257;
          v56 = sub_B504D0(18, v32, v113, (__int64)&v131, 0, 0);
          v57 = *(_QWORD *)(v26 + 96);
          v27 = (_BYTE *)v56;
          if ( v57 )
            sub_B99FD0(v56, 3u, v57);
          sub_B45150((__int64)v27, v110);
          a2 = (__int64)v27;
          (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v26 + 88) + 16LL))(
            *(_QWORD *)(v26 + 88),
            v27,
            v129,
            *(_QWORD *)(v26 + 56),
            *(_QWORD *)(v26 + 64));
          v58 = *(unsigned int **)v26;
          v59 = *(_QWORD *)v26 + 16LL * *(unsigned int *)(v26 + 8);
          while ( (unsigned int *)v59 != v58 )
          {
            v60 = *((_QWORD *)v58 + 1);
            a2 = *v58;
            v58 += 4;
            sub_B99FD0((__int64)v27, a2, v60);
          }
        }
      }
      v28 = 0;
      if ( *v27 > 0x1Cu )
      {
LABEL_36:
        a2 = (__int64)v27;
        sub_109CEB0((__int64)a1, (__int64)v27);
      }
LABEL_37:
      if ( !v24 )
        goto LABEL_112;
      goto LABEL_38;
    }
    v33 = *(__int16 *)(a2 + 10);
    if ( ((*(_WORD *)(a2 + 10) + 1) & 0xFFFD) == 0 )
    {
      v27 = (_BYTE *)**v127;
      v28 = (_WORD)v33 == 0xFFFF;
      goto LABEL_37;
    }
    if ( (((_WORD)v33 + 2) & 0xFFFB) != 0 )
    {
      v25 = (__int64)sub_AD8DD0(*(_QWORD *)(a1[1] + 8), (double)v33);
      goto LABEL_33;
    }
    v54 = *a1;
    v130 = 257;
    v28 = (_WORD)v33 == 0xFFFE;
    if ( *(_BYTE *)(v54 + 108) )
    {
      a2 = 102;
      v27 = (_BYTE *)sub_B35400(v54, 0x66u, v32, v32, v128, (__int64)v129, 0, v93, v94);
    }
    else
    {
      v115 = v54;
      a2 = 14;
      v27 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, unsigned __int64))(**(_QWORD **)(v54 + 80) + 40LL))(
                       *(_QWORD *)(v54 + 80),
                       14,
                       v32,
                       v32,
                       *(unsigned int *)(v54 + 104),
                       v8);
      if ( !v27 )
      {
        v82 = *(_DWORD *)(v115 + 104);
        v133 = 257;
        v111 = v115;
        v118 = v82;
        v83 = sub_B504D0(14, v32, v32, (__int64)&v131, 0, 0);
        v84 = (unsigned int **)v111;
        v27 = (_BYTE *)v83;
        v85 = *(_QWORD *)(v111 + 96);
        if ( v85 )
        {
          sub_B99FD0(v83, 3u, v85);
          v84 = (unsigned int **)v111;
        }
        v112 = v84;
        sub_B45150((__int64)v27, v118);
        (*(void (__fastcall **)(unsigned int *, _BYTE *, unsigned int *, unsigned int *, unsigned int *))(*(_QWORD *)v112[11] + 16LL))(
          v112[11],
          v27,
          v129,
          v112[7],
          v112[8]);
        a2 = (__int64)&(*v112)[4 * *((unsigned int *)v112 + 2)];
        if ( *v112 != (unsigned int *)a2 )
        {
          v119 = v28;
          v86 = (__int64)&(*v112)[4 * *((unsigned int *)v112 + 2)];
          v87 = *v112;
          do
          {
            v88 = *((_QWORD *)v87 + 1);
            a2 = *v87;
            v87 += 4;
            sub_B99FD0((__int64)v27, a2, v88);
          }
          while ( (unsigned int *)v86 != v87 );
          v28 = v119;
        }
      }
    }
    if ( *v27 > 0x1Cu )
      goto LABEL_36;
    if ( !v24 )
    {
LABEL_112:
      v123 = v28;
      v24 = v27;
      goto LABEL_45;
    }
LABEL_38:
    v29 = *a1;
    v30 = *(_BYTE *)(*a1 + 108);
    if ( v123 == v28 )
    {
      v130 = 257;
      if ( v30 )
      {
        a2 = 102;
        v24 = (_BYTE *)sub_B35400(v29, 0x66u, (__int64)v24, (__int64)v27, v128, (__int64)v129, 0, v91, v92);
      }
      else
      {
        a2 = 14;
        v53 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _BYTE *, _QWORD))(**(_QWORD **)(v29 + 80) + 40LL))(
                *(_QWORD *)(v29 + 80),
                14,
                v24,
                v27,
                *(unsigned int *)(v29 + 104));
        if ( v53 )
          goto LABEL_105;
        v73 = *(_DWORD *)(v29 + 104);
        v133 = 257;
        v74 = sub_B504D0(14, (__int64)v24, (__int64)v27, (__int64)&v131, 0, 0);
        v75 = *(_QWORD *)(v29 + 96);
        v24 = (_BYTE *)v74;
        if ( v75 )
          sub_B99FD0(v74, 3u, v75);
        sub_B45150((__int64)v24, v73);
        a2 = (__int64)v24;
        (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v29 + 88) + 16LL))(
          *(_QWORD *)(v29 + 88),
          v24,
          v129,
          *(_QWORD *)(v29 + 56),
          *(_QWORD *)(v29 + 64));
        v76 = *(unsigned int **)v29;
        v77 = *(_QWORD *)v29 + 16LL * *(unsigned int *)(v29 + 8);
        if ( *(_QWORD *)v29 != v77 )
        {
          do
          {
            v78 = *((_QWORD *)v76 + 1);
            a2 = *v76;
            v76 += 4;
            sub_B99FD0((__int64)v24, a2, v78);
          }
          while ( (unsigned int *)v77 != v76 );
        }
      }
      goto LABEL_106;
    }
    v130 = 257;
    if ( !v123 )
    {
      if ( v30 )
      {
        a2 = 115;
        v24 = (_BYTE *)sub_B35400(v29, 0x73u, (__int64)v24, (__int64)v27, v128, (__int64)v129, 0, v97, v98);
      }
      else
      {
        a2 = 16;
        v53 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _BYTE *, _QWORD))(**(_QWORD **)(v29 + 80) + 40LL))(
                *(_QWORD *)(v29 + 80),
                16,
                v24,
                v27,
                *(unsigned int *)(v29 + 104));
        if ( v53 )
        {
LABEL_105:
          v24 = (_BYTE *)v53;
          goto LABEL_106;
        }
        v67 = *(_DWORD *)(v29 + 104);
        v133 = 257;
        v68 = sub_B504D0(16, (__int64)v24, (__int64)v27, (__int64)&v131, 0, 0);
        v69 = *(_QWORD *)(v29 + 96);
        v24 = (_BYTE *)v68;
        if ( v69 )
          sub_B99FD0(v68, 3u, v69);
        sub_B45150((__int64)v24, v67);
        a2 = (__int64)v24;
        (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v29 + 88) + 16LL))(
          *(_QWORD *)(v29 + 88),
          v24,
          v129,
          *(_QWORD *)(v29 + 56),
          *(_QWORD *)(v29 + 64));
        v70 = *(unsigned int **)v29;
        v71 = *(_QWORD *)v29 + 16LL * *(unsigned int *)(v29 + 8);
        if ( *(_QWORD *)v29 != v71 )
        {
          do
          {
            v72 = *((_QWORD *)v70 + 1);
            a2 = *v70;
            v70 += 4;
            sub_B99FD0((__int64)v24, a2, v72);
          }
          while ( (unsigned int *)v71 != v70 );
        }
      }
LABEL_106:
      if ( *v24 > 0x1Cu )
        goto LABEL_44;
      goto LABEL_45;
    }
    if ( v30 )
    {
      a2 = 115;
      v24 = (_BYTE *)sub_B35400(v29, 0x73u, (__int64)v27, (__int64)v24, v128, (__int64)v129, 0, v95, v96);
    }
    else
    {
      a2 = 16;
      v31 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _BYTE *, _QWORD))(**(_QWORD **)(v29 + 80) + 40LL))(
              *(_QWORD *)(v29 + 80),
              16,
              v27,
              v24,
              *(unsigned int *)(v29 + 104));
      if ( v31 )
      {
        v24 = (_BYTE *)v31;
      }
      else
      {
        v61 = *(_DWORD *)(v29 + 104);
        v133 = 257;
        v62 = sub_B504D0(16, (__int64)v27, (__int64)v24, (__int64)&v131, 0, 0);
        v63 = *(_QWORD *)(v29 + 96);
        v24 = (_BYTE *)v62;
        if ( v63 )
          sub_B99FD0(v62, 3u, v63);
        sub_B45150((__int64)v24, v61);
        a2 = (__int64)v24;
        (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v29 + 88) + 16LL))(
          *(_QWORD *)(v29 + 88),
          v24,
          v129,
          *(_QWORD *)(v29 + 56),
          *(_QWORD *)(v29 + 64));
        v64 = *(unsigned int **)v29;
        v65 = *(_QWORD *)v29 + 16LL * *(unsigned int *)(v29 + 8);
        if ( *(_QWORD *)v29 != v65 )
        {
          do
          {
            v66 = *((_QWORD *)v64 + 1);
            a2 = *v64;
            v64 += 4;
            sub_B99FD0((__int64)v24, a2, v66);
          }
          while ( (unsigned int *)v65 != v64 );
        }
      }
    }
    v123 = 0;
    if ( *v24 > 0x1Cu )
    {
LABEL_44:
      a2 = (__int64)v24;
      sub_109CEB0((__int64)a1, (__int64)v24);
    }
LABEL_45:
    ++v127;
  }
  while ( v121 != v127 );
  if ( v123 )
  {
    a2 = (__int64)v24;
    v89 = (__int64 *)*a1;
    v133 = 257;
    v90 = (_BYTE *)sub_109D090(v89, (__int64)v24, v129[0], 0, (__int64)&v131, 0);
    v24 = v90;
    if ( *v90 > 0x1Cu )
    {
      a2 = (__int64)v90;
      sub_109CEB0((__int64)a1, (__int64)v90);
    }
  }
  v19 = v134;
LABEL_87:
  if ( v19 != v136 )
    _libc_free(v19, a2);
  v49 = v138;
  do
  {
    v49 -= 40;
    if ( v49[9] )
    {
      if ( *((void **)v49 + 2) == sub_C33340() )
      {
        v79 = *((_QWORD *)v49 + 3);
        if ( v79 )
        {
          v80 = 24LL * *(_QWORD *)(v79 - 8);
          v81 = (_QWORD *)(v79 + v80);
          if ( v79 != v79 + v80 )
          {
            do
            {
              v81 -= 3;
              sub_91D830(v81);
            }
            while ( *((_QWORD **)v49 + 3) != v81 );
          }
          j_j_j___libc_free_0_0(v81 - 1);
        }
      }
      else
      {
        sub_C338F0((__int64)(v49 + 16));
      }
    }
  }
  while ( v49 != (_BYTE *)v137 );
  return v24;
}
