// Function: sub_28B9AC0
// Address: 0x28b9ac0
//
void __fastcall sub_28B9AC0(_DWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v4; // r13
  unsigned int v5; // ebx
  __int64 v6; // r12
  unsigned int v7; // eax
  unsigned int v8; // ecx
  unsigned int v9; // eax
  unsigned int v10; // r15d
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // esi
  unsigned int v14; // eax
  __int64 *v15; // rcx
  __int64 v16; // r8
  __int64 *v17; // r9
  _DWORD *v18; // r15
  __int64 *v19; // r13
  unsigned int v20; // r12d
  unsigned int v21; // eax
  __int64 *v22; // r14
  unsigned int v23; // eax
  __int64 *v24; // rbx
  bool v25; // cf
  bool v26; // zf
  unsigned int v27; // eax
  unsigned int v28; // eax
  __int64 v29; // rax
  __int64 *v30; // r11
  __int64 *v31; // rax
  unsigned int v32; // eax
  int v33; // edx
  int v34; // eax
  int v35; // esi
  int v36; // ecx
  _QWORD *v37; // rax
  _DWORD *v38; // r10
  __int64 *v39; // rsi
  __int64 *v40; // rdi
  char v41; // dl
  __int64 v42; // rcx
  __int64 v43; // rdx
  int v44; // r9d
  __int64 v45; // rsi
  int v46; // ecx
  __int64 v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rcx
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rsi
  _QWORD *v53; // rdx
  unsigned int v54; // esi
  int v55; // esi
  _QWORD *v56; // rsi
  __int64 *v57; // rdi
  int v58; // eax
  unsigned __int64 j; // rdi
  bool v60; // cc
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rdi
  __int64 *v63; // rax
  int v64; // esi
  int v65; // edx
  char v66; // r12
  __int64 *v67; // rdx
  __int64 *v68; // rax
  __int64 v69; // rdi
  int v70; // esi
  __int64 i; // rax
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rdi
  unsigned int v74; // r15d
  unsigned int v75; // eax
  unsigned int v76; // eax
  unsigned int v77; // eax
  __int64 *v78; // rdx
  __int64 *v79; // rax
  __int64 v80; // rsi
  __int64 *v81; // rdx
  __int64 *v82; // rsi
  __int64 v83; // rdi
  __int64 v84; // rax
  int v85; // esi
  unsigned int v86; // esi
  __int64 v87; // r9
  unsigned int v88; // edi
  unsigned int v89; // eax
  __int64 v90; // rsi
  unsigned int v91; // edx
  unsigned int v92; // eax
  unsigned int v93; // eax
  unsigned int v94; // eax
  __int64 v95; // r15
  __int64 v96; // rbx
  __int64 v97; // rdx
  __int64 v98; // rsi
  __int64 v99; // r12
  unsigned int v100; // eax
  unsigned int v101; // esi
  unsigned int v102; // eax
  __int64 v103; // [rsp+0h] [rbp-1C0h]
  _DWORD *v104; // [rsp+8h] [rbp-1B8h]
  __int64 v105; // [rsp+10h] [rbp-1B0h]
  __int64 v106; // [rsp+18h] [rbp-1A8h]
  _QWORD *v107; // [rsp+20h] [rbp-1A0h]
  _QWORD *v108; // [rsp+20h] [rbp-1A0h]
  __int64 *v109; // [rsp+28h] [rbp-198h]
  __int64 *v110; // [rsp+28h] [rbp-198h]
  __int64 *v111; // [rsp+30h] [rbp-190h]
  __int64 *v112; // [rsp+30h] [rbp-190h]
  __int64 *v113; // [rsp+30h] [rbp-190h]
  __int64 v114; // [rsp+38h] [rbp-188h]
  int v115; // [rsp+48h] [rbp-178h]
  int v116; // [rsp+4Ch] [rbp-174h]
  __int64 v117; // [rsp+50h] [rbp-170h]
  __int64 v118; // [rsp+58h] [rbp-168h]
  __int64 v119; // [rsp+60h] [rbp-160h]
  __int64 v120; // [rsp+68h] [rbp-158h]
  __int64 v121; // [rsp+70h] [rbp-150h]
  unsigned int v122; // [rsp+78h] [rbp-148h]
  char v123; // [rsp+7Fh] [rbp-141h]
  int v124; // [rsp+80h] [rbp-140h]
  int v125; // [rsp+84h] [rbp-13Ch]
  __int64 v126; // [rsp+88h] [rbp-138h]
  __int64 v127; // [rsp+90h] [rbp-130h]
  int v128; // [rsp+98h] [rbp-128h]
  int v129; // [rsp+9Ch] [rbp-124h]
  unsigned int v131; // [rsp+B0h] [rbp-110h]
  __int64 v132; // [rsp+B8h] [rbp-108h]
  __int64 v133; // [rsp+D0h] [rbp-F0h]
  __int64 v134; // [rsp+D8h] [rbp-E8h] BYREF
  __int64 v135; // [rsp+E0h] [rbp-E0h]
  __int64 v136; // [rsp+E8h] [rbp-D8h] BYREF
  unsigned int v137; // [rsp+F0h] [rbp-D0h]
  _BYTE v138[4]; // [rsp+128h] [rbp-98h] BYREF
  int v139; // [rsp+12Ch] [rbp-94h]
  __int64 v140; // [rsp+130h] [rbp-90h]
  __int64 v141; // [rsp+138h] [rbp-88h]
  unsigned int v142; // [rsp+140h] [rbp-80h]
  __int64 v143; // [rsp+148h] [rbp-78h]
  int v144; // [rsp+150h] [rbp-70h]
  __int64 v145; // [rsp+158h] [rbp-68h]
  __int64 v146; // [rsp+160h] [rbp-60h]
  int v147; // [rsp+168h] [rbp-58h]
  __int64 v148; // [rsp+170h] [rbp-50h]
  int v149; // [rsp+178h] [rbp-48h]
  int v150; // [rsp+180h] [rbp-40h]
  __int64 v151; // [rsp+188h] [rbp-38h]

  v105 = a3;
  v106 = a2;
  if ( a2 - (__int64)a1 <= 3072 )
    return;
  if ( !a3 )
  {
    v95 = a2;
    goto LABEL_138;
  }
  v103 = (__int64)(a1 + 48);
  v104 = a1 + 70;
  v114 = (__int64)(a1 + 40);
  v132 = (__int64)(a1 + 30);
  while ( 2 )
  {
    --v105;
    v3 = a1[76];
    v4 = (__int64)&a1[16
                    * (((0xAAAAAAAAAAAAAAABLL * ((v106 - (__int64)a1) >> 6)
                       + ((0xAAAAAAAAAAAAAAABLL * ((v106 - (__int64)a1) >> 6)) >> 63))
                      & 0xFFFFFFFFFFFFFFFELL)
                     + (__int64)(0xAAAAAAAAAAAAAAABLL * ((v106 - (__int64)a1) >> 6)) / 2)];
    v5 = *(_DWORD *)(v4 + 112);
    if ( v3 == v5 )
    {
      v6 = v106 - 192;
      if ( (int)sub_C4C880((__int64)(a1 + 78), v4 + 120) < 0 )
        goto LABEL_89;
      v7 = (unsigned int)sub_C4C880(v4 + 120, (__int64)(a1 + 78)) >> 31;
    }
    else
    {
      v6 = v106 - 192;
      if ( v3 < v5 )
        goto LABEL_89;
      LOBYTE(v7) = v3 > v5;
    }
    if ( (_BYTE)v7
      || ((v8 = *(_DWORD *)(v4 + 152), a1[86] == v8)
        ? (v9 = (unsigned int)sub_C4C880((__int64)(a1 + 88), v4 + 160) >> 31)
        : (LOBYTE(v9) = a1[86] < v8),
          !(_BYTE)v9) )
    {
      v10 = *(_DWORD *)(v106 - 80);
      LOBYTE(v11) = v3 < v10;
      if ( v3 == v10 )
        v11 = (unsigned int)sub_C4C880((__int64)(a1 + 78), v106 - 72) >> 31;
      if ( (_BYTE)v11 )
        goto LABEL_20;
      if ( v3 == v10 )
        v12 = (unsigned int)sub_C4C880(v106 - 72, (__int64)(a1 + 78)) >> 31;
      else
        LOBYTE(v12) = v3 > v10;
      if ( !(_BYTE)v12 )
      {
        v13 = *(_DWORD *)(v106 - 40);
        if ( a1[86] == v13 )
          v14 = (unsigned int)sub_C4C880((__int64)(a1 + 88), v106 - 32) >> 31;
        else
          LOBYTE(v14) = a1[86] < v13;
        if ( (_BYTE)v14 )
          goto LABEL_20;
      }
      if ( v5 == v10 )
        v92 = (unsigned int)sub_C4C880(v4 + 120, v106 - 72) >> 31;
      else
        LOBYTE(v92) = v5 < v10;
      if ( !(_BYTE)v92 )
      {
        if ( v5 == v10 )
          v93 = (unsigned int)sub_C4C880(v106 - 72, v4 + 120) >> 31;
        else
          LOBYTE(v93) = v5 > v10;
        if ( (_BYTE)v93
          || ((v94 = *(_DWORD *)(v106 - 40), *(_DWORD *)(v4 + 152) == v94)
            ? (v94 = (unsigned int)sub_C4C880(v4 + 160, v106 - 32) >> 31)
            : (LOBYTE(v94) = *(_DWORD *)(v4 + 152) < v94),
              !(_BYTE)v94) )
        {
LABEL_134:
          sub_28B7290((__int64)a1, v4);
          goto LABEL_21;
        }
      }
LABEL_146:
      sub_28B7290((__int64)a1, v6);
      goto LABEL_21;
    }
LABEL_89:
    v74 = *(_DWORD *)(v106 - 80);
    LOBYTE(v75) = v5 < v74;
    if ( v5 == v74 )
      v75 = (unsigned int)sub_C4C880(v4 + 120, v106 - 72) >> 31;
    if ( (_BYTE)v75 )
      goto LABEL_134;
    if ( v5 == v74 )
      v76 = (unsigned int)sub_C4C880(v106 - 72, v4 + 120) >> 31;
    else
      LOBYTE(v76) = v5 > v74;
    if ( !(_BYTE)v76 )
    {
      v77 = *(_DWORD *)(v106 - 40);
      if ( *(_DWORD *)(v4 + 152) == v77 )
        v77 = (unsigned int)sub_C4C880(v4 + 160, v106 - 32) >> 31;
      else
        LOBYTE(v77) = *(_DWORD *)(v4 + 152) < v77;
      if ( (_BYTE)v77 )
        goto LABEL_134;
    }
    if ( v3 == v74 )
    {
      if ( (int)sub_C4C880((__int64)(a1 + 78), v106 - 72) < 0 )
        goto LABEL_146;
      v100 = (unsigned int)sub_C4C880(v106 - 72, (__int64)(a1 + 78)) >> 31;
    }
    else
    {
      if ( v3 < v74 )
        goto LABEL_146;
      LOBYTE(v100) = v3 > v74;
    }
    if ( !(_BYTE)v100 )
    {
      v101 = *(_DWORD *)(v106 - 40);
      if ( a1[86] == v101 )
        v102 = (unsigned int)sub_C4C880((__int64)(a1 + 88), v106 - 32) >> 31;
      else
        LOBYTE(v102) = a1[86] < v101;
      if ( (_BYTE)v102 )
        goto LABEL_146;
    }
LABEL_20:
    sub_28B7290((__int64)a1, v103);
LABEL_21:
    v18 = v104;
    v19 = (__int64 *)v106;
    v20 = a1[28];
    while ( 1 )
    {
      v21 = v18[6];
      v22 = (__int64 *)(v18 - 22);
      v131 = v21;
      if ( v20 == v21 )
      {
        if ( (int)sub_C4C880((__int64)(v18 + 8), v132) < 0 )
          goto LABEL_88;
        v21 = (unsigned int)sub_C4C880(v132, (__int64)(v18 + 8)) >> 31;
      }
      else
      {
        if ( v20 > v21 )
          goto LABEL_88;
        LOBYTE(v21) = v20 < v21;
      }
      if ( (_BYTE)v21 )
        break;
      v23 = a1[38];
      if ( v18[16] == v23 )
        v23 = (unsigned int)sub_C4C880((__int64)(v18 + 18), v114) >> 31;
      else
        LOBYTE(v23) = v18[16] < v23;
      if ( !(_BYTE)v23 )
        break;
LABEL_88:
      v18 += 48;
    }
    v24 = v19 - 24;
    v19 = v24;
    v25 = v20 < *((_DWORD *)v24 + 28);
    v26 = v20 == *((_DWORD *)v24 + 28);
    if ( v20 != *((_DWORD *)v24 + 28) )
    {
LABEL_30:
      if ( !v25 )
      {
        LOBYTE(v27) = !v25 && !v26;
        goto LABEL_32;
      }
      goto LABEL_101;
    }
    while ( 1 )
    {
      if ( (int)sub_C4C880(v132, (__int64)(v24 + 15)) >= 0 )
      {
        v27 = (unsigned int)sub_C4C880((__int64)(v24 + 15), v132) >> 31;
LABEL_32:
        if ( (_BYTE)v27 )
          break;
        LODWORD(v15) = *((_DWORD *)v24 + 38);
        if ( a1[38] == (_DWORD)v15 )
          v28 = (unsigned int)sub_C4C880(v114, (__int64)(v24 + 20)) >> 31;
        else
          LOBYTE(v28) = a1[38] < (unsigned int)v15;
        if ( !(_BYTE)v28 )
          break;
      }
LABEL_101:
      v24 -= 24;
      v19 = v24;
      v25 = v20 < *((_DWORD *)v24 + 28);
      v26 = v20 == *((_DWORD *)v24 + 28);
      if ( v20 != *((_DWORD *)v24 + 28) )
        goto LABEL_30;
    }
    if ( v24 > v22 )
    {
      v29 = *((_QWORD *)v18 - 11);
      v30 = (__int64 *)(v18 - 20);
      v134 = 0;
      v135 = 1;
      v133 = v29;
      v31 = &v136;
      do
        *v31++ = -4096;
      while ( v31 != (__int64 *)v138 );
      v32 = *((_DWORD *)v22 + 4);
      v122 = v32 >> 1;
      v33 = v135 & 1 | v32 & 0xFFFFFFFE;
      v34 = HIDWORD(v135);
      v35 = v135 & 0xFFFFFFFE | v22[2] & 1;
      LODWORD(v135) = v33;
      *((_DWORD *)v22 + 4) = v35;
      v36 = *(v18 - 17);
      *(v18 - 17) = v34;
      v37 = v18 - 16;
      v124 = v36;
      v38 = v18 - 16;
      HIDWORD(v135) = v36;
      if ( (v33 & 1) != 0 )
      {
        v40 = (__int64 *)(v18 - 20);
        v39 = &v134;
        if ( (v22[2] & 1) != 0 )
        {
          v81 = &v136;
          v82 = (__int64 *)(v18 - 16);
          do
          {
            v83 = *v81;
            *v81++ = *v82;
            *v82++ = v83;
          }
          while ( v138 != (_BYTE *)v81 );
          v131 = v18[6];
          v122 = (unsigned int)v135 >> 1;
          v41 = v22[2] & 1;
          v124 = HIDWORD(v135);
LABEL_44:
          v45 = *((_QWORD *)v18 + 7);
          v123 = *(_BYTE *)v18;
          v138[0] = *(_BYTE *)v18;
          v46 = v18[1];
          v142 = v131;
          v47 = *((_QWORD *)v18 + 4);
          v115 = v46;
          v139 = v46;
          v127 = v47;
          v121 = *((_QWORD *)v18 + 1);
          v140 = v121;
          v48 = *((_QWORD *)v18 + 2);
          v143 = v47;
          LODWORD(v47) = v18[16];
          v118 = v48;
          v141 = v48;
          v117 = v45;
          v125 = v18[10];
          v144 = v125;
          v49 = *((_QWORD *)v18 + 6);
          v146 = v45;
          v119 = v49;
          v145 = v49;
          LODWORD(v49) = v18[20];
          v128 = v47;
          v147 = v47;
          v129 = v49;
          v18[10] = 0;
          v149 = v49;
          v50 = *((_QWORD *)v18 + 9);
          v51 = *((_QWORD *)v18 + 12);
          LODWORD(v47) = v18[22];
          v18[20] = 0;
          v120 = v50;
          v148 = v50;
          v52 = *v24;
          v126 = v51;
          v151 = v51;
          v15 = v24 + 1;
          v116 = v47;
          v150 = v47;
          *((_QWORD *)v18 - 11) = v52;
          if ( !v41 )
          {
            sub_C7D6A0(*((_QWORD *)v18 - 8), 8LL * (unsigned int)*(v18 - 14), 8);
            v37 = v18 - 16;
            v38 = v18 - 16;
            v30 = (__int64 *)(v18 - 20);
            v15 = v24 + 1;
          }
          *((_DWORD *)v22 + 4) = 1;
          v53 = v37;
          *(v18 - 17) = 0;
          do
          {
            if ( v53 )
              *v53 = -4096;
            ++v53;
          }
          while ( v18 != (_DWORD *)v53 );
          v54 = v24[2] & 0xFFFFFFFE;
          *((_DWORD *)v24 + 4) = v22[2] & 0xFFFFFFFE | v24[2] & 1;
          *((_DWORD *)v22 + 4) = v54 | v22[2] & 1;
          v55 = *(v18 - 17);
          *(v18 - 17) = *((_DWORD *)v24 + 5);
          *((_DWORD *)v24 + 5) = v55;
          v56 = v24 + 3;
          v17 = v24 + 3;
          if ( (v22[2] & 1) != 0 )
          {
            v57 = v15;
            if ( (v24[2] & 1) != 0 )
            {
              do
              {
                v16 = *v37;
                *v37++ = *v56;
                *v56++ = v16;
              }
              while ( v18 != (_DWORD *)v37 );
LABEL_56:
              *(_BYTE *)v53 = *((_BYTE *)v24 + 88);
              *((_DWORD *)v53 + 1) = *((_DWORD *)v24 + 23);
              if ( v24 + 12 != (__int64 *)(v18 + 2) )
              {
                v60 = *((_DWORD *)v53 + 10) <= 0x40u;
                v53[1] = v24[12];
                v53[2] = v24[13];
                *((_DWORD *)v53 + 6) = *((_DWORD *)v24 + 28);
                if ( !v60 )
                {
                  v61 = v53[4];
                  if ( v61 )
                  {
                    v107 = v53;
                    v111 = v15;
                    j_j___libc_free_0_0(v61);
                    v53 = v107;
                    v17 = v24 + 3;
                    v15 = v111;
                  }
                }
                v53[4] = v24[15];
                *((_DWORD *)v53 + 10) = *((_DWORD *)v24 + 32);
                *((_DWORD *)v24 + 32) = 0;
              }
              if ( v24 + 17 != (__int64 *)(v18 + 12) )
              {
                v60 = *((_DWORD *)v53 + 20) <= 0x40u;
                v53[6] = v24[17];
                v53[7] = v24[18];
                *((_DWORD *)v53 + 16) = *((_DWORD *)v24 + 38);
                if ( !v60 )
                {
                  v62 = v53[9];
                  if ( v62 )
                  {
                    v108 = v53;
                    v109 = v17;
                    v112 = v15;
                    j_j___libc_free_0_0(v62);
                    v53 = v108;
                    v17 = v109;
                    v15 = v112;
                  }
                }
                v53[9] = v24[20];
                *((_DWORD *)v53 + 20) = *((_DWORD *)v24 + 42);
                *((_DWORD *)v24 + 42) = 0;
              }
              *((_DWORD *)v53 + 22) = *((_DWORD *)v24 + 44);
              v53[12] = v24[23];
              *v24 = v133;
              if ( (v24[2] & 1) == 0 )
              {
                v110 = v17;
                v113 = v15;
                sub_C7D6A0(v24[3], 8LL * *((unsigned int *)v24 + 8), 8);
                v17 = v110;
                v15 = v113;
              }
              v24[2] = 1;
              v63 = v24 + 3;
              do
              {
                if ( v63 )
                  *v63 = -4096;
                ++v63;
              }
              while ( v63 != v24 + 11 );
              v64 = *((_DWORD *)v24 + 4);
              LODWORD(v135) = v64 & 0xFFFFFFFE | v135 & 1;
              v65 = *((_DWORD *)v24 + 5);
              *((_DWORD *)v24 + 5) = v124;
              *((_DWORD *)v24 + 4) = (2 * v122) | v64 & 1;
              HIDWORD(v135) = v65;
              if ( (v64 & 1) != 0 )
              {
                v67 = &v136;
                v68 = &v134;
                if ( (v135 & 1) != 0 )
                {
                  v78 = v24 + 3;
                  v79 = &v136;
                  do
                  {
                    v15 = (__int64 *)*v79;
                    v80 = *v78;
                    *v78++ = *v79++;
                    *(v79 - 1) = v80;
                  }
                  while ( v138 != (_BYTE *)v79 );
                  goto LABEL_78;
                }
              }
              else
              {
                v66 = v135 & 1;
                if ( (v135 & 1) == 0 )
                {
                  v89 = *((_DWORD *)v24 + 8);
                  v90 = v136;
                  v136 = v24[3];
                  v91 = v137;
                  v24[3] = v90;
                  *((_DWORD *)v24 + 8) = v91;
                  v137 = v89;
LABEL_79:
                  v60 = *((_DWORD *)v24 + 32) <= 0x40u;
                  *((_BYTE *)v24 + 88) = v123;
                  *((_DWORD *)v24 + 23) = v115;
                  v24[12] = v121;
                  v24[13] = v118;
                  *((_DWORD *)v24 + 28) = v131;
                  if ( !v60 )
                  {
                    v72 = v24[15];
                    if ( v72 )
                      j_j___libc_free_0_0(v72);
                  }
                  v60 = *((_DWORD *)v24 + 42) <= 0x40u;
                  v24[15] = v127;
                  *((_DWORD *)v24 + 32) = v125;
                  v24[17] = v119;
                  v24[18] = v117;
                  *((_DWORD *)v24 + 38) = v128;
                  if ( !v60 )
                  {
                    v73 = v24[20];
                    if ( v73 )
                      j_j___libc_free_0_0(v73);
                  }
                  v24[20] = v120;
                  *((_DWORD *)v24 + 42) = v129;
                  *((_DWORD *)v24 + 44) = v116;
                  v24[23] = v126;
                  if ( !v66 )
                    sub_C7D6A0(v136, 8LL * v137, 8);
                  v20 = a1[28];
                  goto LABEL_88;
                }
                v67 = v24 + 3;
                v17 = &v136;
                v68 = v15;
                v15 = &v134;
              }
              *((_BYTE *)v68 + 8) |= 1u;
              v69 = v68[2];
              v70 = *((_DWORD *)v68 + 6);
              for ( i = 0; i != 8; ++i )
              {
                v16 = v17[i];
                v67[i] = v16;
              }
              *((_BYTE *)v15 + 8) &= ~1u;
              v15[2] = v69;
              *((_DWORD *)v15 + 6) = v70;
LABEL_78:
              v123 = v138[0];
              v66 = v135 & 1;
              v115 = v139;
              v116 = v150;
              v126 = v151;
              v121 = v140;
              v118 = v141;
              v131 = v142;
              v127 = v143;
              v125 = v144;
              v119 = v145;
              v117 = v146;
              v128 = v147;
              v120 = v148;
              v129 = v149;
              goto LABEL_79;
            }
          }
          else
          {
            if ( (v24[2] & 1) == 0 )
            {
              v84 = *((_QWORD *)v18 - 8);
              *((_QWORD *)v18 - 8) = v24[3];
              v85 = *((_DWORD *)v24 + 8);
              v24[3] = v84;
              LODWORD(v84) = *(v18 - 14);
              *(v18 - 14) = v85;
              *((_DWORD *)v24 + 8) = v84;
              goto LABEL_56;
            }
            v38 = v24 + 3;
            v57 = v30;
            v56 = v37;
            v30 = v15;
          }
          *((_BYTE *)v57 + 8) |= 1u;
          v16 = v57[2];
          v58 = *((_DWORD *)v57 + 6);
          for ( j = 0; j != 16; j += 2LL )
            v56[j / 2] = *(_QWORD *)&v38[j];
          *((_BYTE *)v30 + 8) &= ~1u;
          v30[2] = v16;
          *((_DWORD *)v30 + 6) = v58;
          goto LABEL_56;
        }
      }
      else
      {
        v39 = (__int64 *)(v18 - 20);
        v40 = &v134;
        v41 = v22[2] & 1;
        if ( !v41 )
        {
          v86 = v137;
          v87 = *((_QWORD *)v18 - 8);
          *((_QWORD *)v18 - 8) = v136;
          v88 = *(v18 - 14);
          v136 = v87;
          v137 = v88;
          *(v18 - 14) = v86;
          goto LABEL_44;
        }
      }
      *((_BYTE *)v40 + 8) |= 1u;
      v42 = v40[2];
      v43 = 2;
      v44 = *((_DWORD *)v40 + 6);
      do
      {
        v16 = v39[v43];
        v40[v43++] = v16;
      }
      while ( v43 != 10 );
      *((_BYTE *)v39 + 8) &= ~1u;
      v39[2] = v42;
      *((_DWORD *)v39 + 6) = v44;
      v131 = v18[6];
      v41 = v22[2] & 1;
      v122 = (unsigned int)v135 >> 1;
      v124 = HIDWORD(v135);
      goto LABEL_44;
    }
    sub_28B9AC0((_DWORD)v18 - 88, v106, v105, (_DWORD)v15, v16, (_DWORD)v17, v103);
    if ( (char *)v22 - (char *)a1 <= 3072 )
      return;
    if ( v105 )
    {
      v106 = (__int64)(v18 - 22);
      continue;
    }
    break;
  }
  v95 = (__int64)(v18 - 22);
LABEL_138:
  v96 = v95 - 192;
  sub_28B6C00((__int64)a1, v95);
  do
  {
    v97 = v96;
    v98 = v96;
    v99 = v96 - (_QWORD)a1;
    v96 -= 192;
    sub_28B7870((__int64)a1, v98, v97);
  }
  while ( v99 > 192 );
}
