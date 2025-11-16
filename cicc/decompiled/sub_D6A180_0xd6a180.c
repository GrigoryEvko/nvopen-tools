// Function: sub_D6A180
// Address: 0xd6a180
//
__int64 __fastcall sub_D6A180(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9
  bool v6; // zf
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r13
  __int64 v11; // rdx
  _QWORD *v12; // r12
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 *v24; // rsi
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // r13
  signed int v28; // eax
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // rdi
  __int64 *v32; // rbx
  unsigned int v33; // r15d
  int v34; // eax
  unsigned int v35; // eax
  unsigned __int64 v36; // r13
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 *v39; // r15
  __int64 *v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // rax
  _BYTE *v43; // r12
  _BYTE *v44; // r15
  __int64 v45; // rax
  unsigned int v46; // eax
  __int64 v47; // rbx
  char *v48; // rax
  __int64 *v49; // r13
  __int64 *k; // r15
  __int64 v51; // r12
  __int64 *v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rdx
  unsigned int v55; // eax
  __int64 v56; // r14
  unsigned __int64 v57; // r15
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 *v63; // rdx
  __int64 v64; // rcx
  __int64 *v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  __int64 v69; // r10
  __int64 v70; // rdi
  _QWORD *v71; // rax
  unsigned __int64 v72; // rdx
  int v73; // eax
  __int64 *v74; // rdx
  __int64 v75; // rdi
  _QWORD *v76; // rax
  __int64 *v77; // rax
  unsigned __int64 v78; // r8
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 *v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 *v85; // rax
  unsigned __int64 v86; // rax
  __int64 v87; // r13
  int v88; // ebx
  int v89; // eax
  __int64 *v90; // r15
  unsigned int j; // r12d
  __int64 v92; // r13
  _BYTE *v93; // rax
  __int64 v94; // rcx
  __int64 v95; // r8
  __int64 v96; // rax
  __int64 v97; // r9
  __int64 v98; // rdx
  __int64 i; // r10
  __int64 v100; // rax
  __int64 v101; // r11
  __int64 v102; // rax
  unsigned int v103; // r15d
  __int64 v104; // r10
  __int64 v105; // rdx
  __int64 v106; // rax
  unsigned int v107; // eax
  __int64 v108; // rdx
  __int64 v109; // rdx
  __int64 v110; // rcx
  unsigned int v111; // eax
  __int64 v112; // rcx
  unsigned __int64 v113; // rdx
  unsigned int v114; // eax
  __int64 v115; // [rsp+18h] [rbp-528h]
  int v116; // [rsp+20h] [rbp-520h]
  unsigned __int8 v117; // [rsp+20h] [rbp-520h]
  __int64 *v118; // [rsp+28h] [rbp-518h]
  __int64 v119; // [rsp+38h] [rbp-508h]
  unsigned int v120; // [rsp+38h] [rbp-508h]
  __int64 v121; // [rsp+38h] [rbp-508h]
  __int64 v122; // [rsp+40h] [rbp-500h]
  unsigned int v123; // [rsp+40h] [rbp-500h]
  unsigned int v124; // [rsp+40h] [rbp-500h]
  __int64 v125; // [rsp+40h] [rbp-500h]
  int v126; // [rsp+40h] [rbp-500h]
  unsigned __int64 v127; // [rsp+40h] [rbp-500h]
  unsigned int v128; // [rsp+40h] [rbp-500h]
  unsigned int v129; // [rsp+4Ch] [rbp-4F4h]
  _BYTE *v130; // [rsp+50h] [rbp-4F0h] BYREF
  __int64 v131; // [rsp+58h] [rbp-4E8h]
  _BYTE v132[64]; // [rsp+60h] [rbp-4E0h] BYREF
  __int64 v133; // [rsp+A0h] [rbp-4A0h] BYREF
  char *v134; // [rsp+A8h] [rbp-498h]
  __int64 v135; // [rsp+B0h] [rbp-490h]
  int v136; // [rsp+B8h] [rbp-488h]
  char v137; // [rsp+BCh] [rbp-484h]
  char v138; // [rsp+C0h] [rbp-480h] BYREF
  __int64 v139; // [rsp+140h] [rbp-400h] BYREF
  __int64 *v140; // [rsp+148h] [rbp-3F8h]
  __int64 v141; // [rsp+150h] [rbp-3F0h]
  int v142; // [rsp+158h] [rbp-3E8h]
  char v143; // [rsp+15Ch] [rbp-3E4h]
  char v144; // [rsp+160h] [rbp-3E0h] BYREF
  _BYTE *v145; // [rsp+1E0h] [rbp-360h] BYREF
  __int64 v146; // [rsp+1E8h] [rbp-358h]
  _BYTE v147[256]; // [rsp+1F0h] [rbp-350h] BYREF
  _BYTE *v148; // [rsp+2F0h] [rbp-250h] BYREF
  __int64 v149; // [rsp+2F8h] [rbp-248h]
  _BYTE v150[576]; // [rsp+300h] [rbp-240h] BYREF

  v3 = *(_QWORD *)a1;
  v148 = v150;
  v115 = a2;
  v149 = 0x2000000000LL;
  sub_B19440(v3);
  v6 = *(_BYTE *)(a1 + 16) == 0;
  v146 = 0x2000000000LL;
  v145 = v147;
  v134 = &v138;
  v133 = 0;
  v135 = 16;
  v136 = 0;
  v137 = 1;
  v139 = 0;
  v140 = (__int64 *)&v144;
  v141 = 16;
  v142 = 0;
  v143 = 1;
  if ( v6 )
    goto LABEL_2;
  v133 = 1;
  v107 = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 20LL) - *(_DWORD *)(*(_QWORD *)(a1 + 24) + 24LL);
  v108 = 1;
  if ( v107 > 0x10 )
  {
    v112 = 2863311531LL;
    a2 = 128;
    v113 = v107 / 3uLL;
    v114 = v107 + v113 - 1;
    if ( v114 )
    {
      _BitScanReverse(&v114, v114);
      v112 = 33 - (v114 ^ 0x1F);
      a2 = (unsigned int)(1 << (33 - (v114 ^ 0x1F)));
      if ( (unsigned int)a2 < 0x80 )
        a2 = 128;
    }
    sub_C8CB60((__int64)&v133, a2, v113, v112, v4, v5);
    v108 = v139 + 1;
    v107 = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 20LL) - *(_DWORD *)(*(_QWORD *)(a1 + 24) + 24LL);
  }
  v139 = v108;
  if ( !v107 )
    goto LABEL_2;
  if ( v143 )
  {
    v109 = v107 - 1;
    if ( v107 > (unsigned int)v141 )
    {
LABEL_167:
      v110 = 2863311531LL;
      a2 = 128;
      v111 = v109 + v107 / 3;
      if ( v111 )
      {
        _BitScanReverse(&v111, v111);
        v110 = 33 - (v111 ^ 0x1F);
        a2 = (unsigned int)(1 << (33 - (v111 ^ 0x1F)));
        if ( (unsigned int)a2 < 0x80 )
          a2 = 128;
      }
      sub_C8CB60((__int64)&v139, a2, v109, v110, v4, v5);
    }
  }
  else
  {
    v109 = v107 - 1;
    a2 = (unsigned int)(4 * v109);
    if ( (unsigned int)a2 >= 3 * (int)v141 )
      goto LABEL_167;
  }
LABEL_2:
  v7 = *(_QWORD *)(a1 + 32);
  v8 = *(_QWORD **)(v7 + 8);
  if ( *(_BYTE *)(v7 + 28) )
    v9 = *(unsigned int *)(v7 + 20);
  else
    v9 = *(unsigned int *)(v7 + 16);
  v10 = &v8[v9];
  if ( v8 != v10 )
  {
    while ( 1 )
    {
      v11 = *v8;
      v12 = v8;
      if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v8 )
        goto LABEL_7;
    }
    while ( v10 != v12 )
    {
      v53 = *(_QWORD *)a1;
      if ( v11 )
      {
        v54 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
        v55 = v54;
      }
      else
      {
        v54 = 0;
        v55 = 0;
      }
      if ( v55 < *(_DWORD *)(v53 + 32) )
      {
        v56 = *(_QWORD *)(*(_QWORD *)(v53 + 24) + 8 * v54);
        if ( v56 )
        {
          v57 = ((unsigned __int64)*(unsigned int *)(v56 + 72) << 32) | *(unsigned int *)(v56 + 16);
          v58 = (unsigned int)v149;
          v59 = (unsigned int)v149 + 1LL;
          if ( v59 > HIDWORD(v149) )
          {
            sub_C8D5F0((__int64)&v148, v150, v59, 0x10u, v4, v5);
            v58 = (unsigned int)v149;
          }
          v60 = (__int64 *)&v148[16 * v58];
          *v60 = v56;
          v61 = (__int64)v148;
          v60[1] = v57;
          LODWORD(v149) = v149 + 1;
          v62 = 16LL * (unsigned int)v149;
          a2 = (v62 >> 4) - 1;
          sub_D67B60(v61, a2, 0, *(_QWORD *)(v61 + v62 - 16), *(_QWORD *)(v61 + v62 - 8));
          if ( !v143 )
          {
LABEL_156:
            a2 = v56;
            sub_C8CC70((__int64)&v139, v56, (__int64)v63, v64, v4, v5);
            goto LABEL_77;
          }
          v65 = v140;
          v64 = HIDWORD(v141);
          v63 = &v140[HIDWORD(v141)];
          if ( v140 == v63 )
          {
LABEL_157:
            if ( HIDWORD(v141) >= (unsigned int)v141 )
              goto LABEL_156;
            ++HIDWORD(v141);
            *v63 = v56;
            ++v139;
          }
          else
          {
            while ( v56 != *v65 )
            {
              if ( v63 == ++v65 )
                goto LABEL_157;
            }
          }
        }
      }
LABEL_77:
      v66 = v12 + 1;
      if ( v12 + 1 == v10 )
        break;
      while ( 1 )
      {
        v11 = *v66;
        v12 = v66;
        if ( *v66 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v10 == ++v66 )
          goto LABEL_7;
      }
    }
  }
LABEL_7:
  result = (unsigned int)v149;
  v14 = a1;
  if ( !(_DWORD)v149 )
    goto LABEL_56;
  do
  {
    v15 = (__int64)v148;
    v16 = (unsigned int)result;
    a2 = *((unsigned int *)v148 + 2);
    v17 = *(_QWORD *)v148;
    v18 = 16LL * (unsigned int)result;
    v129 = *((_DWORD *)v148 + 2);
    if ( v16 != 1 )
    {
      v93 = &v148[v18];
      v94 = *((_QWORD *)v93 - 2);
      v95 = *((_QWORD *)v93 - 1);
      *((_QWORD *)v93 - 2) = v17;
      v93 -= 16;
      *((_DWORD *)v93 + 2) = *(_DWORD *)(v15 + 8);
      *((_DWORD *)v93 + 3) = *(_DWORD *)(v15 + 12);
      v96 = (__int64)&v93[-v15];
      v97 = v96 >> 4;
      v98 = ((v96 >> 4) - 1) / 2;
      if ( v96 <= 32 )
      {
        a2 = 0;
      }
      else
      {
        for ( i = 0; ; i = a2 )
        {
          a2 = 2 * (i + 1);
          v100 = 32 * (i + 1);
          v101 = v15 + v100 - 16;
          v102 = v15 + v100;
          v103 = *(_DWORD *)(v101 + 8);
          if ( *(_DWORD *)(v102 + 8) < v103
            || *(_DWORD *)(v102 + 8) == v103 && *(_DWORD *)(v102 + 12) < *(_DWORD *)(v101 + 12) )
          {
            --a2;
            v102 = v15 + 16 * a2;
          }
          v104 = v15 + 16 * i;
          *(_QWORD *)v104 = *(_QWORD *)v102;
          *(_DWORD *)(v104 + 8) = *(_DWORD *)(v102 + 8);
          *(_DWORD *)(v104 + 12) = *(_DWORD *)(v102 + 12);
          if ( v98 <= a2 )
            break;
        }
      }
      if ( (v97 & 1) == 0 && (v97 - 2) / 2 == a2 )
      {
        v105 = v15 + 32 * (a2 + 1) - 16;
        v106 = v15 + 16 * a2;
        *(_QWORD *)v106 = *(_QWORD *)v105;
        *(_DWORD *)(v106 + 8) = *(_DWORD *)(v105 + 8);
        a2 = 2 * (a2 + 1) - 1;
        *(_DWORD *)(v106 + 12) = *(_DWORD *)(v105 + 12);
      }
      sub_D67B60(v15, a2, 0, v94, v95);
    }
    v19 = (unsigned int)v146;
    LODWORD(v149) = v149 - 1;
    v20 = (unsigned int)v146 + 1LL;
    if ( v20 > HIDWORD(v146) )
    {
      a2 = (__int64)v147;
      sub_C8D5F0((__int64)&v145, v147, v20, 8u, v4, v5);
      v19 = (unsigned int)v146;
    }
    *(_QWORD *)&v145[8 * v19] = v17;
    v6 = (_DWORD)v146 == -1;
    v21 = v146 + 1;
    LODWORD(v146) = v146 + 1;
    if ( v6 )
      goto LABEL_55;
    do
    {
      v22 = (__int64)v145;
      v23 = v21;
      v24 = *(__int64 **)&v145[8 * v21 - 8];
      LODWORD(v146) = v21 - 1;
      v118 = v24;
      v122 = *v24;
      a2 = v122;
      v119 = *(_QWORD *)(v14 + 8);
      v25 = *(_QWORD *)(v122 + 48);
      if ( !v119 )
      {
        v86 = v25 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v86 == v122 + 48 )
          goto LABEL_142;
        if ( !v86 )
LABEL_133:
          BUG();
        v87 = v86 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v86 - 24) - 30 > 0xA )
        {
LABEL_142:
          HIDWORD(v131) = 8;
          v130 = v132;
          v43 = v132;
          v89 = 0;
          v126 = 0;
        }
        else
        {
          a2 = (__int64)v132;
          v88 = sub_B46E30(v87);
          v126 = v88;
          v130 = v132;
          v131 = 0x800000000LL;
          if ( (unsigned __int64)v88 > 8 )
          {
            sub_C8D5F0((__int64)&v130, v132, v88, 8u, v4, v5);
            v43 = v130;
            v89 = v131;
            v90 = (__int64 *)&v130[8 * (unsigned int)v131];
          }
          else
          {
            v43 = v132;
            v89 = 0;
            v90 = (__int64 *)v132;
          }
          if ( v88 )
          {
            for ( j = 0; j != v88; ++j )
            {
              if ( v90 )
              {
                a2 = j;
                *v90 = sub_B46EC0(v87, j);
              }
              ++v90;
            }
            v89 = v131;
            v43 = v130;
          }
        }
        LODWORD(v131) = v89 + v126;
        v42 = (unsigned int)(v89 + v126);
        goto LABEL_33;
      }
      v26 = v25 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v26 == v122 + 48 )
        goto LABEL_132;
      if ( !v26 )
        goto LABEL_133;
      v27 = v26 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30 > 0xA )
      {
LABEL_132:
        HIDWORD(v131) = 8;
        v130 = v132;
        v34 = 0;
        v116 = 0;
      }
      else
      {
        v28 = sub_B46E30(v27);
        v31 = v28;
        v32 = (__int64 *)v132;
        v131 = 0x800000000LL;
        v33 = v28;
        v34 = 0;
        v116 = v31;
        v130 = v132;
        if ( v31 > 8 )
        {
          sub_C8D5F0((__int64)&v130, v132, v31, 8u, v29, v30);
          v34 = v131;
          v32 = (__int64 *)&v130[8 * (unsigned int)v131];
        }
        if ( v33 )
        {
          do
          {
            --v33;
            if ( v32 )
              *v32 = sub_B46EC0(v27, v33);
            ++v32;
          }
          while ( v33 );
          v34 = v131;
        }
      }
      LODWORD(v131) = v116 + v34;
      sub_B1C8F0((__int64)&v130);
      v117 = *(_BYTE *)(v119 + 8);
      a2 = v117;
      v22 = v117 & 1;
      if ( (v117 & 1) != 0 )
      {
        v23 = v119 + 16;
        a2 = 3;
      }
      else
      {
        v23 = *(_QWORD *)(v119 + 16);
        v84 = *(unsigned int *)(v119 + 24);
        if ( !(_DWORD)v84 )
          goto LABEL_136;
        a2 = (unsigned int)(v84 - 1);
      }
      v35 = a2 & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
      v36 = v23 + 72LL * v35;
      v37 = *(_QWORD *)v36;
      if ( v122 != *(_QWORD *)v36 )
      {
        v4 = 1;
        while ( v37 != -4096 )
        {
          v5 = (unsigned int)(v4 + 1);
          v35 = a2 & (v4 + v35);
          v36 = v23 + 72LL * v35;
          v37 = *(_QWORD *)v36;
          if ( v122 == *(_QWORD *)v36 )
            goto LABEL_26;
          v4 = (unsigned int)v5;
        }
        if ( (_BYTE)v22 )
        {
          v92 = 288;
        }
        else
        {
          v84 = *(unsigned int *)(v119 + 24);
LABEL_136:
          v92 = 72 * v84;
        }
        v36 = v23 + v92;
      }
LABEL_26:
      v38 = 288;
      if ( !(_BYTE)v22 )
        v38 = 72LL * *(unsigned int *)(v119 + 24);
      if ( v36 != v23 + v38 )
      {
        v39 = *(__int64 **)(v36 + 8);
        v40 = &v39[*(unsigned int *)(v36 + 16)];
        while ( v40 != v39 )
        {
          v41 = *v39++;
          sub_B1CA60((__int64)&v130, v41);
        }
        a2 = (__int64)&v130[8 * (unsigned int)v131];
        sub_D67E60(
          (__int64)&v130,
          (char *)a2,
          *(char **)(v36 + 40),
          (char *)(*(_QWORD *)(v36 + 40) + 8LL * *(unsigned int *)(v36 + 48)));
      }
      v42 = (unsigned int)v131;
      v43 = v130;
LABEL_33:
      v44 = &v43[8 * v42];
      if ( v44 == v43 )
        goto LABEL_45;
      do
      {
        while ( 1 )
        {
          v45 = *(_QWORD *)v43;
          v22 = *(_QWORD *)v14;
          if ( *(_QWORD *)v43 )
          {
            v23 = (unsigned int)(*(_DWORD *)(v45 + 44) + 1);
            v46 = *(_DWORD *)(v45 + 44) + 1;
          }
          else
          {
            v23 = 0;
            v46 = 0;
          }
          if ( v46 >= *(_DWORD *)(v22 + 32) )
            BUG();
          v47 = *(_QWORD *)(*(_QWORD *)(v22 + 24) + 8 * v23);
          v4 = *(unsigned int *)(v47 + 16);
          if ( v129 < (unsigned int)v4 )
            goto LABEL_43;
          if ( !v137 )
            goto LABEL_89;
          v48 = v134;
          v23 = HIDWORD(v135);
          v22 = (__int64)&v134[8 * HIDWORD(v135)];
          if ( v134 != (char *)v22 )
          {
            while ( v47 != *(_QWORD *)v48 )
            {
              v48 += 8;
              if ( (char *)v22 == v48 )
                goto LABEL_106;
            }
            goto LABEL_43;
          }
LABEL_106:
          if ( HIDWORD(v135) < (unsigned int)v135 )
          {
            v23 = (unsigned int)++HIDWORD(v135);
            *(_QWORD *)v22 = v47;
            ++v133;
          }
          else
          {
LABEL_89:
            a2 = v47;
            v123 = *(_DWORD *)(v47 + 16);
            sub_C8CC70((__int64)&v133, v47, v22, v23, v4, v5);
            v4 = v123;
            if ( !(_BYTE)v22 )
              goto LABEL_43;
          }
          v69 = *(_QWORD *)v47;
          if ( !*(_BYTE *)(v14 + 16) )
            goto LABEL_96;
          v70 = *(_QWORD *)(v14 + 24);
          if ( *(_BYTE *)(v70 + 28) )
          {
            v71 = *(_QWORD **)(v70 + 8);
            v22 = (__int64)&v71[*(unsigned int *)(v70 + 20)];
            if ( v71 == (_QWORD *)v22 )
              goto LABEL_43;
            while ( v69 != *v71 )
            {
              if ( (_QWORD *)v22 == ++v71 )
                goto LABEL_43;
            }
LABEL_96:
            v72 = *(unsigned int *)(v115 + 8);
            v23 = *(unsigned int *)(v115 + 12);
            v73 = *(_DWORD *)(v115 + 8);
            if ( v72 < v23 )
              goto LABEL_97;
            goto LABEL_117;
          }
          a2 = *(_QWORD *)v47;
          v120 = v4;
          v125 = *(_QWORD *)v47;
          v85 = sub_C8CA60(v70, *(_QWORD *)v47);
          v69 = v125;
          v4 = v120;
          if ( !v85 )
            goto LABEL_43;
          v72 = *(unsigned int *)(v115 + 8);
          v23 = *(unsigned int *)(v115 + 12);
          v73 = *(_DWORD *)(v115 + 8);
          if ( v72 < v23 )
          {
LABEL_97:
            v23 = *(_QWORD *)v115;
            v74 = (__int64 *)(*(_QWORD *)v115 + 8 * v72);
            if ( v74 )
            {
              *v74 = v69;
              v73 = *(_DWORD *)(v115 + 8);
            }
            *(_DWORD *)(v115 + 8) = v73 + 1;
            goto LABEL_100;
          }
LABEL_117:
          if ( v23 < v72 + 1 )
          {
            a2 = v115 + 16;
            v121 = v69;
            v128 = v4;
            sub_C8D5F0(v115, (const void *)(v115 + 16), v72 + 1, 8u, v4, v5);
            v72 = *(unsigned int *)(v115 + 8);
            v69 = v121;
            v4 = v128;
          }
          *(_QWORD *)(*(_QWORD *)v115 + 8 * v72) = v69;
          ++*(_DWORD *)(v115 + 8);
LABEL_100:
          v75 = *(_QWORD *)(v14 + 32);
          if ( *(_BYTE *)(v75 + 28) )
            break;
          a2 = v69;
          v124 = v4;
          v77 = sub_C8CA60(v75, v69);
          v4 = v124;
          if ( !v77 )
            goto LABEL_109;
LABEL_43:
          v43 += 8;
          if ( v44 == v43 )
            goto LABEL_44;
        }
        v76 = *(_QWORD **)(v75 + 8);
        v22 = (__int64)&v76[*(unsigned int *)(v75 + 20)];
        if ( v76 != (_QWORD *)v22 )
        {
          while ( v69 != *v76 )
          {
            if ( (_QWORD *)v22 == ++v76 )
              goto LABEL_109;
          }
          goto LABEL_43;
        }
LABEL_109:
        v78 = ((unsigned __int64)*(unsigned int *)(v47 + 72) << 32) | v4;
        v79 = (unsigned int)v149;
        v80 = (unsigned int)v149 + 1LL;
        if ( v80 > HIDWORD(v149) )
        {
          v127 = v78;
          sub_C8D5F0((__int64)&v148, v150, v80, 0x10u, v78, v5);
          v79 = (unsigned int)v149;
          v78 = v127;
        }
        v81 = (__int64 *)&v148[16 * v79];
        v43 += 8;
        *v81 = v47;
        v82 = (__int64)v148;
        v81[1] = v78;
        LODWORD(v149) = v149 + 1;
        v83 = 16LL * (unsigned int)v149;
        a2 = (v83 >> 4) - 1;
        sub_D67B60(v82, a2, 0, *(_QWORD *)(v82 + v83 - 16), *(_QWORD *)(v82 + v83 - 8));
      }
      while ( v44 != v43 );
LABEL_44:
      v43 = v130;
LABEL_45:
      if ( v43 != v132 )
        _libc_free(v43, a2);
      v49 = (__int64 *)v118[3];
      for ( k = &v49[*((unsigned int *)v118 + 8)]; k != v49; LODWORD(v146) = v146 + 1 )
      {
LABEL_48:
        v51 = *v49;
        if ( !v143 )
          goto LABEL_81;
        v52 = v140;
        v23 = HIDWORD(v141);
        v22 = (__int64)&v140[HIDWORD(v141)];
        if ( v140 != (__int64 *)v22 )
        {
          while ( v51 != *v52 )
          {
            if ( (__int64 *)v22 == ++v52 )
              goto LABEL_86;
          }
LABEL_53:
          if ( k == ++v49 )
            break;
          goto LABEL_48;
        }
LABEL_86:
        if ( HIDWORD(v141) < (unsigned int)v141 )
        {
          ++HIDWORD(v141);
          *(_QWORD *)v22 = v51;
          ++v139;
        }
        else
        {
LABEL_81:
          a2 = *v49;
          sub_C8CC70((__int64)&v139, *v49, v22, v23, v4, v5);
          if ( !(_BYTE)v22 )
            goto LABEL_53;
        }
        v67 = (unsigned int)v146;
        v23 = HIDWORD(v146);
        v68 = (unsigned int)v146 + 1LL;
        if ( v68 > HIDWORD(v146) )
        {
          a2 = (__int64)v147;
          sub_C8D5F0((__int64)&v145, v147, v68, 8u, v4, v5);
          v67 = (unsigned int)v146;
        }
        v22 = (__int64)v145;
        ++v49;
        *(_QWORD *)&v145[8 * v67] = v51;
      }
      v21 = v146;
    }
    while ( (_DWORD)v146 );
LABEL_55:
    result = (unsigned int)v149;
  }
  while ( (_DWORD)v149 );
LABEL_56:
  if ( !v143 )
    result = _libc_free(v140, a2);
  if ( !v137 )
    result = _libc_free(v134, a2);
  if ( v145 != v147 )
    result = _libc_free(v145, a2);
  if ( v148 != v150 )
    return _libc_free(v148, a2);
  return result;
}
