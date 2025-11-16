// Function: sub_2659FF0
// Address: 0x2659ff0
//
void __fastcall sub_2659FF0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rdi
  __int64 v5; // rdx
  _QWORD *v6; // r12
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rcx
  int v12; // edx
  unsigned int v13; // eax
  __int64 v14; // rdi
  char *v15; // r9
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // rbx
  char *v20; // r15
  __int64 v21; // r12
  char *v22; // rax
  char *v23; // r13
  __int64 v24; // rdx
  char *v25; // r9
  char *v26; // r15
  __int64 v27; // rdx
  char *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  char *v31; // rdx
  __int64 v32; // rdi
  __int64 v33; // rsi
  volatile signed __int32 *v34; // rdi
  char *v35; // rbx
  volatile signed __int32 *v36; // rdi
  __int64 v37; // rdx
  int v38; // r8d
  unsigned __int8 v39; // al
  __int64 v40; // rbx
  char v41; // al
  volatile signed __int32 *v42; // rax
  _QWORD *v43; // rdx
  char v44; // al
  char v45; // cl
  __int64 v46; // r15
  char v47; // dl
  unsigned __int8 v48; // di
  __int64 v49; // rbx
  __int64 **v50; // r14
  __int64 **v51; // rbx
  unsigned int v52; // esi
  __int64 v53; // r8
  int v54; // r11d
  __int64 v55; // rdi
  unsigned int v56; // edx
  __int64 *v57; // rax
  __int64 v58; // r9
  __int64 *v59; // r12
  char v60; // r13
  unsigned int v61; // edx
  int v62; // eax
  __int64 v63; // r8
  __int64 v64; // rax
  __int64 v65; // r12
  int v66; // r10d
  __int64 v67; // r13
  __int64 v68; // r11
  unsigned __int64 v69; // r9
  __int64 v70; // rcx
  int v71; // ebx
  unsigned int i; // edi
  char v73; // dl
  __int64 v74; // rsi
  unsigned int v75; // edx
  __int64 v76; // rax
  __int64 v77; // r8
  char v78; // al
  __int64 v79; // r10
  int v80; // r11d
  unsigned int v81; // edx
  __int64 v82; // r8
  int v83; // eax
  __int64 v84; // r8
  __int64 v85; // rsi
  int v86; // edx
  __int64 v87; // r8
  int v88; // r9d
  unsigned int v89; // edx
  __int64 v90; // rdi
  _DWORD *v91; // r15
  _DWORD *v92; // r13
  __int64 v93; // rcx
  char *v94; // rax
  char v95; // di
  char v96; // si
  int v97; // r11d
  unsigned int v98; // ebx
  volatile signed __int32 *v99; // rax
  __int64 v100; // r13
  __int64 v101; // rbx
  char v102; // al
  __int64 *v103; // rbx
  __int64 *v104; // r14
  volatile signed __int32 *v105; // r15
  __int64 v106; // r13
  volatile signed __int32 **v107; // rax
  __int64 v108; // rdx
  volatile signed __int32 **v109; // rdx
  int v110; // ecx
  signed __int64 v111; // rdx
  volatile signed __int32 **v112; // [rsp+10h] [rbp-150h]
  __int64 v113; // [rsp+18h] [rbp-148h]
  int v114; // [rsp+18h] [rbp-148h]
  __int64 v115; // [rsp+28h] [rbp-138h]
  __int64 *v116; // [rsp+28h] [rbp-138h]
  unsigned __int8 v119; // [rsp+48h] [rbp-118h]
  __int64 v120; // [rsp+50h] [rbp-110h]
  __int64 v121; // [rsp+50h] [rbp-110h]
  _QWORD *v122; // [rsp+50h] [rbp-110h]
  __int64 *v123; // [rsp+50h] [rbp-110h]
  __int64 v125; // [rsp+60h] [rbp-100h]
  char *v126; // [rsp+60h] [rbp-100h]
  volatile signed __int32 **v127; // [rsp+60h] [rbp-100h]
  __int64 v128; // [rsp+60h] [rbp-100h]
  __int64 v129; // [rsp+60h] [rbp-100h]
  _QWORD *v130; // [rsp+68h] [rbp-F8h] BYREF
  _QWORD v131[2]; // [rsp+70h] [rbp-F0h] BYREF
  volatile signed __int32 **v132; // [rsp+80h] [rbp-E0h] BYREF
  volatile signed __int32 **v133; // [rsp+88h] [rbp-D8h]
  _QWORD v134[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+B0h] [rbp-B0h]
  __int64 v136; // [rsp+B8h] [rbp-A8h]
  char *v137; // [rsp+C0h] [rbp-A0h] BYREF
  char *v138; // [rsp+C8h] [rbp-98h]
  __int64 v139; // [rsp+D0h] [rbp-90h]
  __int64 v140; // [rsp+D8h] [rbp-88h]
  _QWORD v141[2]; // [rsp+E0h] [rbp-80h] BYREF
  _DWORD *v142; // [rsp+F0h] [rbp-70h]
  _DWORD *v143; // [rsp+F8h] [rbp-68h]
  _QWORD *v144; // [rsp+100h] [rbp-60h] BYREF
  __int64 v145; // [rsp+108h] [rbp-58h]
  __int64 v146; // [rsp+110h] [rbp-50h]
  __int64 v147; // [rsp+118h] [rbp-48h]
  char v148; // [rsp+120h] [rbp-40h]

  v4 = (_QWORD *)a2;
  v130 = (_QWORD *)a2;
  if ( (_BYTE)qword_4FF39C8 && *(_BYTE *)(a2 + 2) )
  {
    sub_264C780((_QWORD *)a2);
    v4 = v130;
  }
  if ( v4[1] && !sub_10394B0(*((_BYTE *)v4 + 2)) )
  {
    sub_26463C0((__int64)&v144, a3, (__int64 *)&v130);
    sub_2640E50(&v144, v130 + 9, v5);
    v6 = (_QWORD *)v145;
    v7 = v144;
    if ( v144 != (_QWORD *)v145 )
    {
      while ( 1 )
      {
        v8 = (_QWORD *)*v7;
        if ( (*(_QWORD *)*v7 || v8[1]) && !*((_BYTE *)v8 + 17) )
        {
          v9 = v8[1];
          v10 = *(_DWORD *)(a3 + 24);
          v11 = *(_QWORD *)(a3 + 8);
          if ( !v10 )
            goto LABEL_46;
          v12 = v10 - 1;
          v13 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v14 = *(_QWORD *)(v11 + 8LL * v13);
          if ( v9 != v14 )
            break;
        }
LABEL_12:
        v7 += 2;
        if ( v6 == v7 )
          goto LABEL_13;
      }
      v38 = 1;
      while ( v14 != -4096 )
      {
        v13 = v12 & (v38 + v13);
        v14 = *(_QWORD *)(v11 + 8LL * v13);
        if ( v9 == v14 )
          goto LABEL_12;
        ++v38;
      }
LABEL_46:
      if ( !*(_QWORD *)(v9 + 120) )
        sub_2659FF0(a1, v9, a3, a4);
      goto LABEL_12;
    }
LABEL_13:
    sub_2644030((unsigned __int64 *)&v144);
    if ( sub_10394B0(*((_BYTE *)v130 + 2)) )
      return;
    v15 = (char *)v130[10];
    v16 = v130[9];
    v17 = (__int64)&v15[-v16];
    if ( (unsigned __int64)&v15[-v16] <= 0x10 )
      return;
    v18 = v131;
    v131[0] = 0x400000003LL;
    v19 = v17 >> 4;
    v131[1] = 0x200000001LL;
    if ( v17 <= 0 )
    {
LABEL_136:
      v23 = 0;
      sub_26496D0((char *)v16, v15, (__int64)v18);
    }
    else
    {
      v20 = v15;
      while ( 1 )
      {
        v125 = (__int64)v18;
        v21 = 16 * v19;
        v22 = (char *)sub_2207800(16 * v19);
        v18 = (_QWORD *)v125;
        v23 = v22;
        if ( v22 )
          break;
        v19 >>= 1;
        if ( !v19 )
        {
          v15 = v20;
          goto LABEL_136;
        }
      }
      v24 = *(_QWORD *)v16;
      *((_QWORD *)v22 + 1) = 0;
      v25 = v20;
      v26 = &v22[v21];
      *(_QWORD *)v16 = 0;
      *(_QWORD *)v22 = v24;
      v27 = *(_QWORD *)(v16 + 8);
      *(_QWORD *)(v16 + 8) = 0;
      *((_QWORD *)v22 + 1) = v27;
      v28 = v22 + 16;
      if ( &v22[v21] == v22 + 16 )
      {
        v31 = v22;
      }
      else
      {
        do
        {
          v29 = *((_QWORD *)v28 - 2);
          *((_QWORD *)v28 - 2) = 0;
          v28 += 16;
          *((_QWORD *)v28 - 2) = v29;
          v30 = *((_QWORD *)v28 - 3);
          *((_QWORD *)v28 - 3) = 0;
          *((_QWORD *)v28 - 1) = v30;
        }
        while ( v26 != v28 );
        v31 = &v23[v21 - 16];
      }
      v32 = *(_QWORD *)v31;
      v33 = *((_QWORD *)v31 + 1);
      *(_QWORD *)v31 = 0;
      *((_QWORD *)v31 + 1) = 0;
      *(_QWORD *)v16 = v32;
      v34 = *(volatile signed __int32 **)(v16 + 8);
      *(_QWORD *)(v16 + 8) = v33;
      if ( v34 )
      {
        v120 = v125;
        v126 = v25;
        sub_A191D0(v34);
        sub_26499B0((char *)v16, v126, v23, v19, v120);
      }
      else
      {
        sub_26499B0((char *)v16, v25, v23, v19, v125);
      }
      v35 = v23;
      do
      {
        v36 = (volatile signed __int32 *)*((_QWORD *)v35 + 1);
        if ( v36 )
          sub_A191D0(v36);
        v35 += 16;
      }
      while ( v26 != v35 );
    }
    j_j___libc_free_0((unsigned __int64)v23);
    v134[0] = 0;
    v134[1] = 0;
    v135 = 0;
    v136 = 0;
    if ( (_BYTE)qword_4FF3708 && !byte_4FF3548 )
    {
      v137 = 0;
      v138 = 0;
      v139 = 0;
      v140 = 0;
      v116 = (__int64 *)v130[10];
      if ( (__int64 *)v130[9] != v116 )
      {
        v123 = (__int64 *)v130[9];
        do
        {
          sub_264D1D0((__int64)&v137, *(_DWORD *)(*v123 + 40));
          v128 = *v123;
          sub_22B0690(v141, (__int64 *)(*v123 + 24));
          v129 = *(_QWORD *)(v128 + 32) + 4LL * *(unsigned int *)(v128 + 48);
          if ( v142 != (_DWORD *)v129 )
          {
            v91 = v142;
            v92 = v143;
            do
            {
              LODWORD(v132) = *v91;
              sub_22B6470((__int64)&v144, (__int64)&v137, (int *)&v132);
              if ( !v148 )
                sub_22B6470((__int64)&v144, (__int64)v134, (int *)&v132);
              do
                ++v91;
              while ( v91 != v92 && *v91 > 0xFFFFFFFD );
            }
            while ( (_DWORD *)v129 != v91 );
          }
          v123 += 2;
        }
        while ( v116 != v123 );
      }
      sub_2342640((__int64)&v137);
    }
    sub_2640E50(&v132, v130 + 9, v37);
    v112 = v133;
    v127 = v132;
    if ( v132 == v133 )
    {
LABEL_50:
      if ( (_BYTE)qword_4FF39C8 )
      {
        if ( *((_BYTE *)v130 + 2) )
          sub_264C780(v130);
      }
      sub_2644030((unsigned __int64 *)&v132);
      sub_2342640((__int64)v134);
      return;
    }
    while ( 1 )
    {
      if ( *(_QWORD *)*v127 || *((_QWORD *)*v127 + 1) )
      {
        if ( sub_10394B0(*((_BYTE *)v130 + 2)) || v130[10] - v130[9] <= 0x10u )
          goto LABEL_50;
        if ( *(_QWORD *)(*((_QWORD *)*v127 + 1) + 8LL) )
          break;
      }
LABEL_38:
      v127 += 2;
      if ( v112 == v127 )
        goto LABEL_50;
    }
    sub_264FDE0((__int64)v141, (__int64)(*v127 + 6), a4);
    if ( (_DWORD)v135 )
    {
      sub_264C5E0((__int64)&v144, (__int64)v141, (__int64)v134);
      sub_2641870((__int64)v141, (__int64)&v144);
      sub_2342640((__int64)&v144);
    }
    if ( !(_DWORD)v142 )
    {
LABEL_37:
      sub_2342640((__int64)v141);
      goto LABEL_38;
    }
    v39 = sub_26484B0((__int64)a1, (__int64)v141);
    v137 = 0;
    v119 = v39;
    v138 = 0;
    v139 = 0;
    sub_ED8CE0((__int64)&v137, (__int64)(v130[7] - v130[6]) >> 4);
    if ( v130[7] != v130[6] )
    {
      v121 = v130[7];
      v40 = v130[6];
      do
      {
        if ( *(_DWORD *)(*(_QWORD *)v40 + 40LL) < (unsigned int)v142 )
          v41 = sub_2642AE0((__int64)a1, *(_QWORD *)v40 + 24LL, (__int64)v141);
        else
          v41 = sub_2642AE0((__int64)a1, (__int64)v141, *(_QWORD *)v40 + 24LL);
        LOBYTE(v144) = v41;
        v40 += 16;
        sub_2659FC0((__int64)&v137, (char *)&v144);
      }
      while ( v121 != v40 );
    }
    v42 = *v127;
    if ( !*((_BYTE *)*v127 + 17) )
    {
      v43 = v130;
      v44 = *((_BYTE *)v130 + 2);
      if ( v119 == 3 )
      {
        v45 = 1;
        if ( v44 == 3 )
        {
LABEL_150:
          v93 = v130[6];
          v94 = v137;
          if ( v138 - v137 != (v130[7] - v93) >> 4 )
            goto LABEL_65;
          if ( v138 == v137 )
            goto LABEL_173;
          while ( 1 )
          {
            v96 = *v94;
            if ( *v94 )
            {
              v95 = *(_BYTE *)(*(_QWORD *)v93 + 16LL);
              if ( v95 )
              {
                if ( v96 != 3 )
                {
                  if ( v95 == 3 )
                    v95 = 1;
LABEL_155:
                  if ( v95 != v96 )
                    goto LABEL_65;
                  goto LABEL_156;
                }
                v96 = 1;
                if ( v95 != 3 )
                  goto LABEL_155;
              }
            }
LABEL_156:
            ++v94;
            v93 += 16;
            if ( v138 == v94 )
              goto LABEL_173;
          }
        }
      }
      else
      {
        v45 = v119;
        if ( v44 == 3 )
        {
          v45 = v119;
          v44 = 1;
        }
      }
      if ( v44 != v45 )
        goto LABEL_65;
      goto LABEL_150;
    }
    v85 = *((_QWORD *)v42 + 1);
    if ( *(_QWORD *)(v85 + 120) )
      goto LABEL_131;
    v86 = *(_DWORD *)(a3 + 24);
    v87 = *(_QWORD *)(a3 + 8);
    if ( v86 )
    {
      v88 = v86 - 1;
      v89 = (v86 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
      v90 = *(_QWORD *)(v87 + 8LL * v89);
      if ( v85 == v90 )
        goto LABEL_131;
      v110 = 1;
      while ( v90 != -4096 )
      {
        v89 = v88 & (v110 + v89);
        v90 = *(_QWORD *)(v87 + 8LL * v89);
        if ( v85 == v90 )
          goto LABEL_131;
        ++v110;
      }
    }
    v98 = *((_DWORD *)v42 + 10);
    sub_2659FF0(a1, v85, a3, v141);
    sub_264F9A0(*((_QWORD *)*v127 + 1));
    v99 = *v127;
    if ( *((_DWORD *)*v127 + 10) >= v98 || (v103 = (__int64 *)v130[9], (__int64 *)v130[10] == v103) )
    {
LABEL_171:
      if ( !*(_QWORD *)v99 && !*((_QWORD *)v99 + 1) )
        goto LABEL_173;
      sub_264FDE0((__int64)&v144, (__int64)v141, (__int64)(v99 + 6));
      sub_2641870((__int64)v141, (__int64)&v144);
      sub_2342640((__int64)&v144);
      if ( !(_DWORD)v142 )
        goto LABEL_173;
      goto LABEL_175;
    }
    v104 = (__int64 *)v130[10];
    while ( 1 )
    {
      v105 = (volatile signed __int32 *)v103[1];
      v106 = *v103;
      if ( !v105 )
        break;
      if ( &_pthread_key_create )
        _InterlockedAdd(v105 + 2, 1u);
      else
        ++*((_DWORD *)v105 + 2);
      if ( *(_QWORD *)(*(_QWORD *)(v106 + 8) + 120LL) == *((_QWORD *)*v127 + 1) )
      {
        sub_264FDE0((__int64)&v144, (__int64)v141, v106 + 24);
        if ( (_DWORD)v146 )
          goto LABEL_197;
        sub_2342640((__int64)&v144);
      }
LABEL_188:
      sub_A191D0(v105);
LABEL_189:
      v103 += 2;
      if ( v104 == v103 )
      {
        v99 = *v127;
        goto LABEL_171;
      }
    }
    if ( *((_QWORD *)*v127 + 1) != *(_QWORD *)(*(_QWORD *)(v106 + 8) + 120LL) )
      goto LABEL_189;
    sub_264FDE0((__int64)&v144, (__int64)v141, v106 + 24);
    if ( !(_DWORD)v146 )
    {
      sub_2342640((__int64)&v144);
      goto LABEL_189;
    }
LABEL_197:
    v107 = v132;
    v108 = ((char *)v133 - (char *)v132) >> 6;
    if ( v108 > 0 )
    {
      v109 = &v132[8 * v108];
      while ( (volatile signed __int32 *)v106 != *v107 )
      {
        if ( (volatile signed __int32 *)v106 == v107[2] )
        {
          v107 += 2;
          break;
        }
        if ( (volatile signed __int32 *)v106 == v107[4] )
        {
          v107 += 4;
          break;
        }
        if ( (volatile signed __int32 *)v106 == v107[6] )
        {
          v107 += 6;
          break;
        }
        v107 += 8;
        if ( v109 == v107 )
          goto LABEL_217;
      }
LABEL_204:
      if ( v133 == v107 )
        goto LABEL_220;
      sub_2342640((__int64)&v144);
      if ( !v105 )
        goto LABEL_189;
      goto LABEL_188;
    }
LABEL_217:
    v111 = (char *)v133 - (char *)v107;
    if ( (char *)v133 - (char *)v107 != 32 )
    {
      if ( v111 != 48 )
      {
        if ( v111 != 16 )
          goto LABEL_220;
        goto LABEL_232;
      }
      if ( (volatile signed __int32 *)v106 == *v107 )
        goto LABEL_204;
      v107 += 2;
    }
    if ( (volatile signed __int32 *)v106 == *v107 )
      goto LABEL_204;
    v107 += 2;
LABEL_232:
    if ( (volatile signed __int32 *)v106 == *v107 )
      goto LABEL_204;
LABEL_220:
    sub_2649AA0((__int64)v141, (__int64)&v144);
    *v127 = (volatile signed __int32 *)v106;
    sub_2647420(v127 + 1, v105);
    sub_2342640((__int64)&v144);
    if ( v105 )
      sub_A191D0(v105);
    if ( !*(_QWORD *)*v127 && !*((_QWORD *)*v127 + 1) )
    {
LABEL_173:
      sub_2646800((unsigned __int64 *)&v137);
      goto LABEL_37;
    }
LABEL_175:
    v119 = sub_26484B0((__int64)a1, (__int64)v141);
    if ( v137 != v138 )
      v138 = v137;
    v43 = v130;
    v100 = v130[6];
    v101 = v130[7];
    if ( v101 != v100 )
    {
      do
      {
        if ( *(_DWORD *)(*(_QWORD *)v100 + 40LL) < (unsigned int)v142 )
          v102 = sub_2642AE0((__int64)a1, *(_QWORD *)v100 + 24LL, (__int64)v141);
        else
          v102 = sub_2642AE0((__int64)a1, (__int64)v141, *(_QWORD *)v100 + 24LL);
        v100 += 16;
        LOBYTE(v144) = v102;
        sub_2659FC0((__int64)&v137, (char *)&v144);
      }
      while ( v101 != v100 );
LABEL_131:
      v43 = v130;
    }
LABEL_65:
    v115 = v43[13];
    if ( v115 == v43[12] )
    {
LABEL_132:
      v144 = 0;
      v145 = 0;
      v146 = 0;
      v147 = 0;
      sub_264A680((__int64)&v144, (__int64)v141);
      sub_2650560(a1, (__int64 *)v127, (__int64)&v144);
      sub_2342640((__int64)&v144);
    }
    else
    {
      v46 = v43[12];
      while ( 1 )
      {
        v48 = *(_BYTE *)(*(_QWORD *)v46 + 2LL);
        v122 = *(_QWORD **)v46;
        if ( v48 != 3 )
          break;
        if ( v119 == 3 || v119 == 1 )
          goto LABEL_74;
LABEL_70:
        v46 += 8;
        if ( v115 == v46 )
          goto LABEL_132;
      }
      v47 = 1;
      if ( v119 != 3 )
        v47 = v119;
      if ( v47 != v48 )
        goto LABEL_70;
LABEL_74:
      if ( !sub_10394B0(v48) || !sub_10394B0(v119) )
      {
        v49 = v122[15];
        v144 = 0;
        v145 = 0;
        v146 = 0;
        LODWORD(v147) = 0;
        v50 = (__int64 **)v122[7];
        v113 = v49;
        v51 = (__int64 **)v122[6];
        if ( v51 != v50 )
        {
          v52 = 0;
          v53 = 0;
          while ( 1 )
          {
            v59 = *v51;
            v60 = *((_BYTE *)*v51 + 16);
            if ( !v52 )
              break;
            v54 = 1;
            v55 = 0;
            v56 = (v52 - 1) & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
            v57 = (__int64 *)(v53 + 16LL * v56);
            v58 = *v57;
            if ( *v59 == *v57 )
            {
LABEL_79:
              v51 += 2;
              *((_BYTE *)v57 + 8) = v60;
              if ( v50 == v51 )
                goto LABEL_88;
              goto LABEL_80;
            }
            while ( v58 != -4096 )
            {
              if ( !v55 && v58 == -8192 )
                v55 = (__int64)v57;
              v56 = (v52 - 1) & (v54 + v56);
              v57 = (__int64 *)(v53 + 16LL * v56);
              v58 = *v57;
              if ( *v59 == *v57 )
                goto LABEL_79;
              ++v54;
            }
            if ( !v55 )
              v55 = (__int64)v57;
            v144 = (_QWORD *)((char *)v144 + 1);
            v62 = v146 + 1;
            if ( 4 * ((int)v146 + 1) >= 3 * v52 )
              goto LABEL_83;
            if ( v52 - (v62 + HIDWORD(v146)) <= v52 >> 3 )
            {
              sub_26455D0((__int64)&v144, v52);
              if ( !(_DWORD)v147 )
              {
LABEL_234:
                LODWORD(v146) = v146 + 1;
                BUG();
              }
              v79 = 0;
              v80 = 1;
              v81 = (v147 - 1) & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
              v62 = v146 + 1;
              v55 = v145 + 16LL * v81;
              v82 = *(_QWORD *)v55;
              if ( *v59 != *(_QWORD *)v55 )
              {
                while ( v82 != -4096 )
                {
                  if ( !v79 && v82 == -8192 )
                    v79 = v55;
                  v81 = (v147 - 1) & (v80 + v81);
                  v55 = v145 + 16LL * v81;
                  v82 = *(_QWORD *)v55;
                  if ( *v59 == *(_QWORD *)v55 )
                    goto LABEL_85;
                  ++v80;
                }
                goto LABEL_120;
              }
            }
LABEL_85:
            LODWORD(v146) = v62;
            if ( *(_QWORD *)v55 != -4096 )
              --HIDWORD(v146);
            v64 = *v59;
            v51 += 2;
            *(_BYTE *)(v55 + 8) = 0;
            *(_QWORD *)v55 = v64;
            *(_BYTE *)(v55 + 8) = v60;
            if ( v50 == v51 )
            {
LABEL_88:
              v65 = v145;
              v66 = v147;
              v67 = 16LL * (unsigned int)v147;
              goto LABEL_89;
            }
LABEL_80:
            v53 = v145;
            v52 = v147;
          }
          v144 = (_QWORD *)((char *)v144 + 1);
LABEL_83:
          sub_26455D0((__int64)&v144, 2 * v52);
          if ( !(_DWORD)v147 )
            goto LABEL_234;
          v61 = (v147 - 1) & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
          v62 = v146 + 1;
          v55 = v145 + 16LL * v61;
          v63 = *(_QWORD *)v55;
          if ( *v59 != *(_QWORD *)v55 )
          {
            v97 = 1;
            v79 = 0;
            while ( v63 != -4096 )
            {
              if ( !v79 && v63 == -8192 )
                v79 = v55;
              v61 = (v147 - 1) & (v97 + v61);
              v55 = v145 + 16LL * v61;
              v63 = *(_QWORD *)v55;
              if ( *v59 == *(_QWORD *)v55 )
                goto LABEL_85;
              ++v97;
            }
LABEL_120:
            if ( v79 )
              v55 = v79;
            goto LABEL_85;
          }
          goto LABEL_85;
        }
        v67 = 0;
        v66 = 0;
        v65 = 0;
LABEL_89:
        v68 = *(_QWORD *)(v113 + 48);
        v69 = (*(_QWORD *)(v113 + 56) - v68) >> 4;
        if ( v69 )
        {
          v70 = 0;
          v71 = v66 - 1;
          for ( i = 0; i < v69; v70 = ++i )
          {
            while ( 1 )
            {
              v74 = **(_QWORD **)(v68 + 16 * v70);
              if ( !v66 )
                goto LABEL_94;
              v75 = v71 & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
              v76 = v65 + 16LL * v75;
              v77 = *(_QWORD *)v76;
              if ( *(_QWORD *)v76 != v74 )
              {
                v83 = 1;
                if ( v77 == -4096 )
                  goto LABEL_94;
                while ( 1 )
                {
                  v75 = v71 & (v83 + v75);
                  v114 = v83 + 1;
                  v76 = v65 + 16LL * v75;
                  v84 = *(_QWORD *)v76;
                  if ( v74 == *(_QWORD *)v76 )
                    break;
                  v83 = v114;
                  if ( v84 == -4096 )
                    goto LABEL_94;
                }
              }
              if ( v76 == v65 + v67 )
                goto LABEL_94;
              v73 = v137[v70];
              if ( !v73 )
                goto LABEL_94;
              v78 = *(_BYTE *)(v76 + 8);
              if ( !v78 )
                goto LABEL_94;
              if ( v78 == 3 )
                break;
              if ( v73 == 3 )
                v73 = 1;
LABEL_93:
              if ( v78 != v73 )
              {
                sub_C7D6A0(v65, v67, 8);
                goto LABEL_70;
              }
LABEL_94:
              v70 = ++i;
              if ( i >= v69 )
                goto LABEL_103;
            }
            v78 = 1;
            if ( v73 != 3 )
              goto LABEL_93;
          }
        }
LABEL_103:
        sub_C7D6A0(v65, v67, 8);
      }
      v144 = 0;
      v145 = 0;
      v146 = 0;
      v147 = 0;
      sub_264A680((__int64)&v144, (__int64)v141);
      sub_264FE30((__int64)a1, (__int64 *)v127, (__int64)v122, 0, (__int64)&v144);
      sub_2342640((__int64)&v144);
    }
    if ( v137 )
      j_j___libc_free_0((unsigned __int64)v137);
    goto LABEL_37;
  }
}
