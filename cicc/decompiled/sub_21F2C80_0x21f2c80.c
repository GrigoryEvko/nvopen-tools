// Function: sub_21F2C80
// Address: 0x21f2c80
//
__int64 **__fastcall sub_21F2C80(__int64 a1, __int64 a2, __int64 a3, __int64 **a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  char v11; // al
  _BYTE *v12; // r11
  __int64 v13; // r14
  __int64 *v14; // r12
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned int v19; // r8d
  _QWORD *v20; // r11
  _QWORD *v21; // r13
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r11
  int v26; // r8d
  __int64 *v27; // r10
  __int64 *v28; // rcx
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rdi
  unsigned __int64 *v33; // r12
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  char v39; // al
  __int64 v40; // rdx
  bool v41; // zf
  int v42; // eax
  __int64 v43; // r14
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rax
  int v47; // r9d
  int v48; // r14d
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // rbx
  int v52; // r8d
  __int64 v53; // rax
  __int64 v54; // r8
  int v55; // r9d
  __int64 v56; // rax
  __int64 v57; // rdx
  int v58; // eax
  const char *v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  int v63; // r14d
  __int64 v64; // r13
  __int64 v65; // rax
  int v66; // r8d
  __int64 v67; // rax
  __int64 v68; // r8
  int v69; // r9d
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // r14
  __int64 *v73; // r12
  __int64 v74; // rdi
  unsigned __int64 *v75; // r13
  __int64 v76; // rax
  unsigned __int64 v77; // rsi
  __int64 v78; // rsi
  __int64 v79; // rsi
  __int64 v80; // rdx
  unsigned __int8 *v81; // rsi
  _BYTE *v82; // rsi
  char v83; // al
  __int64 v84; // r13
  __int64 *v85; // r14
  __int64 *v86; // rbx
  __int64 *v87; // rax
  __int64 v88; // rdi
  unsigned __int64 *v89; // r12
  __int64 v90; // rax
  unsigned __int64 v91; // rcx
  __int64 v92; // rsi
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  __int64 v95; // rax
  __int64 v96; // r13
  unsigned int v97; // eax
  __int64 v98; // r8
  unsigned __int64 v99; // r9
  __int64 v100; // rax
  unsigned __int64 v101; // rax
  int v102; // eax
  int v103; // eax
  _QWORD *v104; // rax
  __int64 v105; // rax
  __int64 *v106; // rax
  __int64 **v107; // rdx
  unsigned __int64 v108; // rcx
  __int64 v109; // [rsp+0h] [rbp-D0h]
  int v110; // [rsp+8h] [rbp-C8h]
  __int64 v111; // [rsp+8h] [rbp-C8h]
  _QWORD *v112; // [rsp+10h] [rbp-C0h]
  __int64 v113; // [rsp+10h] [rbp-C0h]
  int v114; // [rsp+18h] [rbp-B8h]
  __int64 v115; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v116; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v117; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v118; // [rsp+18h] [rbp-B8h]
  _QWORD *v119; // [rsp+20h] [rbp-B0h]
  __int64 v120; // [rsp+20h] [rbp-B0h]
  unsigned int v121; // [rsp+20h] [rbp-B0h]
  unsigned int v122; // [rsp+28h] [rbp-A8h]
  __int64 v123; // [rsp+28h] [rbp-A8h]
  unsigned int v124; // [rsp+28h] [rbp-A8h]
  __int64 v125; // [rsp+28h] [rbp-A8h]
  __int64 v126; // [rsp+28h] [rbp-A8h]
  __int64 v127; // [rsp+28h] [rbp-A8h]
  __int64 v128; // [rsp+28h] [rbp-A8h]
  __int64 v129; // [rsp+30h] [rbp-A0h]
  int v130; // [rsp+30h] [rbp-A0h]
  __int64 v131; // [rsp+30h] [rbp-A0h]
  __int64 v132; // [rsp+38h] [rbp-98h]
  const void *v133; // [rsp+38h] [rbp-98h]
  const char *v135; // [rsp+48h] [rbp-88h]
  const void *v136; // [rsp+48h] [rbp-88h]
  __int64 v137; // [rsp+48h] [rbp-88h]
  __int64 *v138; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int8 *v139; // [rsp+58h] [rbp-78h] BYREF
  const char *v140; // [rsp+60h] [rbp-70h] BYREF
  const char *v141; // [rsp+68h] [rbp-68h]
  __int16 v142; // [rsp+70h] [rbp-60h]
  __int64 v143[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v144; // [rsp+90h] [rbp-40h]

  v8 = a1;
  v9 = *(unsigned __int8 *)(a3 + 8);
  v10 = 100990;
  v135 = (const char *)a5;
  v132 = a6;
  if ( _bittest64(&v10, v9) )
  {
    v11 = *(_BYTE *)(a5 + 16);
    if ( v11 )
    {
      if ( v11 == 1 )
      {
        v140 = ".ldgsplit";
        v142 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a5 + 17) == 1 )
        {
          v60 = *(const char **)a5;
        }
        else
        {
          v60 = (const char *)a5;
          v11 = 2;
        }
        v140 = v60;
        v141 = ".ldgsplit";
        LOBYTE(v142) = v11;
        HIBYTE(v142) = 3;
      }
    }
    else
    {
      v142 = 256;
    }
    v12 = *(_BYTE **)(a1 + 96);
    v13 = *(unsigned int *)(a1 + 56);
    v14 = *(__int64 **)(a1 + 48);
    if ( v12[16] <= 0x10u )
    {
      if ( !*(_DWORD *)(a1 + 56) )
      {
LABEL_120:
        v107 = *(__int64 ***)(a1 + 48);
        v108 = *(unsigned int *)(a1 + 56);
        BYTE4(v143[0]) = 0;
        v21 = (_QWORD *)sub_15A2E80(0, (__int64)v12, v107, v108, 1u, (__int64)v143, 0);
LABEL_28:
        v39 = v135[16];
        if ( v39 )
        {
          if ( v39 == 1 )
          {
            v143[0] = (__int64)".load";
            v144 = 259;
          }
          else
          {
            if ( v135[17] == 1 )
            {
              v40 = *(_QWORD *)v135;
            }
            else
            {
              v40 = (__int64)v135;
              v39 = 2;
            }
            v143[0] = v40;
            v143[1] = (__int64)".load";
            LOBYTE(v144) = v39;
            HIBYTE(v144) = 3;
          }
        }
        else
        {
          v144 = 256;
        }
        v72 = -(__int64)(unsigned int)(*(_DWORD *)(v8 + 104) | *(_DWORD *)(v8 + 108))
            & (unsigned int)(*(_DWORD *)(v8 + 104) | *(_DWORD *)(v8 + 108));
        v73 = sub_1648A60(64, 1u);
        if ( v73 )
          sub_15F9210((__int64)v73, *(_QWORD *)(*v21 + 24LL), (__int64)v21, 0, 0, 0);
        v74 = *(_QWORD *)(a2 + 8);
        if ( v74 )
        {
          v75 = *(unsigned __int64 **)(a2 + 16);
          sub_157E9D0(v74 + 40, (__int64)v73);
          v76 = v73[3];
          v77 = *v75;
          v73[4] = (__int64)v75;
          v77 &= 0xFFFFFFFFFFFFFFF8LL;
          v73[3] = v77 | v76 & 7;
          *(_QWORD *)(v77 + 8) = v73 + 3;
          *v75 = *v75 & 7 | (unsigned __int64)(v73 + 3);
        }
        sub_164B780((__int64)v73, v143);
        v78 = *(_QWORD *)a2;
        if ( *(_QWORD *)a2 )
        {
          v140 = *(const char **)a2;
          sub_1623A60((__int64)&v140, v78, 2);
          v79 = v73[6];
          v80 = (__int64)(v73 + 6);
          if ( v79 )
          {
            sub_161E7C0((__int64)(v73 + 6), v79);
            v80 = (__int64)(v73 + 6);
          }
          v81 = (unsigned __int8 *)v140;
          v73[6] = (__int64)v140;
          if ( v81 )
            sub_1623210((__int64)&v140, v81, v80);
        }
        sub_15F8F50((__int64)v73, v72);
        v138 = v73;
        v82 = *(_BYTE **)(v132 + 8);
        if ( v82 == *(_BYTE **)(v132 + 16) )
        {
          sub_14147F0(v132, v82, &v138);
          v73 = v138;
        }
        else
        {
          if ( v82 )
          {
            *(_QWORD *)v82 = v73;
            v82 = *(_BYTE **)(v132 + 8);
            v73 = v138;
          }
          *(_QWORD *)(v132 + 8) = v82 + 8;
        }
        v83 = v135[16];
        if ( v83 )
        {
          if ( v83 == 1 )
          {
            v140 = ".ldgsplitinsert";
            v142 = 259;
          }
          else
          {
            if ( v135[17] == 1 )
              v135 = *(const char **)v135;
            else
              v83 = 2;
            LOBYTE(v142) = v83;
            HIBYTE(v142) = 3;
            v140 = v135;
            v141 = ".ldgsplitinsert";
          }
        }
        else
        {
          v142 = 256;
        }
        v84 = *(unsigned int *)(v8 + 24);
        v85 = *a4;
        if ( *((_BYTE *)*a4 + 16) > 0x10u || *((_BYTE *)v73 + 16) > 0x10u )
        {
          v136 = *(const void **)(v8 + 16);
          v144 = 257;
          v87 = sub_1648A60(88, 2u);
          v86 = v87;
          if ( v87 )
          {
            v133 = v136;
            v137 = (__int64)v87;
            sub_15F1EA0((__int64)v87, *v85, 63, (__int64)(v87 - 6), 2, 0);
            v86[7] = (__int64)(v86 + 9);
            v86[8] = 0x400000000LL;
            sub_15FAD90((__int64)v86, (__int64)v85, (__int64)v73, v133, v84, (__int64)v143);
          }
          else
          {
            v137 = 0;
          }
          v88 = *(_QWORD *)(a2 + 8);
          if ( v88 )
          {
            v89 = *(unsigned __int64 **)(a2 + 16);
            sub_157E9D0(v88 + 40, (__int64)v86);
            v90 = v86[3];
            v91 = *v89;
            v86[4] = (__int64)v89;
            v91 &= 0xFFFFFFFFFFFFFFF8LL;
            v86[3] = v91 | v90 & 7;
            *(_QWORD *)(v91 + 8) = v86 + 3;
            *v89 = *v89 & 7 | (unsigned __int64)(v86 + 3);
          }
          sub_164B780(v137, (__int64 *)&v140);
          v92 = *(_QWORD *)a2;
          if ( *(_QWORD *)a2 )
          {
            v139 = *(unsigned __int8 **)a2;
            sub_1623A60((__int64)&v139, v92, 2);
            v93 = v86[6];
            if ( v93 )
              sub_161E7C0((__int64)(v86 + 6), v93);
            v94 = v139;
            v86[6] = (__int64)v139;
            if ( v94 )
              sub_1623210((__int64)&v139, v94, (__int64)(v86 + 6));
          }
        }
        else
        {
          v86 = (__int64 *)sub_15A3A20(*a4, v73, *(_DWORD **)(v8 + 16), *(unsigned int *)(v8 + 24), 0);
        }
        *a4 = v86;
        return a4;
      }
      v15 = 0;
      while ( *(_BYTE *)(v14[v15] + 16) <= 0x10u )
      {
        if ( v13 == ++v15 )
          goto LABEL_120;
      }
    }
    v16 = *(_DWORD *)(a1 + 56) + 1;
    v144 = 257;
    v17 = *(_QWORD *)v12;
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
      v17 = **(_QWORD **)(v17 + 16);
    v119 = v12;
    v122 = v16;
    v129 = *(_QWORD *)(v17 + 24);
    v18 = sub_1648A60(72, v16);
    v19 = v122;
    v20 = v119;
    v21 = v18;
    if ( v18 )
    {
      v123 = (__int64)v18;
      v22 = *v119;
      v120 = (__int64)&v21[-3 * v19];
      if ( *(_BYTE *)(v22 + 8) == 16 )
        v22 = **(_QWORD **)(v22 + 16);
      v110 = v19;
      v112 = v20;
      v114 = *(_DWORD *)(v22 + 8) >> 8;
      v23 = (__int64 *)sub_15F9F50(v129, (__int64)v14, v13);
      v24 = (__int64 *)sub_1646BA0(v23, v114);
      v25 = (__int64)v112;
      v26 = v110;
      v27 = v24;
      if ( *(_BYTE *)(*v112 + 8LL) == 16 )
      {
        v106 = sub_16463B0(v24, *(_QWORD *)(*v112 + 32LL));
        v25 = (__int64)v112;
        v26 = v110;
        v27 = v106;
      }
      else
      {
        v28 = &v14[v13];
        if ( v14 != v28 )
        {
          v29 = v14;
          while ( 1 )
          {
            v30 = *(_QWORD *)*v29;
            if ( *(_BYTE *)(v30 + 8) == 16 )
              break;
            if ( v28 == ++v29 )
              goto LABEL_20;
          }
          v31 = sub_16463B0(v27, *(_QWORD *)(v30 + 32));
          v26 = v110;
          v25 = (__int64)v112;
          v27 = v31;
        }
      }
LABEL_20:
      v115 = v25;
      sub_15F1EA0((__int64)v21, (__int64)v27, 32, v120, v26, 0);
      v21[7] = v129;
      v21[8] = sub_15F9F50(v129, (__int64)v14, v13);
      sub_15F9CE0((__int64)v21, v115, v14, v13, (__int64)v143);
    }
    else
    {
      v123 = 0;
    }
    sub_15FA2E0((__int64)v21, 1);
    v32 = *(_QWORD *)(a2 + 8);
    if ( v32 )
    {
      v33 = *(unsigned __int64 **)(a2 + 16);
      sub_157E9D0(v32 + 40, (__int64)v21);
      v34 = v21[3];
      v35 = *v33;
      v21[4] = v33;
      v35 &= 0xFFFFFFFFFFFFFFF8LL;
      v21[3] = v35 | v34 & 7;
      *(_QWORD *)(v35 + 8) = v21 + 3;
      *v33 = *v33 & 7 | (unsigned __int64)(v21 + 3);
    }
    sub_164B780(v123, (__int64 *)&v140);
    v36 = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 )
    {
      v139 = *(unsigned __int8 **)a2;
      sub_1623A60((__int64)&v139, v36, 2);
      v37 = v21[6];
      if ( v37 )
        sub_161E7C0((__int64)(v21 + 6), v37);
      v38 = v139;
      v21[6] = v139;
      if ( v38 )
        sub_1623210((__int64)&v139, v38, (__int64)(v21 + 6));
    }
    goto LABEL_28;
  }
  v41 = (_BYTE)v9 == 14;
  v42 = *(_DWORD *)(a1 + 104);
  if ( v41 )
  {
    v121 = *(_DWORD *)(a1 + 104);
    v43 = 1;
    *(_DWORD *)(a1 + 104) = (*(_DWORD *)(a1 + 108) | v42) & -(*(_DWORD *)(a1 + 108) | v42);
    v44 = *(_QWORD *)a1;
    v45 = *(_QWORD *)(a3 + 24);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v45 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v95 = *(_QWORD *)(v45 + 32);
          v45 = *(_QWORD *)(v45 + 24);
          v43 *= v95;
          continue;
        case 1:
          v61 = 16;
          break;
        case 2:
          v61 = 32;
          break;
        case 3:
        case 9:
          v61 = 64;
          break;
        case 4:
          v61 = 80;
          break;
        case 5:
        case 6:
          v61 = 128;
          break;
        case 7:
          v61 = 8 * (unsigned int)sub_15A9520(v44, 0);
          break;
        case 0xB:
          v61 = *(_DWORD *)(v45 + 8) >> 8;
          break;
        case 0xD:
          v61 = 8LL * *(_QWORD *)sub_15A9930(v44, v45);
          break;
        case 0xE:
          v96 = *(_QWORD *)(v45 + 24);
          v131 = *(_QWORD *)(v45 + 32);
          v97 = sub_15A9FE0(v44, v96);
          v98 = 1;
          v99 = v97;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v96 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v105 = *(_QWORD *)(v96 + 32);
                v96 = *(_QWORD *)(v96 + 24);
                v98 *= v105;
                continue;
              case 1:
                v100 = 16;
                goto LABEL_106;
              case 2:
                v100 = 32;
                goto LABEL_106;
              case 3:
              case 9:
                v100 = 64;
                goto LABEL_106;
              case 4:
                v100 = 80;
                goto LABEL_106;
              case 5:
              case 6:
                v100 = 128;
                goto LABEL_106;
              case 7:
                v116 = v99;
                v126 = v98;
                v102 = sub_15A9520(v44, 0);
                v98 = v126;
                v99 = v116;
                v100 = (unsigned int)(8 * v102);
                goto LABEL_106;
              case 0xB:
                v100 = *(_DWORD *)(v96 + 8) >> 8;
                goto LABEL_106;
              case 0xD:
                v118 = v99;
                v128 = v98;
                v104 = (_QWORD *)sub_15A9930(v44, v96);
                v98 = v128;
                v99 = v118;
                v100 = 8LL * *v104;
                goto LABEL_106;
              case 0xE:
                v113 = *(_QWORD *)(v96 + 24);
                sub_15A9FE0(v44, v113);
                sub_127FA20(v44, v113);
                JUMPOUT(0x21F38D3);
              case 0xF:
                v117 = v99;
                v127 = v98;
                v103 = sub_15A9520(v44, *(_DWORD *)(v96 + 8) >> 8);
                v98 = v127;
                v99 = v117;
                v100 = (unsigned int)(8 * v103);
LABEL_106:
                v101 = (v99 + ((unsigned __int64)(v100 * v98 + 7) >> 3) - 1) / v99;
                a6 = v131 * v99;
                v61 = 8 * a6 * v101;
                break;
            }
            break;
          }
          break;
        case 0xF:
          v61 = 8 * (unsigned int)sub_15A9520(v44, *(_DWORD *)(v45 + 8) >> 8);
          break;
      }
      break;
    }
    v62 = *(_QWORD *)(a3 + 32);
    v130 = (unsigned __int64)(v43 * v61 + 7) >> 3;
    if ( (_DWORD)v62 )
    {
      v63 = 0;
      v64 = 0;
      v125 = (unsigned int)v62;
      v65 = *(unsigned int *)(v8 + 24);
      do
      {
        v66 = v64;
        if ( *(_DWORD *)(v8 + 28) <= (unsigned int)v65 )
        {
          sub_16CD150(v8 + 16, (const void *)(v8 + 32), 0, 4, v64, a6);
          v65 = *(unsigned int *)(v8 + 24);
          v66 = v64;
        }
        *(_DWORD *)(*(_QWORD *)(v8 + 16) + 4 * v65) = v66;
        ++*(_DWORD *)(v8 + 24);
        v67 = sub_1643350(*(_QWORD **)(a2 + 24));
        v68 = sub_159C470(v67, v64, 0);
        v70 = *(unsigned int *)(v8 + 56);
        if ( (unsigned int)v70 >= *(_DWORD *)(v8 + 60) )
        {
          v109 = v68;
          sub_16CD150(v8 + 48, (const void *)(v8 + 64), 0, 8, v68, v69);
          v70 = *(unsigned int *)(v8 + 56);
          v68 = v109;
        }
        ++v64;
        *(_QWORD *)(*(_QWORD *)(v8 + 48) + 8 * v70) = v68;
        ++*(_DWORD *)(v8 + 56);
        *(_DWORD *)(v8 + 108) = v63;
        sub_21F2C80(v8, a2, *(_QWORD *)(a3 + 24), a4, v135, v132);
        v71 = *(_DWORD *)(v8 + 24);
        --*(_DWORD *)(v8 + 56);
        v63 += v130;
        v65 = (unsigned int)(v71 - 1);
        *(_DWORD *)(v8 + 24) = v65;
      }
      while ( v125 != v64 );
    }
    *(_DWORD *)(v8 + 104) = v121;
    return (__int64 **)v121;
  }
  else
  {
    v124 = *(_DWORD *)(a1 + 104);
    *(_DWORD *)(a1 + 104) = (*(_DWORD *)(a1 + 108) | v42) & -(*(_DWORD *)(a1 + 108) | v42);
    v46 = sub_15A9930(*(_QWORD *)a1, a3);
    v48 = *(_DWORD *)(a3 + 12);
    v49 = v46;
    if ( v48 )
    {
      v50 = *(unsigned int *)(a1 + 24);
      v51 = 0;
      do
      {
        v52 = v51;
        if ( *(_DWORD *)(a1 + 28) <= (unsigned int)v50 )
        {
          sub_16CD150(a1 + 16, (const void *)(a1 + 32), 0, 4, v51, v47);
          v50 = *(unsigned int *)(a1 + 24);
          v52 = v51;
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v50) = v52;
        ++*(_DWORD *)(a1 + 24);
        v53 = sub_1643350(*(_QWORD **)(a2 + 24));
        v54 = sub_159C470(v53, v51, 0);
        v56 = *(unsigned int *)(a1 + 56);
        if ( (unsigned int)v56 >= *(_DWORD *)(a1 + 60) )
        {
          v111 = v54;
          sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 8, v54, v55);
          v56 = *(unsigned int *)(a1 + 56);
          v54 = v111;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v56) = v54;
        ++*(_DWORD *)(a1 + 56);
        *(_DWORD *)(a1 + 108) = *(_QWORD *)(v49 + 8 * v51 + 16);
        v57 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v51++);
        sub_21F2C80(a1, a2, v57, a4, v135, v132);
        v58 = *(_DWORD *)(a1 + 24);
        --*(_DWORD *)(a1 + 56);
        v50 = (unsigned int)(v58 - 1);
        *(_DWORD *)(a1 + 24) = v50;
      }
      while ( v48 != v51 );
      v8 = a1;
    }
    *(_DWORD *)(v8 + 104) = v124;
    return (__int64 **)v124;
  }
}
