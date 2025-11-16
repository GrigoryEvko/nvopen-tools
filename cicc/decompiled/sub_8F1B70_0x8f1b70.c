// Function: sub_8F1B70
// Address: 0x8f1b70
//
__int64 __fastcall sub_8F1B70(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rax
  int v9; // r10d
  int v10; // ecx
  int v11; // r10d
  int v12; // eax
  int v13; // esi
  __int64 v14; // rax
  char *v15; // rax
  char *v16; // r9
  int v17; // r15d
  int v18; // esi
  int v19; // r11d
  int v20; // edx
  unsigned int v21; // r11d
  unsigned int v22; // r11d
  int v23; // eax
  __int64 v24; // rax
  int v26; // esi
  int v27; // r9d
  char *v28; // r8
  int v29; // r13d
  int v30; // r12d
  __int64 v31; // r15
  char *v32; // r14
  int v33; // eax
  int v34; // ecx
  unsigned int v35; // r11d
  __int64 v36; // rax
  int v37; // r15d
  int v38; // eax
  int v39; // eax
  int v40; // r14d
  int v41; // eax
  __int64 v42; // rax
  int v43; // eax
  int v44; // edi
  char v45; // r9
  int v46; // r8d
  int v47; // eax
  int v48; // r8d
  __int64 v49; // rcx
  int v50; // edx
  bool v51; // sf
  int v52; // eax
  char *v53; // rdx
  int v54; // esi
  int v55; // eax
  int v56; // edx
  int v57; // r15d
  int v58; // eax
  int v59; // r9d
  int v60; // r8d
  int v61; // ecx
  int v62; // eax
  char *v63; // r14
  int v64; // esi
  int v65; // eax
  int v66; // r11d
  int v67; // eax
  int v68; // r10d
  int v69; // r15d
  int v70; // r14d
  int v71; // edx
  int v72; // eax
  int v73; // r8d
  char v74; // dl
  int v75; // esi
  int v76; // edx
  int v77; // r8d
  int v78; // eax
  int v79; // r9d
  int v80; // eax
  int v81; // eax
  int v82; // r14d
  char *v83; // rax
  int v84; // edx
  int v85; // edi
  int v86; // ecx
  int v87; // eax
  int v88; // r8d
  int v89; // r14d
  int v90; // ecx
  int v91; // eax
  int v92; // eax
  int v93; // eax
  int v94; // edi
  int v95; // r14d
  __int64 v96; // r9
  __int64 v97; // r8
  unsigned int *v98; // rsi
  unsigned int *v99; // r8
  int v100; // r10d
  unsigned int v101; // edx
  int v102; // eax
  int v103; // ecx
  int v104; // eax
  int v105; // eax
  char *v106; // rax
  int v107; // edx
  _BOOL4 v108; // eax
  int v109; // eax
  char v110; // r9
  int v111; // r15d
  int v112; // eax
  int v113; // eax
  int v114; // r14d
  int v115; // eax
  int v116; // edx
  char *s; // [rsp+10h] [rbp-60h]
  int v118; // [rsp+1Ch] [rbp-54h]
  __int64 *v119; // [rsp+20h] [rbp-50h]
  __int64 v120; // [rsp+28h] [rbp-48h]
  int v121; // [rsp+30h] [rbp-40h]
  int v122; // [rsp+30h] [rbp-40h]
  int v123; // [rsp+30h] [rbp-40h]
  int v124; // [rsp+34h] [rbp-3Ch]
  int v125; // [rsp+34h] [rbp-3Ch]
  char *v126; // [rsp+38h] [rbp-38h]
  __int64 v127; // [rsp+38h] [rbp-38h]
  int v128; // [rsp+38h] [rbp-38h]
  int v129; // [rsp+38h] [rbp-38h]
  int v130; // [rsp+38h] [rbp-38h]
  int v131; // [rsp+38h] [rbp-38h]
  int v132; // [rsp+38h] [rbp-38h]
  int v133; // [rsp+38h] [rbp-38h]

  v4 = a1;
  v5 = qword_4F690E0;
  v124 = a3;
  *(_DWORD *)(qword_4F690E0 + 2088) = 0;
  v6 = *(_QWORD *)v5;
  *(_DWORD *)(v6 + 2088) = 0;
  v7 = *(_QWORD *)v6;
  *(_DWORD *)(v7 + 2088) = 0;
  v8 = **(_QWORD **)v7;
  v119 = *(__int64 **)v7;
  *(_DWORD *)(*(_QWORD *)v7 + 2088LL) = 0;
  v9 = *(_DWORD *)(a2 + 48);
  qword_4F690E0 = v8;
  v10 = *(_DWORD *)(a2 + 8) - v9;
  v121 = v10;
  if ( v10 < 0 )
  {
    v121 = 0;
    v11 = v9 - *(_DWORD *)(a2 + 8);
  }
  else
  {
    if ( v10 )
      a3 += v10;
    v11 = 0;
  }
  v12 = *(_DWORD *)(a1 + 8) - *(_DWORD *)(a1 + 28);
  v13 = v11 + v12;
  if ( v12 < 0 )
  {
    a3 -= v12;
    v13 = v11;
  }
  if ( v13 > a3 )
  {
    v26 = v13 - a3;
    *(_DWORD *)(v7 + 8) = 1;
    *(_DWORD *)(v7 + 2088) = 1;
    v27 = v26;
    if ( v11 )
    {
      sub_8F0C00(v7, v11);
      v27 = v26;
    }
    sub_8F0920(v7, v27);
    v118 = 1;
  }
  else
  {
    *(_DWORD *)(v7 + 8) = 1;
    *(_DWORD *)(v7 + 2088) = 1;
    v118 = a3 - v13 + 1;
    if ( v11 )
      sub_8F0C00(v7, v11);
  }
  s = (char *)(a1 + 12);
  sub_8EEB70(v6, (unsigned __int8 *)(a1 + 12), *(_DWORD *)(a1 + 28));
  sub_8F0920(v6, 1);
  v120 = sub_8F0A50(v6, v7);
  v14 = qword_4F690E0;
  qword_4F690E0 = v6;
  *(_QWORD *)v6 = v14;
  v15 = *(char **)(a2 + 16);
  v16 = *(char **)(a2 + 24);
  v17 = *(_DWORD *)(a2 + 48);
  if ( v15 != v16 )
  {
    if ( !v17 )
      goto LABEL_19;
    v18 = 1;
    v19 = 0;
    while ( 1 )
    {
      v20 = *v15++;
      --v17;
      v18 *= 10;
      v19 = v20 + 10 * v19 - 48;
      if ( v15 == v16 )
        break;
      if ( !v17 )
        goto LABEL_18;
      if ( v18 > 429496728 )
      {
        v126 = v15;
        sub_8EEC10(v5, v18);
        sub_8EEC80(v5, v21);
        v16 = *(char **)(a2 + 24);
        v15 = v126;
        v18 = 1;
      }
    }
    v28 = *(char **)(a2 + 32);
    if ( *(char **)(a2 + 40) == v28 || !v17 )
      goto LABEL_18;
    goto LABEL_31;
  }
  v28 = *(char **)(a2 + 32);
  if ( *(char **)(a2 + 40) != v28 && v17 )
  {
    v19 = 0;
    v18 = 1;
LABEL_31:
    v127 = v7;
    v29 = (int)v28;
    v30 = v17;
    v31 = a2;
    v32 = v28;
    do
    {
      if ( v18 <= 429496728 )
      {
        v33 = 10 * v19;
        v18 *= 10;
      }
      else
      {
        sub_8EEC10(v5, v18);
        sub_8EEC80(v5, v35);
        v18 = 10;
        v33 = 0;
      }
      v34 = *v32++;
      v19 = v33 + v34 - 48;
    }
    while ( *(char **)(v31 + 40) != v32 && v30 + v29 != (_DWORD)v32 );
    v7 = v127;
    v4 = a1;
LABEL_18:
    sub_8EEC10(v5, v18);
    sub_8EEC80(v5, v22);
  }
LABEL_19:
  if ( v121 )
    sub_8F0C00(v5, v121);
  sub_8F0920(v5, v118);
  v23 = sub_8EECF0(v5, v120);
  if ( v23 >= 0 )
  {
    if ( !v23 )
    {
      *((_DWORD *)v119 + 522) = 0;
      if ( (int)sub_8EECF0((__int64)v119, v7) >= 0 || v124 )
        *(_DWORD *)(v4 + 8) -= v124;
      goto LABEL_25;
    }
    v42 = *(unsigned int *)(v5 + 2088);
    *((_DWORD *)v119 + 522) = v42;
    memcpy(v119 + 1, (const void *)(v5 + 8), 4 * v42);
    sub_8EED40((__int64)v119, v120);
    if ( (int)sub_8EECF0((__int64)v119, v7) < 0 && !v124 )
      goto LABEL_25;
    v43 = sub_8EEDF0((__int64)v119, v7);
    v44 = *(_DWORD *)(v4 + 28);
    v45 = v43;
    v46 = v44 + 14;
    v47 = v43 / 2;
    if ( v44 + 7 >= 0 )
      v46 = v44 + 7;
    v48 = v46 >> 3;
    if ( v44 > 8 )
    {
      if ( !v47 )
        goto LABEL_99;
      v49 = v4;
      while ( 1 )
      {
        v50 = *(unsigned __int8 *)(v49 + 12);
        v51 = v50 + v47 < 0;
        v52 = v50 + v47;
        *(_BYTE *)(v49 + 12) = v52;
        if ( v51 )
          v52 += 255;
        v47 = v52 >> 8;
        if ( v49 == v4 + (unsigned int)(v48 - 2) )
          break;
        ++v49;
        if ( !v47 )
          goto LABEL_99;
      }
    }
    if ( v47 )
    {
      v53 = &s[v48 - 1];
      v54 = (unsigned __int8)*v53;
      if ( (v44 & 7) != 0 )
      {
        v85 = -1 << (v44 % 8);
        v55 = (unsigned __int8)(~(_BYTE)v85 & v54) + v47;
        *v53 = v55 & ~(_BYTE)v85;
        v56 = ~(unsigned __int8)~(_BYTE)v85;
      }
      else
      {
        v55 = v54 + v47;
        *v53 = v55;
        v56 = -256;
      }
      if ( (v55 & v56) != 0 )
      {
        v57 = *(_DWORD *)(v4 + 28);
        if ( !v124 )
        {
          if ( v57 <= 0 || (*(_BYTE *)(v4 + 12) & 1) == 0 )
          {
            sub_8EE7D0(s, v57);
            v72 = *(_DWORD *)(v4 + 8) + 1;
            goto LABEL_98;
          }
          v58 = sub_8EE5A0((__int64)s, 0);
          if ( !(v58 | v59) && !*((_DWORD *)v119 + 522) )
          {
            v60 = 1;
            goto LABEL_74;
          }
          v60 = 1;
LABEL_153:
          v63 = s;
          v133 = v60;
          v108 = sub_8EE9E0((__int64)s, v57, v124);
          v60 = v133;
          if ( !v108 )
            goto LABEL_75;
          v64 = *(_DWORD *)(v4 + 28);
LABEL_79:
          while ( 1 )
          {
            sub_8EE7D0(v63, v64);
            v65 = *(_DWORD *)(v4 + 8) + 1;
            *(_DWORD *)(v4 + 8) = v65;
            if ( !v66 )
              break;
            v64 = *(_DWORD *)(v4 + 28);
          }
          goto LABEL_91;
        }
        if ( v124 > v57 )
        {
          if ( v57 != (unsigned int)sub_8EE4D0(s, v57) || (v110 & 1) != 0 )
          {
            v60 = 1;
            goto LABEL_106;
          }
          v60 = 1;
          goto LABEL_116;
        }
        if ( v124 > 0 )
        {
          v75 = v124 - 1;
          v116 = *(unsigned __int8 *)(v4 + ((v124 - 1) >> 3) + 12);
          if ( !_bittest(&v116, ((_BYTE)v124 - 1) & 7) )
          {
            v63 = s;
            v64 = *(_DWORD *)(v4 + 28);
            goto LABEL_79;
          }
          v77 = 1;
LABEL_104:
          v128 = v77;
          v78 = sub_8EE5A0((__int64)s, v75);
          v60 = v128;
          if ( v78 | v79 )
            goto LABEL_105;
          goto LABEL_116;
        }
        v60 = 1;
        goto LABEL_182;
      }
    }
LABEL_99:
    if ( !v124 )
    {
      if ( (v45 & 1) == 0 )
      {
        v72 = *(_DWORD *)(v4 + 8);
        goto LABEL_98;
      }
      v57 = *(_DWORD *)(v4 + 28);
      v60 = 0;
LABEL_116:
      if ( !*((_DWORD *)v119 + 522) )
      {
        if ( v124 >= v57 )
          goto LABEL_75;
LABEL_74:
        v61 = *(unsigned __int8 *)(v4 + v124 / 8 + 12);
        if ( !_bittest(&v61, v124 % 8) )
        {
LABEL_75:
          v62 = v60;
          goto LABEL_76;
        }
        goto LABEL_153;
      }
LABEL_105:
      if ( v124 < v57 )
        goto LABEL_153;
LABEL_106:
      v80 = v57 + 14;
      if ( v57 + 7 >= 0 )
        v80 = v57 + 7;
      v81 = v80 >> 3;
      v82 = v81 - 1;
      if ( v81 - 1 <= 0 )
      {
        v83 = s;
      }
      else
      {
        v129 = v60;
        memset(s, 0, (unsigned int)(v81 - 2) + 1LL);
        v60 = v129;
        v83 = &s[v82];
      }
      LOBYTE(v84) = 0x80;
      if ( (v57 & 7) != 0 )
        v84 = 1 << (v57 % 8 - 1);
      *v83 = v84;
      v62 = v60;
      --v124;
LABEL_76:
      if ( v62 )
      {
        v63 = s;
        v64 = *(_DWORD *)(v4 + 28);
        goto LABEL_79;
      }
LABEL_90:
      v65 = *(_DWORD *)(v4 + 8);
LABEL_91:
      v72 = v65 - v124;
LABEL_98:
      *(_DWORD *)(v4 + 8) = v72;
      goto LABEL_25;
    }
    v57 = *(_DWORD *)(v4 + 28);
    if ( v124 > 0 )
    {
      if ( v124 > v57 )
        goto LABEL_90;
      v75 = v124 - 1;
      v76 = *(unsigned __int8 *)(v4 + ((v124 - 1) >> 3) + 12);
      if ( !_bittest(&v76, ((_BYTE)v124 - 1) & 7) )
        goto LABEL_90;
      v77 = 0;
      goto LABEL_104;
    }
    v60 = 0;
LABEL_182:
    if ( (v45 & 1) != 0 )
      goto LABEL_105;
    goto LABEL_116;
  }
  v36 = *(unsigned int *)(v120 + 2088);
  *((_DWORD *)v119 + 522) = v36;
  memcpy(v119 + 1, (const void *)(v120 + 8), 4 * v36);
  sub_8EED40((__int64)v119, v5);
  if ( (int)sub_8EECF0((__int64)v119, v7) >= 0 || v124 )
  {
    v67 = sub_8EEDF0((__int64)v119, v7);
    v69 = sub_8EEAD0((unsigned __int8 *)s, *(_DWORD *)(v4 + 28), v67 / 2);
    if ( v69 )
    {
      v94 = *(_DWORD *)(v7 + 2088);
      v95 = v68 % 2 + 2 * v69;
      if ( v94 )
      {
        v96 = (unsigned int)(v94 - 1);
        v97 = v7 + 4LL * v94;
        v98 = (unsigned int *)(v97 + 4);
        v99 = (unsigned int *)(v97 - 4 * v96);
        v100 = 0;
        do
        {
          v101 = *v98--;
          v102 = v100 | (v101 >> 1);
          v100 = v101 << 31;
          v98[1] = v102;
        }
        while ( v99 != v98 );
        if ( !*(_DWORD *)(v7 + 4 * v96 + 8) )
          *(_DWORD *)(v7 + 2088) = v96;
      }
      if ( (int)sub_8EECF0((__int64)v119, v7) > 0 )
        sub_8EED40((__int64)v119, v7);
      else
        --v95;
      v103 = *(_DWORD *)(v4 + 28);
      v104 = v103 + 14;
      if ( v103 + 7 >= 0 )
        v104 = v103 + 7;
      v105 = v104 >> 3;
      v123 = v105 - 1;
      if ( v105 - 1 <= 0 )
      {
        v106 = s;
      }
      else
      {
        v132 = *(_DWORD *)(v4 + 28);
        memset(s, 255, (unsigned int)(v105 - 2) + 1LL);
        v103 = v132;
        v106 = &s[v123];
      }
      LOBYTE(v107) = -1;
      if ( (v103 & 7) != 0 )
        v107 = ~(-1 << (v103 % 8));
      *v106 = v107;
      sub_8EEAD0((unsigned __int8 *)s, *(_DWORD *)(v4 + 28), v95);
    }
    v70 = *(_DWORD *)(v4 + 28);
    if ( v124 )
    {
      if ( v124 > 0 )
      {
        if ( v124 > v70 )
          goto LABEL_89;
        v71 = *(unsigned __int8 *)(v4 + ((v124 - 1) >> 3) + 12);
        if ( !_bittest(&v71, ((_BYTE)v124 - 1) & 7) )
          goto LABEL_89;
        if ( (unsigned int)sub_8EE5A0((__int64)s, v124 - 1) )
        {
          if ( v124 == v70 )
          {
            if ( (unsigned int)(v70 + 7) >> 3 != 1 )
              memset(s, 255, ((unsigned int)(v70 + 7) >> 3) - 2 + 1LL);
            LOBYTE(v109) = -1;
            if ( (v70 & 7) != 0 )
              v109 = ~(-1 << (v70 % 8));
            v124 = v70;
            *(_BYTE *)(v4 + (int)(((unsigned int)(v70 + 7) >> 3) - 1) + 12) = v109;
            goto LABEL_160;
          }
          goto LABEL_88;
        }
      }
      if ( !(*((_DWORD *)v119 + 522) | v68 & 1) && v124 < v70 )
      {
        v86 = *(unsigned __int8 *)(v4 + v124 / 8 + 12);
        if ( _bittest(&v86, v124 % 8) )
        {
LABEL_88:
          if ( !sub_8EE9E0((__int64)s, v70, v124) )
            goto LABEL_89;
          sub_8EE7D0(s, *(_DWORD *)(v4 + 28));
          --v124;
          if ( !v69 )
          {
            v72 = *(_DWORD *)(v4 + 8) - v124;
            goto LABEL_98;
          }
          if ( v124 )
            goto LABEL_160;
          v72 = *(_DWORD *)(v4 + 8);
LABEL_97:
          --v72;
          goto LABEL_98;
        }
      }
LABEL_89:
      if ( !v69 )
        goto LABEL_90;
LABEL_160:
      v72 = *(_DWORD *)(v4 + 8) - (v124 - 1);
      goto LABEL_98;
    }
    if ( sub_8EE3E0((unsigned __int8 *)s, v70) )
    {
      v130 = v73;
      sub_8F0920((__int64)v119, 1);
      v87 = sub_8EECF0((__int64)v119, v7);
      v88 = v130;
      v89 = v87;
      if ( v87 > 0 || (_BYTE)v130 )
      {
        v90 = *(_DWORD *)(v4 + 28);
        v91 = v90 + 14;
        if ( v90 + 7 >= 0 )
          v91 = v90 + 7;
        v92 = v91 >> 3;
        v122 = v92 - 1;
        if ( v92 - 1 > 0 )
        {
          v125 = v130;
          v131 = *(_DWORD *)(v4 + 28);
          memset(s, 255, (unsigned int)(v92 - 2) + 1LL);
          s += v122;
          v88 = v125;
          v90 = v131;
        }
        LOBYTE(v93) = -1;
        if ( (v90 & 7) != 0 )
          v93 = ~(-1 << (v90 % 8));
        *s = v93;
        v72 = *(_DWORD *)(v4 + 8) - 1;
        *(_DWORD *)(v4 + 8) = v72;
        if ( v88 && v89 >= 0 )
        {
          --*(_BYTE *)(v4 + 12);
          if ( !v69 )
            goto LABEL_98;
          goto LABEL_97;
        }
        goto LABEL_136;
      }
    }
    else
    {
      if ( v70 > 0 )
      {
        v74 = *(_BYTE *)(v4 + 12);
        if ( (v74 & 1) != 0 )
        {
          v72 = *(_DWORD *)(v4 + 8);
          if ( v73 )
          {
            *(_BYTE *)(v4 + 12) = v74 - 1;
            if ( !v69 )
              goto LABEL_98;
            goto LABEL_97;
          }
LABEL_136:
          if ( !v69 )
            goto LABEL_98;
          goto LABEL_97;
        }
      }
      if ( v73 && *((_DWORD *)v119 + 522) && (unsigned int)sub_8EEAD0((unsigned __int8 *)s, v70, 1) )
      {
        v111 = *(_DWORD *)(v4 + 28);
        v112 = v111 + 14;
        if ( v111 + 7 >= 0 )
          v112 = v111 + 7;
        v113 = v112 >> 3;
        v114 = v113 - 1;
        if ( v113 - 1 <= 0 )
          v114 = 0;
        else
          memset(s, 255, (unsigned int)(v113 - 2) + 1LL);
        LOBYTE(v115) = -1;
        if ( (v111 & 7) != 0 )
          v115 = ~(-1 << (v111 % 8));
        *(_BYTE *)(v4 + v114 + 12) = v115;
        v72 = *(_DWORD *)(v4 + 8);
        goto LABEL_97;
      }
    }
    v72 = *(_DWORD *)(v4 + 8);
    goto LABEL_136;
  }
  if ( sub_8EE3E0((unsigned __int8 *)s, *(_DWORD *)(v4 + 28)) )
  {
    sub_8F0920((__int64)v119, 1);
    if ( (int)sub_8EECF0((__int64)v119, v7) > 0 )
    {
      v37 = *(_DWORD *)(v4 + 28);
      v38 = v37 + 14;
      if ( v37 + 7 >= 0 )
        v38 = v37 + 7;
      v39 = v38 >> 3;
      v40 = v39 - 1;
      if ( v39 - 1 > 0 )
      {
        memset(s, 255, (unsigned int)(v39 - 2) + 1LL);
        s += v40;
      }
      LOBYTE(v41) = -1;
      if ( (v37 & 7) != 0 )
        v41 = ~(-1 << (v37 % 8));
      *s = v41;
      --*(_DWORD *)(v4 + 8);
    }
  }
LABEL_25:
  v24 = qword_4F690E0;
  qword_4F690E0 = v5;
  *v119 = v24;
  *(_QWORD *)v7 = v119;
  *(_QWORD *)v120 = v7;
  *(_QWORD *)v5 = v120;
  return v120;
}
