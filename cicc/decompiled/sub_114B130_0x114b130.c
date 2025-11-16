// Function: sub_114B130
// Address: 0x114b130
//
__int64 __fastcall sub_114B130(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // r13
  char v5; // al
  __int64 v6; // r10
  __int64 *v7; // r8
  __int64 v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 *v14; // r11
  __int64 v15; // rax
  _QWORD *v16; // rdi
  _QWORD *v17; // rsi
  __int64 v18; // rcx
  __int64 *v19; // r10
  __int64 v21; // rsi
  __int64 *v22; // r14
  __int64 *v23; // rdi
  __int64 v24; // r15
  __int64 v25; // rax
  _QWORD *v26; // rdi
  _QWORD *v27; // rsi
  bool v28; // al
  _QWORD *v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rdi
  _QWORD *v32; // rsi
  bool v33; // al
  __int64 v34; // rax
  _QWORD *v35; // rdi
  _QWORD *v36; // rsi
  bool v37; // al
  __int64 v38; // rdi
  __int64 *v39; // r14
  __int64 *v40; // rsi
  __int64 v41; // r15
  __int64 v42; // rdi
  __int64 *v43; // r14
  __int64 *v44; // rsi
  __int64 v45; // r15
  __int64 v46; // rdi
  __int64 *v47; // r14
  __int64 *v48; // rsi
  __int64 v49; // r15
  _BYTE *v50; // rax
  _QWORD *v51; // rdi
  _QWORD *v52; // rsi
  bool v53; // al
  _QWORD *v54; // rdi
  _QWORD *v55; // rsi
  bool v56; // al
  int v57; // esi
  int v58; // esi
  int v59; // esi
  __int64 v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rdi
  _QWORD *v64; // rsi
  bool v65; // al
  int v66; // edi
  __int64 v67; // rdi
  _QWORD *v68; // r10
  _QWORD *v69; // rsi
  _BYTE *v70; // r11
  __int64 v71; // rax
  __int64 v72; // rax
  _QWORD *v73; // rdi
  _QWORD *v74; // rsi
  __int64 v75; // rax
  _QWORD *v76; // rdi
  _QWORD *v77; // rsi
  bool v78; // al
  __int64 v79; // rsi
  __int64 v80; // rdi
  unsigned int v81; // eax
  __int64 v82; // rdi
  __int64 *v83; // rsi
  __int64 v84; // rdi
  __int64 v85; // r11
  int v86; // r14d
  __int64 v87; // rsi
  int v88; // eax
  __int64 v89; // rax
  unsigned int v90; // r13d
  bool v91; // al
  __int64 v92; // rdi
  __int64 v93; // r11
  int v94; // r14d
  __int64 v95; // rsi
  int v96; // esi
  int v97; // r14d
  int v98; // ecx
  int v99; // r10d
  int v100; // esi
  int v101; // r11d
  int v102; // ecx
  int v103; // r10d
  __int64 v104; // rcx
  int v105; // ecx
  int v106; // r10d
  __int64 v107; // rcx
  int v108; // [rsp+4h] [rbp-5Ch]
  int v109; // [rsp+4h] [rbp-5Ch]
  int v110; // [rsp+4h] [rbp-5Ch]
  int v111; // [rsp+4h] [rbp-5Ch]
  __int64 *v112; // [rsp+10h] [rbp-50h]
  __int64 v113; // [rsp+18h] [rbp-48h]
  __int64 v114; // [rsp+20h] [rbp-40h] BYREF
  __int64 v115[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 + 16);
  if ( v2 )
  {
    v113 = a1 + 288;
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)(v2 + 24);
        v114 = v4;
        v5 = *(_BYTE *)v4;
        if ( *(_BYTE *)v4 == 61 )
        {
          if ( (*(_BYTE *)(v4 + 2) & 1) != 0 )
            return 0;
          v115[0] = v4;
          sub_114AC10(v113, v115);
          goto LABEL_5;
        }
        if ( v5 != 84 )
          break;
        v6 = 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
        {
          v7 = *(__int64 **)(v4 - 8);
          v8 = v6 >> 5;
          v9 = &v7[(unsigned __int64)v6 / 8];
          v10 = v6 >> 7;
          v112 = v9;
          if ( !v10 )
            goto LABEL_77;
        }
        else
        {
          v112 = (__int64 *)v4;
          v7 = (__int64 *)(v4 - v6);
          v8 = v6 >> 5;
          v10 = v6 >> 7;
          if ( !v10 )
          {
LABEL_77:
            v12 = v8;
            v11 = v7;
            if ( v8 != 2 )
              goto LABEL_111;
            goto LABEL_78;
          }
        }
        v11 = v7;
        do
        {
          v12 = *v11;
          if ( *(_BYTE *)*v11 <= 0x1Cu )
            goto LABEL_16;
          v12 = v11[4];
          if ( *(_BYTE *)v12 <= 0x1Cu )
          {
            v11 += 4;
            goto LABEL_16;
          }
          v12 = v11[8];
          if ( *(_BYTE *)v12 <= 0x1Cu )
          {
            v11 += 8;
            goto LABEL_16;
          }
          v12 = v11[12];
          if ( *(_BYTE *)v12 <= 0x1Cu )
          {
            v11 += 12;
LABEL_16:
            if ( v112 != v11 )
              return 0;
            goto LABEL_17;
          }
          v11 += 16;
        }
        while ( &v7[16 * v10] != v11 );
        v12 = ((char *)v112 - (char *)v11) >> 5;
        if ( v12 != 2 )
        {
LABEL_111:
          if ( v12 != 3 )
          {
            if ( v12 != 1 )
              goto LABEL_113;
LABEL_80:
            v12 = *v11;
            if ( *(_BYTE *)*v11 > 0x1Cu )
              goto LABEL_113;
            goto LABEL_81;
          }
          v12 = *v11;
          if ( *(_BYTE *)*v11 <= 0x1Cu )
            goto LABEL_81;
          v11 += 4;
        }
LABEL_78:
        v12 = *v11;
        if ( *(_BYTE *)*v11 > 0x1Cu )
        {
          v11 += 4;
          goto LABEL_80;
        }
LABEL_81:
        if ( v11 != v112 )
          return 0;
LABEL_113:
        if ( !v10 )
        {
          if ( v8 != 2 )
            goto LABEL_52;
LABEL_115:
          v13 = *(_QWORD *)(a1 + 424);
LABEL_116:
          v62 = *v7;
          v115[0] = v62;
          if ( v62 == v13 )
            goto LABEL_120;
          if ( !*(_DWORD *)(a1 + 304) )
          {
            v63 = *(_QWORD **)(a1 + 320);
            v64 = &v63[*(unsigned int *)(a1 + 328)];
            v65 = v64 != sub_1149D50(v63, (__int64)v64, v115);
LABEL_119:
            if ( !v65 )
              goto LABEL_21;
LABEL_120:
            v7 += 4;
            goto LABEL_136;
          }
          v12 = *(unsigned int *)(a1 + 312);
          v84 = *(_QWORD *)(a1 + 296);
          v85 = v84 + 8 * v12;
          if ( (_DWORD)v12 )
          {
            v86 = v12 - 1;
            v12 = ((_DWORD)v12 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
            v18 = v84 + 8 * v12;
            v87 = *(_QWORD *)v18;
            if ( *(_QWORD *)v18 == v62 )
            {
LABEL_156:
              v65 = v18 != v85;
              goto LABEL_119;
            }
            v102 = 1;
            while ( v87 != -4096 )
            {
              v103 = v102 + 1;
              v104 = v86 & (unsigned int)(v12 + v102);
              v12 = (unsigned int)v104;
              v18 = v84 + 8 * v104;
              v87 = *(_QWORD *)v18;
              if ( v62 == *(_QWORD *)v18 )
                goto LABEL_156;
              v102 = v103;
            }
          }
          v18 = v85;
          goto LABEL_156;
        }
LABEL_17:
        v13 = *(_QWORD *)(a1 + 424);
        v14 = v115;
        while ( 2 )
        {
          v15 = *v7;
          v115[0] = v15;
          if ( v15 != v13 )
          {
            if ( *(_DWORD *)(a1 + 304) )
            {
              v18 = *(unsigned int *)(a1 + 312);
              v21 = *(_QWORD *)(a1 + 296);
              v12 = v18;
              v22 = (__int64 *)(v21 + 8 * v18);
              if ( !(_DWORD)v18 )
                goto LABEL_21;
              v12 = (unsigned int)(v18 - 1);
              v18 = (unsigned int)v12 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
              v23 = (__int64 *)(v21 + 8 * v18);
              v24 = *v23;
              if ( v15 != *v23 )
              {
                v66 = 1;
                while ( v24 != -4096 )
                {
                  v18 = (unsigned int)v12 & (v66 + (_DWORD)v18);
                  v108 = v66 + 1;
                  v23 = (__int64 *)(v21 + 8LL * (unsigned int)v18);
                  v24 = *v23;
                  if ( v15 == *v23 )
                    goto LABEL_27;
                  v66 = v108;
                }
                goto LABEL_21;
              }
LABEL_27:
              if ( v22 == v23 )
                goto LABEL_21;
            }
            else
            {
              v16 = *(_QWORD **)(a1 + 320);
              v17 = &v16[*(unsigned int *)(a1 + 328)];
              if ( v17 == sub_1149D50(v16, (__int64)v17, v14) )
                goto LABEL_21;
            }
          }
          v25 = v7[4];
          v115[0] = v25;
          if ( v25 == v13 )
            goto LABEL_40;
          if ( *(_DWORD *)(a1 + 304) )
          {
            v18 = *(unsigned int *)(a1 + 312);
            v38 = *(_QWORD *)(a1 + 296);
            v12 = v18;
            v39 = (__int64 *)(v38 + 8 * v18);
            if ( (_DWORD)v18 )
            {
              v12 = (unsigned int)(v18 - 1);
              v18 = (unsigned int)v12 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v40 = (__int64 *)(v38 + 8 * v18);
              v41 = *v40;
              if ( v25 == *v40 )
              {
LABEL_57:
                v28 = v39 != v40;
                goto LABEL_31;
              }
              v57 = 1;
              while ( v41 != -4096 )
              {
                v18 = (unsigned int)v12 & (v57 + (_DWORD)v18);
                v109 = v57 + 1;
                v40 = (__int64 *)(v38 + 8LL * (unsigned int)v18);
                v41 = *v40;
                if ( v25 == *v40 )
                  goto LABEL_57;
                v57 = v109;
              }
            }
            v40 = v39;
            goto LABEL_57;
          }
          v26 = *(_QWORD **)(a1 + 320);
          v27 = &v26[*(unsigned int *)(a1 + 328)];
          v28 = v27 != sub_1149D50(v26, (__int64)v27, v14);
LABEL_31:
          if ( !v28 )
          {
            v7 += 4;
            v19 = v115;
            if ( v7 == v112 )
              goto LABEL_22;
            goto LABEL_33;
          }
LABEL_40:
          v30 = v7[8];
          v115[0] = v30;
          if ( v30 == v13 )
            goto LABEL_45;
          if ( *(_DWORD *)(a1 + 304) )
          {
            v18 = *(unsigned int *)(a1 + 312);
            v42 = *(_QWORD *)(a1 + 296);
            v12 = v18;
            v43 = (__int64 *)(v42 + 8 * v18);
            if ( (_DWORD)v18 )
            {
              v12 = (unsigned int)(v18 - 1);
              v18 = (unsigned int)v12 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v44 = (__int64 *)(v42 + 8 * v18);
              v45 = *v44;
              if ( *v44 == v30 )
              {
LABEL_60:
                v33 = v43 != v44;
                goto LABEL_43;
              }
              v58 = 1;
              while ( v45 != -4096 )
              {
                v18 = (unsigned int)v12 & (v58 + (_DWORD)v18);
                v110 = v58 + 1;
                v44 = (__int64 *)(v42 + 8LL * (unsigned int)v18);
                v45 = *v44;
                if ( v30 == *v44 )
                  goto LABEL_60;
                v58 = v110;
              }
            }
            v44 = v43;
            goto LABEL_60;
          }
          v31 = *(_QWORD **)(a1 + 320);
          v32 = &v31[*(unsigned int *)(a1 + 328)];
          v33 = v32 != sub_1149D50(v31, (__int64)v32, v14);
LABEL_43:
          if ( !v33 )
          {
            v7 += 8;
            goto LABEL_21;
          }
LABEL_45:
          v34 = v7[12];
          v115[0] = v34;
          if ( v34 == v13 )
            goto LABEL_50;
          if ( *(_DWORD *)(a1 + 304) )
          {
            v18 = *(unsigned int *)(a1 + 312);
            v46 = *(_QWORD *)(a1 + 296);
            v12 = v18;
            v47 = (__int64 *)(v46 + 8 * v18);
            if ( (_DWORD)v18 )
            {
              v12 = (unsigned int)(v18 - 1);
              v18 = (unsigned int)v12 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
              v48 = (__int64 *)(v46 + 8 * v18);
              v49 = *v48;
              if ( v34 == *v48 )
              {
LABEL_63:
                v37 = v47 != v48;
                goto LABEL_48;
              }
              v59 = 1;
              while ( v49 != -4096 )
              {
                v18 = (unsigned int)v12 & (v59 + (_DWORD)v18);
                v111 = v59 + 1;
                v48 = (__int64 *)(v46 + 8LL * (unsigned int)v18);
                v49 = *v48;
                if ( v34 == *v48 )
                  goto LABEL_63;
                v59 = v111;
              }
            }
            v48 = v47;
            goto LABEL_63;
          }
          v35 = *(_QWORD **)(a1 + 320);
          v36 = &v35[*(unsigned int *)(a1 + 328)];
          v37 = v36 != sub_1149D50(v35, (__int64)v36, v14);
LABEL_48:
          if ( !v37 )
          {
            v7 += 12;
            goto LABEL_21;
          }
LABEL_50:
          v7 += 16;
          if ( --v10 )
            continue;
          break;
        }
        v8 = ((char *)v112 - (char *)v7) >> 5;
        if ( v8 == 2 )
          goto LABEL_115;
LABEL_52:
        if ( v8 == 3 )
        {
          v75 = *v7;
          v13 = *(_QWORD *)(a1 + 424);
          v115[0] = v75;
          if ( v75 == v13 )
            goto LABEL_144;
          if ( !*(_DWORD *)(a1 + 304) )
          {
            v76 = *(_QWORD **)(a1 + 320);
            v77 = &v76[*(unsigned int *)(a1 + 328)];
            v78 = v77 != sub_1149D50(v76, (__int64)v77, v115);
LABEL_143:
            if ( !v78 )
              goto LABEL_21;
LABEL_144:
            v7 += 4;
            goto LABEL_116;
          }
          v12 = *(unsigned int *)(a1 + 312);
          v92 = *(_QWORD *)(a1 + 296);
          v93 = v92 + 8 * v12;
          if ( (_DWORD)v12 )
          {
            v94 = v12 - 1;
            v12 = ((_DWORD)v12 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            v18 = v92 + 8 * v12;
            v95 = *(_QWORD *)v18;
            if ( v75 == *(_QWORD *)v18 )
            {
LABEL_175:
              v78 = v93 != v18;
              goto LABEL_143;
            }
            v105 = 1;
            while ( v95 != -4096 )
            {
              v106 = v105 + 1;
              v107 = v94 & (unsigned int)(v12 + v105);
              v12 = (unsigned int)v107;
              v18 = v92 + 8 * v107;
              v95 = *(_QWORD *)v18;
              if ( v75 == *(_QWORD *)v18 )
                goto LABEL_175;
              v105 = v106;
            }
          }
          v18 = v93;
          goto LABEL_175;
        }
        if ( v8 != 1 )
          goto LABEL_54;
        v13 = *(_QWORD *)(a1 + 424);
LABEL_136:
        v72 = *v7;
        v115[0] = v72;
        if ( v72 == v13 )
        {
LABEL_54:
          v19 = v115;
          goto LABEL_22;
        }
        if ( *(_DWORD *)(a1 + 304) )
        {
          v18 = *(unsigned int *)(a1 + 312);
          v82 = *(_QWORD *)(a1 + 296);
          if ( !(_DWORD)v18 )
            goto LABEL_21;
          v12 = ((_DWORD)v18 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
          v83 = (__int64 *)(v82 + 8 * v12);
          v13 = *v83;
          if ( v72 == *v83 )
          {
LABEL_152:
            v19 = v115;
            if ( v83 != (__int64 *)(v82 + 8 * v18) )
              goto LABEL_22;
          }
          else
          {
            v100 = 1;
            while ( v13 != -4096 )
            {
              v101 = v100 + 1;
              v12 = ((_DWORD)v18 - 1) & (unsigned int)(v100 + v12);
              v83 = (__int64 *)(v82 + 8LL * (unsigned int)v12);
              v13 = *v83;
              if ( v72 == *v83 )
                goto LABEL_152;
              v100 = v101;
            }
          }
LABEL_21:
          v19 = v115;
          if ( v7 == v112 )
            goto LABEL_22;
LABEL_33:
          if ( !*(_BYTE *)(a1 + 28) )
            goto LABEL_100;
          v29 = *(_QWORD **)(a1 + 8);
          v18 = *(unsigned int *)(a1 + 20);
          v12 = (__int64)&v29[v18];
          if ( v29 != (_QWORD *)v12 )
          {
            while ( v4 != *v29 )
            {
              if ( (_QWORD *)v12 == ++v29 )
                goto LABEL_37;
            }
            goto LABEL_5;
          }
LABEL_37:
          if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 16) )
            goto LABEL_100;
          goto LABEL_38;
        }
        v73 = *(_QWORD **)(a1 + 320);
        v74 = &v73[*(unsigned int *)(a1 + 328)];
        if ( v74 == sub_1149D50(v73, (__int64)v74, v115) )
          goto LABEL_21;
LABEL_22:
        v115[0] = v4;
LABEL_23:
        sub_114AC10(v113, v19);
        if ( !(unsigned __int8)sub_114B130(a1, v4) )
          return 0;
LABEL_5:
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return 1;
      }
      if ( v5 != 86 )
      {
        if ( v5 != 63 )
        {
          if ( v5 == 85 )
          {
            v71 = *(_QWORD *)(v4 - 32);
            if ( v71 )
            {
              if ( !*(_BYTE *)v71 && *(_QWORD *)(v71 + 24) == *(_QWORD *)(v4 + 80) && (*(_BYTE *)(v71 + 33) & 0x20) != 0 )
              {
                v88 = *(_DWORD *)(v71 + 36);
                if ( v88 == 238 || (unsigned int)(v88 - 240) <= 1 )
                {
                  v89 = *(_QWORD *)(v4 + 32 * (3LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
                  v90 = *(_DWORD *)(v89 + 32);
                  if ( v90 <= 0x40 )
                    v91 = *(_QWORD *)(v89 + 24) == 0;
                  else
                    v91 = v90 == (unsigned int)sub_C444A0(v89 + 24);
                  if ( !v91 )
                    return 0;
                  sub_114AC10(v113, &v114);
                  goto LABEL_5;
                }
              }
            }
LABEL_131:
            if ( !sub_B46A10(v4) )
              return 0;
            goto LABEL_5;
          }
          if ( v5 != 79 )
            goto LABEL_131;
          v60 = *(_QWORD *)(v4 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v60 + 8) - 17 <= 1 )
            v60 = **(_QWORD **)(v60 + 16);
          if ( *(_DWORD *)(a1 + 432) != *(_DWORD *)(v60 + 8) >> 8 && !(unsigned __int8)sub_F0C730(*(_QWORD *)(a1 + 416)) )
          {
            v4 = v114;
            goto LABEL_131;
          }
        }
        sub_114AC10(v113, &v114);
        if ( !(unsigned __int8)sub_114B130(a1, v114) )
          return 0;
        goto LABEL_5;
      }
      v50 = *(_BYTE **)(v4 - 64);
      if ( *v50 <= 0x1Cu )
        return 0;
      v7 = *(__int64 **)(v4 - 32);
      if ( *(_BYTE *)v7 <= 0x1Cu )
        return 0;
      v13 = *(_QWORD *)(a1 + 424);
      v115[0] = *(_QWORD *)(v4 - 64);
      if ( v50 == (_BYTE *)v13 )
        goto LABEL_71;
      if ( !*(_DWORD *)(a1 + 304) )
      {
        v51 = *(_QWORD **)(a1 + 320);
        v52 = &v51[*(unsigned int *)(a1 + 328)];
        v53 = v52 != sub_1149D50(v51, (__int64)v52, v115);
        goto LABEL_70;
      }
      v18 = *(unsigned int *)(a1 + 312);
      v67 = *(_QWORD *)(a1 + 296);
      v12 = v18;
      v68 = (_QWORD *)(v67 + 8 * v18);
      if ( !(_DWORD)v18 )
        goto LABEL_164;
      v12 = (unsigned int)(v18 - 1);
      v18 = (unsigned int)v12 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v69 = (_QWORD *)(v67 + 8 * v18);
      v70 = (_BYTE *)*v69;
      if ( v50 != (_BYTE *)*v69 )
        break;
LABEL_127:
      v53 = v69 != v68;
LABEL_70:
      if ( !v53 )
        goto LABEL_101;
LABEL_71:
      v115[0] = (__int64)v7;
      if ( v7 == (__int64 *)v13 )
        goto LABEL_75;
      if ( !*(_DWORD *)(a1 + 304) )
      {
        v54 = *(_QWORD **)(a1 + 320);
        v55 = &v54[*(unsigned int *)(a1 + 328)];
        v56 = v55 != sub_1149D50(v54, (__int64)v55, v115);
        goto LABEL_74;
      }
      v79 = *(_QWORD *)(a1 + 296);
      v12 = *(unsigned int *)(a1 + 312);
      v80 = v79 + 8 * v12;
      if ( !(_DWORD)v12 )
        goto LABEL_163;
      v12 = (unsigned int)(v12 - 1);
      v81 = v12 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v18 = v79 + 8LL * v81;
      v13 = *(_QWORD *)v18;
      if ( v7 != *(__int64 **)v18 )
      {
        v98 = 1;
        while ( v13 != -4096 )
        {
          v99 = v98 + 1;
          v81 = v12 & (v98 + v81);
          v18 = v79 + 8LL * v81;
          v13 = *(_QWORD *)v18;
          if ( v7 == *(__int64 **)v18 )
            goto LABEL_147;
          v98 = v99;
        }
LABEL_163:
        v18 = v80;
      }
LABEL_147:
      v56 = v18 != v80;
LABEL_74:
      if ( v56 )
      {
LABEL_75:
        v115[0] = v4;
        v19 = v115;
        goto LABEL_23;
      }
LABEL_101:
      if ( !*(_BYTE *)(a1 + 28) )
        goto LABEL_100;
      v61 = *(_QWORD **)(a1 + 8);
      v18 = *(unsigned int *)(a1 + 20);
      v12 = (__int64)&v61[v18];
      if ( v61 == (_QWORD *)v12 )
        goto LABEL_37;
      do
      {
        if ( v4 == *v61 )
          goto LABEL_5;
        ++v61;
      }
      while ( (_QWORD *)v12 != v61 );
      if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 16) )
      {
LABEL_100:
        sub_C8CC70(a1, v4, v12, v18, (__int64)v7, v13);
        goto LABEL_5;
      }
LABEL_38:
      *(_DWORD *)(a1 + 20) = v18 + 1;
      *(_QWORD *)v12 = v4;
      ++*(_QWORD *)a1;
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return 1;
    }
    v96 = 1;
    while ( v70 != (_BYTE *)-4096LL )
    {
      v97 = v96 + 1;
      v18 = (unsigned int)v12 & ((_DWORD)v18 + v96);
      v69 = (_QWORD *)(v67 + 8LL * (unsigned int)v18);
      v70 = (_BYTE *)*v69;
      if ( v50 == (_BYTE *)*v69 )
        goto LABEL_127;
      v96 = v97;
    }
LABEL_164:
    v69 = v68;
    goto LABEL_127;
  }
  return 1;
}
