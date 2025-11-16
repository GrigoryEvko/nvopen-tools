// Function: sub_2CEDAC0
// Address: 0x2cedac0
//
__int64 __fastcall sub_2CEDAC0(_QWORD *a1, __int64 a2, int a3, __int64 a4, unsigned __int8 *a5, _BYTE *a6)
{
  __int64 v6; // rbx
  unsigned int v8; // esi
  unsigned __int8 *v9; // r12
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned int v14; // esi
  int v15; // r11d
  unsigned __int8 **v16; // rdx
  unsigned int v17; // edi
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rcx
  _DWORD *v20; // rax
  unsigned __int64 *v22; // rdx
  unsigned __int64 v23; // rsi
  int v24; // eax
  unsigned __int8 *v25; // rax
  _BYTE *v26; // r13
  unsigned int v27; // r14d
  unsigned int v28; // eax
  unsigned int v29; // esi
  __int64 v30; // r8
  int v31; // r11d
  unsigned __int8 **v32; // rdx
  unsigned int v33; // edi
  unsigned __int8 **v34; // rax
  unsigned __int8 *v35; // rcx
  unsigned __int64 v36; // r13
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rdi
  unsigned int v40; // esi
  unsigned __int8 **v41; // rcx
  __int64 v42; // rcx
  unsigned __int64 v43; // rcx
  _QWORD *v44; // r14
  _QWORD *v45; // rdx
  _QWORD *v46; // r9
  _QWORD *v47; // r8
  _QWORD *v48; // rax
  unsigned int v49; // r14d
  __int64 v50; // r13
  __int64 v51; // r11
  __int64 v52; // rax
  __int64 v53; // r11
  __int64 v54; // rbx
  unsigned __int8 *v55; // rsi
  unsigned __int8 v56; // al
  __int64 v57; // rcx
  __int64 v58; // r8
  unsigned int v59; // edi
  __int64 v60; // rdx
  unsigned __int8 *v61; // r9
  unsigned __int64 *v62; // rdx
  char v63; // al
  unsigned int v64; // esi
  __int64 v65; // rdi
  int v66; // r11d
  unsigned __int8 **v67; // r10
  unsigned int v68; // edx
  _QWORD *v69; // rax
  unsigned __int8 *v70; // rcx
  unsigned __int64 *v71; // rax
  __int64 *v72; // rdx
  __int64 v73; // rdx
  int v74; // eax
  unsigned __int64 v75; // r13
  _BYTE *v76; // rsi
  int v77; // ecx
  int v78; // r10d
  int v79; // edx
  _QWORD *v80; // rax
  int v81; // r10d
  int v82; // r10d
  unsigned int v83; // ecx
  int v84; // eax
  unsigned __int8 *v85; // rsi
  unsigned __int64 *v86; // rsi
  __int64 v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // rdx
  char v90; // di
  int v91; // eax
  int v92; // r9d
  int v93; // r9d
  __int64 v94; // rdi
  unsigned __int8 **v95; // rsi
  unsigned int v96; // r13d
  int v97; // r10d
  unsigned __int8 *v98; // rcx
  int v99; // eax
  int v100; // r11d
  int v101; // r11d
  __int64 v102; // r10
  unsigned int v103; // ecx
  unsigned __int8 *v104; // r8
  int v105; // edi
  unsigned __int8 **v106; // rsi
  int v107; // r10d
  int v108; // r10d
  __int64 v109; // r9
  unsigned __int8 **v110; // rcx
  unsigned int v111; // r13d
  int v112; // esi
  unsigned __int8 *v113; // rdi
  int v114; // eax
  int v115; // r11d
  int v116; // r11d
  __int64 v117; // r9
  __int64 v118; // rdx
  unsigned __int8 *v119; // rsi
  int v120; // r8d
  unsigned __int8 **v121; // rcx
  __int64 *v122; // rdx
  int v123; // r9d
  int v124; // r9d
  __int64 v125; // r8
  int v126; // edi
  __int64 v127; // r14
  unsigned __int8 **v128; // rdx
  unsigned __int8 *v129; // rcx
  int v130; // r11d
  unsigned __int8 **v131; // rdi
  _QWORD *v132; // [rsp+0h] [rbp-C0h]
  __int64 v133; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v134; // [rsp+8h] [rbp-B8h]
  _BYTE *v135; // [rsp+18h] [rbp-A8h]
  int v136; // [rsp+18h] [rbp-A8h]
  int v137; // [rsp+18h] [rbp-A8h]
  int v138; // [rsp+18h] [rbp-A8h]
  __int64 v139; // [rsp+18h] [rbp-A8h]
  _QWORD *v140; // [rsp+18h] [rbp-A8h]
  __int64 v142; // [rsp+28h] [rbp-98h]
  __int64 v145; // [rsp+40h] [rbp-80h]
  unsigned __int8 v147; // [rsp+5Fh] [rbp-61h] BYREF
  __int64 v148[2]; // [rsp+60h] [rbp-60h] BYREF
  char v149; // [rsp+70h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a2 + 56);
  v142 = (__int64)a5;
  v147 = 0;
  v145 = a2 + 48;
  if ( v6 == a2 + 48 )
    return 0;
LABEL_8:
  while ( 2 )
  {
    v9 = (unsigned __int8 *)(v6 - 24);
    if ( !v6 )
      v9 = 0;
    if ( a3 == 1 )
    {
      sub_2CE11B0(a1, (char *)v9);
      v10 = *v9;
      if ( (_BYTE)v10 != 85 )
        goto LABEL_12;
    }
    else
    {
      v10 = *v9;
      if ( (_BYTE)v10 != 85 )
        goto LABEL_12;
    }
    v43 = *((_QWORD *)v9 - 4);
    if ( !v43
      || *(_BYTE *)v43
      || *(_QWORD *)(v43 + 24) != *((_QWORD *)v9 + 10)
      || (*(_BYTE *)(v43 + 33) & 0x20) == 0
      || a3 != 1
      || (unsigned int)(*(_DWORD *)(v43 + 36) - 8927) > 5 )
    {
      if ( *(_BYTE *)(*((_QWORD *)v9 + 1) + 8LL) != 14 )
        goto LABEL_7;
LABEL_54:
      if ( !v43 )
        goto LABEL_95;
      if ( !*(_BYTE *)v43 && *(_QWORD *)(v43 + 24) == *((_QWORD *)v9 + 10) && (*(_BYTE *)(v43 + 33) & 0x20) != 0 )
      {
        v8 = 15;
        if ( *(_DWORD *)(v43 + 36) == 8170 )
        {
          if ( (v9[7] & 0x40) != 0 )
            v86 = (unsigned __int64 *)*((_QWORD *)v9 - 1);
          else
            v86 = (unsigned __int64 *)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
          v137 = sub_2CED090((__int64)a1, *v86, a4, *(_QWORD *)(a2 + 72));
          sub_2CEC7C0((__int64)v9, v137, a4, &v147);
          v8 = v137;
        }
        goto LABEL_6;
      }
      v44 = (_QWORD *)a1[2];
      if ( !v44 )
        goto LABEL_95;
      v45 = (_QWORD *)v44[2];
      v46 = v44 + 1;
      if ( !v45 )
        goto LABEL_95;
      v47 = v44 + 1;
      v48 = (_QWORD *)v44[2];
      do
      {
        if ( v48[4] < v43 )
        {
          v48 = (_QWORD *)v48[3];
        }
        else
        {
          v47 = v48;
          v48 = (_QWORD *)v48[2];
        }
      }
      while ( v48 );
      if ( v46 != v47 && (v75 = (unsigned __int64)(v44 + 1), v47[4] <= v43) )
      {
        do
        {
          if ( v45[4] < v43 )
          {
            v45 = (_QWORD *)v45[3];
          }
          else
          {
            v75 = (unsigned __int64)v45;
            v45 = (_QWORD *)v45[2];
          }
        }
        while ( v45 );
        if ( (_QWORD *)v75 == v46 || *(_QWORD *)(v75 + 32) > v43 )
        {
          v134 = v43;
          v139 = v75;
          v132 = v44 + 1;
          v87 = sub_22077B0(0x30u);
          *(_DWORD *)(v87 + 40) = 0;
          v75 = v87;
          *(_QWORD *)(v87 + 32) = v134;
          v88 = sub_2CBBD90(v44, v139, (unsigned __int64 *)(v87 + 32));
          if ( v89 )
          {
            v90 = v88 || v132 == v89 || v89[4] > v134;
            sub_220F040(v90, v75, v89, v132);
            ++v44[5];
          }
          else
          {
            v140 = v88;
            j_j___libc_free_0(v75);
            v75 = (unsigned __int64)v140;
          }
        }
        v8 = *(_DWORD *)(v75 + 40);
        if ( v8 <= 6 )
        {
          if ( v8 )
          {
            switch ( v8 )
            {
              case 1u:
              case 4u:
                goto LABEL_6;
              case 3u:
                v8 = 2;
                break;
              case 5u:
                v8 = 8;
                break;
              case 6u:
                v8 = 32;
                break;
              default:
                goto LABEL_95;
            }
            goto LABEL_6;
          }
          goto LABEL_95;
        }
        if ( v8 != 101 )
          goto LABEL_95;
        v8 = 16;
      }
      else
      {
LABEL_95:
        v8 = 15;
      }
LABEL_6:
      sub_2CEC7C0((__int64)v9, v8, a4, &v147);
      goto LABEL_7;
    }
    v148[0] = (__int64)v9;
    v76 = (_BYTE *)a1[13];
    if ( v76 == (_BYTE *)a1[14] )
    {
      sub_2CE3020((__int64)(a1 + 12), v76, v148);
    }
    else
    {
      if ( v76 )
      {
        *(_QWORD *)v76 = v9;
        v76 = (_BYTE *)a1[13];
      }
      a1[13] = v76 + 8;
    }
    v10 = *v9;
LABEL_12:
    v11 = *((_QWORD *)v9 + 1);
    v12 = *(unsigned __int8 *)(v11 + 8);
    if ( (_BYTE)v12 != 14 )
    {
      if ( (_BYTE)v10 == 61 )
      {
        if ( (unsigned int)(v12 - 15) > 1 )
          goto LABEL_7;
LABEL_5:
        v8 = sub_2CED390(a1, (unsigned __int64)v9, a4, *(_QWORD *)(a2 + 72), a6);
        goto LABEL_6;
      }
      if ( (_BYTE)v10 != 84 || (unsigned int)(v12 - 15) > 1 )
        goto LABEL_7;
LABEL_66:
      v49 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
      if ( !v49 )
        goto LABEL_76;
      v50 = 0;
      v51 = v49;
      v49 = 0;
      v52 = 32 * v51;
      v53 = v6;
      v54 = v52;
      while ( 1 )
      {
        v55 = *(unsigned __int8 **)(*((_QWORD *)v9 - 1) + v50);
        v148[0] = (__int64)v55;
        v56 = *v55;
        if ( *v55 == 20 || (unsigned int)v56 - 12 <= 1 )
          goto LABEL_68;
        v57 = *(unsigned int *)(a4 + 24);
        v58 = *(_QWORD *)(a4 + 8);
        if ( !(_DWORD)v57 )
          goto LABEL_131;
        v59 = (v57 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
        v60 = v58 + 16LL * v59;
        v61 = *(unsigned __int8 **)v60;
        if ( v55 != *(unsigned __int8 **)v60 )
        {
          v79 = 1;
          while ( v61 != (unsigned __int8 *)-4096LL )
          {
            v59 = (v57 - 1) & (v79 + v59);
            v138 = v79 + 1;
            v60 = v58 + 16LL * v59;
            v61 = *(unsigned __int8 **)v60;
            if ( v55 == *(unsigned __int8 **)v60 )
              goto LABEL_73;
            v79 = v138;
          }
          goto LABEL_131;
        }
LABEL_73:
        if ( v60 == v58 + 16 * v57 )
        {
LABEL_131:
          if ( v56 <= 0x1Cu )
          {
            v133 = v53;
            v136 = sub_2CECAD0((__int64)a1, (unsigned __int64)v55, a4, *(_QWORD *)(a2 + 72));
            v80 = sub_10E84F0(a4, v148);
            v53 = v133;
            *(_DWORD *)v80 = v136;
            v49 |= v136;
          }
          else
          {
            *a6 = 1;
          }
LABEL_68:
          v50 += 32;
          if ( v54 == v50 )
            goto LABEL_75;
        }
        else
        {
          v50 += 32;
          v49 |= *(_DWORD *)(v60 + 8);
          if ( v54 == v50 )
          {
LABEL_75:
            v6 = v53;
LABEL_76:
            sub_2CEC7C0((__int64)v9, v49, a4, &v147);
LABEL_7:
            v6 = *(_QWORD *)(v6 + 8);
            if ( v145 == v6 )
              return v147;
            goto LABEL_8;
          }
        }
      }
    }
    v13 = (unsigned int)(v10 - 60);
    switch ( (int)v13 )
    {
      case 0:
        sub_2CEC7C0((__int64)v9, 8, a4, &v147);
        goto LABEL_7;
      case 1:
        goto LABEL_5;
      case 3:
      case 18:
        if ( (v9[7] & 0x40) != 0 )
        {
          v62 = (unsigned __int64 *)*((_QWORD *)v9 - 1);
          v23 = *v62;
          v24 = *(_DWORD *)(*(_QWORD *)(*v62 + 8) + 8LL) >> 8;
          if ( v24 )
            goto LABEL_21;
        }
        else
        {
          v22 = (unsigned __int64 *)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
          v23 = *v22;
          v24 = *(_DWORD *)(*(_QWORD *)(*v22 + 8) + 8LL) >> 8;
          if ( v24 )
          {
LABEL_21:
            switch ( v24 )
            {
              case 1:
                goto LABEL_118;
              case 2:
                goto LABEL_95;
              case 3:
                goto LABEL_121;
              case 4:
                goto LABEL_120;
              case 5:
                goto LABEL_117;
              case 6:
                goto LABEL_119;
              default:
                goto LABEL_97;
            }
          }
        }
        v8 = sub_2CED090((__int64)a1, v23, a4, *(_QWORD *)(a2 + 72));
        goto LABEL_6;
      case 17:
        v38 = *(unsigned int *)(a4 + 24);
        v39 = *(_QWORD *)(a4 + 8);
        if ( !(_DWORD)v38 )
          goto LABEL_40;
        v40 = (v38 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v41 = (unsigned __int8 **)(v39 + 16LL * v40);
        a5 = *v41;
        if ( v9 == *v41 )
          goto LABEL_39;
        v77 = 1;
        while ( 2 )
        {
          if ( a5 != (unsigned __int8 *)-4096LL )
          {
            v78 = v77 + 1;
            v40 = (v38 - 1) & (v77 + v40);
            v41 = (unsigned __int8 **)(v39 + 16LL * v40);
            a5 = *v41;
            if ( v9 != *v41 )
            {
              v77 = v78;
              continue;
            }
LABEL_39:
            if ( v41 != (unsigned __int8 **)(v39 + 16 * v38) )
              goto LABEL_7;
          }
          break;
        }
LABEL_40:
        if ( !(_BYTE)qword_5013D28 || !unk_50142AD )
          goto LABEL_42;
        if ( !(unsigned __int8)sub_CE9220(*(_QWORD *)(a2 + 72))
          || ((v9[7] & 0x40) == 0
            ? (v122 = (__int64 *)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)])
            : (v122 = (__int64 *)*((_QWORD *)v9 - 1)),
              v8 = 1,
              !(unsigned __int8)sub_2CE8060(*v122)) )
        {
          v11 = *((_QWORD *)v9 + 1);
LABEL_42:
          v42 = v11;
          if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
            v42 = **(_QWORD **)(v11 + 16);
          v24 = *(_DWORD *)(v42 + 8) >> 8;
          if ( *(_DWORD *)(v42 + 8) > 0x6FFu )
          {
LABEL_97:
            v8 = (v24 == 101) + 15;
          }
          else
          {
            v8 = 15;
            if ( v24 )
            {
              switch ( v24 )
              {
                case 1:
LABEL_118:
                  v8 = 1;
                  break;
                case 3:
LABEL_121:
                  v8 = 2;
                  break;
                case 4:
LABEL_120:
                  v8 = 4;
                  break;
                case 5:
LABEL_117:
                  v8 = 8;
                  break;
                case 6:
LABEL_119:
                  v8 = 32;
                  break;
                default:
                  goto LABEL_95;
              }
            }
          }
        }
        goto LABEL_6;
      case 19:
        v37 = *(_DWORD *)(v11 + 8) >> 8;
        if ( v37 )
        {
          switch ( v37 )
          {
            case 1:
              goto LABEL_112;
            case 2:
              goto LABEL_126;
            case 3:
              goto LABEL_114;
            case 4:
              goto LABEL_115;
            case 5:
              goto LABEL_113;
            case 6:
              goto LABEL_116;
            default:
              goto LABEL_96;
          }
        }
        if ( (v9[7] & 0x40) != 0 )
          v72 = (__int64 *)*((_QWORD *)v9 - 1);
        else
          v72 = (__int64 *)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
        v73 = *v72;
        v74 = *(_DWORD *)(*(_QWORD *)(v73 + 8) + 8LL) >> 8;
        if ( *(_DWORD *)(*(_QWORD *)(v73 + 8) + 8LL) > 0x6FFu )
        {
          v49 = (v74 == 101) + 15;
        }
        else
        {
          v49 = 15;
          if ( v74 )
          {
            switch ( v74 )
            {
              case 1:
                v49 = 1;
                break;
              case 3:
                v49 = 2;
                break;
              case 4:
                v49 = 4;
                break;
              case 5:
                v49 = 8;
                break;
              case 6:
                v49 = 32;
                break;
              default:
                v49 = 15;
                break;
            }
          }
        }
        v148[0] = v73;
        *(_DWORD *)sub_2791170(a4, v148) = v49;
        goto LABEL_76;
      case 24:
        v37 = *(_DWORD *)(v11 + 8) >> 8;
        if ( !v37 )
          goto LABEL_66;
        switch ( v37 )
        {
          case 1:
LABEL_112:
            v49 = 1;
            break;
          case 2:
LABEL_126:
            v49 = 15;
            break;
          case 3:
LABEL_114:
            v49 = 2;
            break;
          case 4:
LABEL_115:
            v49 = 4;
            break;
          case 5:
LABEL_113:
            v49 = 8;
            break;
          case 6:
LABEL_116:
            v49 = 32;
            break;
          default:
LABEL_96:
            v49 = (v37 == 101) + 15;
            break;
        }
        goto LABEL_76;
      case 25:
        v43 = *((_QWORD *)v9 - 4);
        goto LABEL_54;
      case 26:
        if ( (v9[7] & 0x40) != 0 )
          v25 = (unsigned __int8 *)*((_QWORD *)v9 - 1);
        else
          v25 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
        v26 = (_BYTE *)*((_QWORD *)v25 + 8);
        v135 = (_BYTE *)*((_QWORD *)v25 + 4);
        v27 = sub_2CED090((__int64)a1, (unsigned __int64)v135, a4, *(_QWORD *)(a2 + 72));
        v28 = sub_2CED090((__int64)a1, (unsigned __int64)v26, a4, *(_QWORD *)(a2 + 72));
        v8 = v28;
        if ( *v135 != 20 )
        {
          v8 = v27 | v28;
          if ( *v26 == 20 )
            v8 = v27;
        }
        goto LABEL_6;
      case 33:
        if ( a3 != 1 )
        {
          v29 = *(_DWORD *)(v142 + 24);
          if ( v29 )
          {
            v30 = *(_QWORD *)(v142 + 8);
            v31 = 1;
            v32 = 0;
            v33 = (v29 - 1) & (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9));
            v34 = (unsigned __int8 **)(v30 + 16LL * v33);
            v35 = *v34;
            if ( v9 == *v34 )
            {
LABEL_31:
              v36 = (unsigned __int64)v34[1];
LABEL_32:
              if ( !v36 )
                goto LABEL_95;
              v8 = sub_2CED090((__int64)a1, v36, a4, *(_QWORD *)(a2 + 72));
              goto LABEL_6;
            }
            while ( v35 != (unsigned __int8 *)-4096LL )
            {
              if ( !v32 && v35 == (unsigned __int8 *)-8192LL )
                v32 = v34;
              v33 = (v29 - 1) & (v31 + v33);
              v34 = (unsigned __int8 **)(v30 + 16LL * v33);
              v35 = *v34;
              if ( v9 == *v34 )
                goto LABEL_31;
              ++v31;
            }
            if ( !v32 )
              v32 = v34;
            ++*(_QWORD *)v142;
            v99 = *(_DWORD *)(v142 + 16) + 1;
            if ( 4 * v99 < 3 * v29 )
            {
              if ( v29 - *(_DWORD *)(v142 + 20) - v99 > v29 >> 3 )
              {
LABEL_212:
                *(_DWORD *)(v142 + 16) = v99;
                if ( *v32 != (unsigned __int8 *)-4096LL )
                  --*(_DWORD *)(v142 + 20);
                *v32 = v9;
                v8 = 15;
                v32[1] = 0;
                goto LABEL_6;
              }
              sub_2CED8E0(v142, v29);
              v107 = *(_DWORD *)(v142 + 24);
              if ( v107 )
              {
                v108 = v107 - 1;
                v109 = *(_QWORD *)(v142 + 8);
                v110 = 0;
                v111 = v108 & (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9));
                v112 = 1;
                v99 = *(_DWORD *)(v142 + 16) + 1;
                v32 = (unsigned __int8 **)(v109 + 16LL * v111);
                v113 = *v32;
                if ( v9 != *v32 )
                {
                  while ( v113 != (unsigned __int8 *)-4096LL )
                  {
                    if ( !v110 && v113 == (unsigned __int8 *)-8192LL )
                      v110 = v32;
                    v111 = v108 & (v112 + v111);
                    v32 = (unsigned __int8 **)(v109 + 16LL * v111);
                    v113 = *v32;
                    if ( v9 == *v32 )
                      goto LABEL_212;
                    ++v112;
                  }
                  if ( v110 )
                    v32 = v110;
                }
                goto LABEL_212;
              }
LABEL_300:
              ++*(_DWORD *)(v142 + 16);
              BUG();
            }
          }
          else
          {
            ++*(_QWORD *)v142;
          }
          sub_2CED8E0(v142, 2 * v29);
          v100 = *(_DWORD *)(v142 + 24);
          if ( v100 )
          {
            v101 = v100 - 1;
            v102 = *(_QWORD *)(v142 + 8);
            v103 = v101 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v99 = *(_DWORD *)(v142 + 16) + 1;
            v32 = (unsigned __int8 **)(v102 + 16LL * v103);
            v104 = *v32;
            if ( v9 != *v32 )
            {
              v105 = 1;
              v106 = 0;
              while ( v104 != (unsigned __int8 *)-4096LL )
              {
                if ( v104 == (unsigned __int8 *)-8192LL && !v106 )
                  v106 = v32;
                v103 = v101 & (v105 + v103);
                v32 = (unsigned __int8 **)(v102 + 16LL * v103);
                v104 = *v32;
                if ( v9 == *v32 )
                  goto LABEL_212;
                ++v105;
              }
              if ( v106 )
                v32 = v106;
            }
            goto LABEL_212;
          }
          goto LABEL_300;
        }
        v148[0] = (__int64)&v149;
        v148[1] = 0x400000000LL;
        v36 = *((_QWORD *)v9 - 4);
        while ( 1 )
        {
          v63 = *(_BYTE *)v36;
          if ( *(_BYTE *)v36 <= 0x1Cu )
          {
            if ( v63 != 22 )
              v36 = 0;
            goto LABEL_84;
          }
          if ( v63 != 93 )
            break;
          v36 = *(_QWORD *)(v36 - 32);
          if ( !v36 )
            BUG();
        }
        if ( v63 != 84 )
        {
          if ( v63 == 61 )
            goto LABEL_84;
          while ( v63 == 94 )
          {
            if ( (unsigned __int8)sub_2CDD660(v36, (__int64)v9, v13, v12, (unsigned int)a5) )
            {
              v36 = *(_QWORD *)(v36 - 32);
              goto LABEL_84;
            }
            v36 = *(_QWORD *)(v36 - 64);
            if ( !v36 )
              BUG();
            v63 = *(_BYTE *)v36;
          }
          goto LABEL_110;
        }
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v36 + 8) + 8LL) - 15 > 1 )
        {
LABEL_110:
          v36 = 0;
          goto LABEL_84;
        }
        if ( ***(_BYTE ***)(v36 - 8) != 61 )
          v36 = 0;
LABEL_84:
        v64 = *(_DWORD *)(v142 + 24);
        if ( v64 )
        {
          v65 = *(_QWORD *)(v142 + 8);
          v66 = 1;
          v67 = 0;
          v68 = (v64 - 1) & (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9));
          v69 = (_QWORD *)(v65 + 16LL * v68);
          v70 = (unsigned __int8 *)*v69;
          if ( v9 == (unsigned __int8 *)*v69 )
          {
LABEL_86:
            v71 = v69 + 1;
LABEL_87:
            *v71 = v36;
            goto LABEL_32;
          }
          while ( v70 != (unsigned __int8 *)-4096LL )
          {
            if ( !v67 && v70 == (unsigned __int8 *)-8192LL )
              v67 = (unsigned __int8 **)v69;
            v68 = (v64 - 1) & (v66 + v68);
            v69 = (_QWORD *)(v65 + 16LL * v68);
            v70 = (unsigned __int8 *)*v69;
            if ( v9 == (unsigned __int8 *)*v69 )
              goto LABEL_86;
            ++v66;
          }
          if ( !v67 )
            v67 = (unsigned __int8 **)v69;
          ++*(_QWORD *)v142;
          v114 = *(_DWORD *)(v142 + 16) + 1;
          if ( 4 * v114 < 3 * v64 )
          {
            if ( v64 - *(_DWORD *)(v142 + 20) - v114 > v64 >> 3 )
            {
LABEL_239:
              *(_DWORD *)(v142 + 16) = v114;
              if ( *v67 != (unsigned __int8 *)-4096LL )
                --*(_DWORD *)(v142 + 20);
              *v67 = v9;
              v71 = (unsigned __int64 *)(v67 + 1);
              v67[1] = 0;
              goto LABEL_87;
            }
            sub_2CED8E0(v142, v64);
            v123 = *(_DWORD *)(v142 + 24);
            if ( v123 )
            {
              v124 = v123 - 1;
              v125 = *(_QWORD *)(v142 + 8);
              v126 = 1;
              LODWORD(v127) = v124 & (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9));
              v128 = 0;
              v114 = *(_DWORD *)(v142 + 16) + 1;
              v67 = (unsigned __int8 **)(v125 + 16LL * (unsigned int)v127);
              v129 = *v67;
              if ( v9 != *v67 )
              {
                while ( v129 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v128 && v129 == (unsigned __int8 *)-8192LL )
                    v128 = v67;
                  v127 = v124 & (unsigned int)(v127 + v126);
                  v67 = (unsigned __int8 **)(v125 + 16 * v127);
                  v129 = *v67;
                  if ( v9 == *v67 )
                    goto LABEL_239;
                  ++v126;
                }
                if ( v128 )
                  v67 = v128;
              }
              goto LABEL_239;
            }
LABEL_303:
            ++*(_DWORD *)(v142 + 16);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)v142;
        }
        sub_2CED8E0(v142, 2 * v64);
        v115 = *(_DWORD *)(v142 + 24);
        if ( v115 )
        {
          v116 = v115 - 1;
          v117 = *(_QWORD *)(v142 + 8);
          LODWORD(v118) = v116 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v114 = *(_DWORD *)(v142 + 16) + 1;
          v67 = (unsigned __int8 **)(v117 + 16LL * (unsigned int)v118);
          v119 = *v67;
          if ( v9 != *v67 )
          {
            v120 = 1;
            v121 = 0;
            while ( v119 != (unsigned __int8 *)-4096LL )
            {
              if ( !v121 && v119 == (unsigned __int8 *)-8192LL )
                v121 = v67;
              v118 = v116 & (unsigned int)(v118 + v120);
              v67 = (unsigned __int8 **)(v117 + 16 * v118);
              v119 = *v67;
              if ( v9 == *v67 )
                goto LABEL_239;
              ++v120;
            }
            if ( v121 )
              v67 = v121;
          }
          goto LABEL_239;
        }
        goto LABEL_303;
      default:
        v14 = *(_DWORD *)(a4 + 24);
        if ( !v14 )
        {
          ++*(_QWORD *)a4;
          goto LABEL_141;
        }
        a5 = *(unsigned __int8 **)(a4 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v14 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v18 = &a5[16 * v17];
        v19 = *(unsigned __int8 **)v18;
        if ( v9 != *(unsigned __int8 **)v18 )
        {
          while ( v19 != (unsigned __int8 *)-4096LL )
          {
            if ( v19 == (unsigned __int8 *)-8192LL && !v16 )
              v16 = (unsigned __int8 **)v18;
            v17 = (v14 - 1) & (v15 + v17);
            v18 = &a5[16 * v17];
            v19 = *(unsigned __int8 **)v18;
            if ( v9 == *(unsigned __int8 **)v18 )
              goto LABEL_16;
            ++v15;
          }
          if ( !v16 )
            v16 = (unsigned __int8 **)v18;
          v91 = *(_DWORD *)(a4 + 16);
          ++*(_QWORD *)a4;
          v84 = v91 + 1;
          if ( 4 * v84 < 3 * v14 )
          {
            if ( v14 - *(_DWORD *)(a4 + 20) - v84 > v14 >> 3 )
            {
LABEL_143:
              *(_DWORD *)(a4 + 16) = v84;
              if ( *v16 != (unsigned __int8 *)-4096LL )
                --*(_DWORD *)(a4 + 20);
              *v16 = v9;
              v20 = v16 + 1;
              *((_DWORD *)v16 + 2) = 0;
              goto LABEL_17;
            }
            sub_D39D40(a4, v14);
            v92 = *(_DWORD *)(a4 + 24);
            if ( v92 )
            {
              v93 = v92 - 1;
              v94 = *(_QWORD *)(a4 + 8);
              v95 = 0;
              v96 = v93 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v97 = 1;
              v84 = *(_DWORD *)(a4 + 16) + 1;
              v16 = (unsigned __int8 **)(v94 + 16LL * v96);
              v98 = *v16;
              if ( v9 != *v16 )
              {
                while ( v98 != (unsigned __int8 *)-4096LL )
                {
                  if ( v98 == (unsigned __int8 *)-8192LL && !v95 )
                    v95 = v16;
                  LODWORD(a5) = v97 + 1;
                  v96 = v93 & (v97 + v96);
                  v16 = (unsigned __int8 **)(v94 + 16LL * v96);
                  v98 = *v16;
                  if ( v9 == *v16 )
                    goto LABEL_143;
                  ++v97;
                }
                if ( v95 )
                  v16 = v95;
              }
              goto LABEL_143;
            }
LABEL_302:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
LABEL_141:
          sub_D39D40(a4, 2 * v14);
          v81 = *(_DWORD *)(a4 + 24);
          if ( v81 )
          {
            v82 = v81 - 1;
            a5 = *(unsigned __int8 **)(a4 + 8);
            v83 = v82 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v84 = *(_DWORD *)(a4 + 16) + 1;
            v16 = (unsigned __int8 **)&a5[16 * v83];
            v85 = *v16;
            if ( v9 != *v16 )
            {
              v130 = 1;
              v131 = 0;
              while ( v85 != (unsigned __int8 *)-4096LL )
              {
                if ( !v131 && v85 == (unsigned __int8 *)-8192LL )
                  v131 = v16;
                v83 = v82 & (v130 + v83);
                v16 = (unsigned __int8 **)&a5[16 * v83];
                v85 = *v16;
                if ( v9 == *v16 )
                  goto LABEL_143;
                ++v130;
              }
              if ( v131 )
                v16 = v131;
            }
            goto LABEL_143;
          }
          goto LABEL_302;
        }
LABEL_16:
        v20 = v18 + 8;
LABEL_17:
        *v20 = 15;
        v6 = *(_QWORD *)(v6 + 8);
        if ( v145 != v6 )
          continue;
        return v147;
    }
  }
}
