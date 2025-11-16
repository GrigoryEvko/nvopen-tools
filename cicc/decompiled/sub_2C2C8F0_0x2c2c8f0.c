// Function: sub_2C2C8F0
// Address: 0x2c2c8f0
//
__int64 __fastcall sub_2C2C8F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 *v7; // rax
  unsigned int v8; // r15d
  __int64 *v9; // r10
  __int64 *v10; // r13
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rsi
  int v21; // ecx
  __int64 v22; // rdi
  int v23; // ecx
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r8
  const char *v27; // r11
  __int64 v28; // rbx
  _QWORD *v29; // rdi
  __int64 v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // r8
  _QWORD *v33; // r9
  __int64 v34; // r11
  __int64 v35; // rax
  int v36; // eax
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rbx
  unsigned __int8 v41; // al
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r10
  __int64 v48; // rcx
  unsigned __int8 *v49; // rax
  unsigned int v50; // eax
  __int64 v51; // rax
  char v52; // cl
  __int64 v53; // rdx
  _QWORD *v54; // rcx
  int v55; // eax
  int v56; // r9d
  __int64 v57; // rax
  char v58; // cl
  __int64 v59; // rdx
  _QWORD *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rdx
  char v64; // cl
  __int64 v65; // rax
  __int64 v66; // rax
  char v67; // cl
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdi
  __int64 v72; // r10
  _BYTE *v73; // rax
  int v74; // eax
  __int64 v75; // r10
  bool v76; // al
  char v77; // cl
  __int64 *v79; // [rsp+8h] [rbp-168h]
  __int64 *v80; // [rsp+10h] [rbp-160h]
  __int64 v81; // [rsp+18h] [rbp-158h]
  __int64 v82; // [rsp+18h] [rbp-158h]
  __int64 v83; // [rsp+18h] [rbp-158h]
  const char *v84; // [rsp+20h] [rbp-150h]
  __int64 v85; // [rsp+20h] [rbp-150h]
  __int64 v86; // [rsp+20h] [rbp-150h]
  __int64 v87; // [rsp+20h] [rbp-150h]
  __int64 v88; // [rsp+20h] [rbp-150h]
  __int64 v89; // [rsp+20h] [rbp-150h]
  __int64 v90; // [rsp+20h] [rbp-150h]
  __int64 v91; // [rsp+20h] [rbp-150h]
  __int64 v92; // [rsp+20h] [rbp-150h]
  __int64 v93; // [rsp+20h] [rbp-150h]
  __int64 v95; // [rsp+30h] [rbp-140h]
  __int64 v96; // [rsp+38h] [rbp-138h]
  __int64 v97; // [rsp+48h] [rbp-128h]
  __int64 v98; // [rsp+48h] [rbp-128h]
  __int64 v99; // [rsp+48h] [rbp-128h]
  __int64 v100; // [rsp+48h] [rbp-128h]
  __int64 v101; // [rsp+48h] [rbp-128h]
  __int64 v102; // [rsp+48h] [rbp-128h]
  __int64 v103; // [rsp+48h] [rbp-128h]
  __int64 v104; // [rsp+48h] [rbp-128h]
  __int64 v105; // [rsp+48h] [rbp-128h]
  __int64 v106; // [rsp+48h] [rbp-128h]
  __int64 v107; // [rsp+48h] [rbp-128h]
  __int64 v108; // [rsp+48h] [rbp-128h]
  __int64 v109; // [rsp+48h] [rbp-128h]
  __int64 v110; // [rsp+50h] [rbp-120h]
  _BYTE *v111; // [rsp+50h] [rbp-120h]
  __int64 v112; // [rsp+50h] [rbp-120h]
  __int64 v113; // [rsp+58h] [rbp-118h]
  __int64 v114; // [rsp+60h] [rbp-110h]
  __int64 *v115; // [rsp+68h] [rbp-108h]
  __int64 v116; // [rsp+70h] [rbp-100h] BYREF
  __int64 v117; // [rsp+78h] [rbp-F8h] BYREF
  __int64 v118; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v119; // [rsp+88h] [rbp-E8h] BYREF
  _QWORD v120[2]; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v121; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v122; // [rsp+A8h] [rbp-C8h]
  const char *v123; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v124; // [rsp+B8h] [rbp-B8h]
  __int16 v125; // [rsp+D0h] [rbp-A0h]
  const char *v126; // [rsp+E0h] [rbp-90h] BYREF
  int v127; // [rsp+E8h] [rbp-88h]
  __int64 *v128; // [rsp+F0h] [rbp-80h]
  __int16 v129; // [rsp+100h] [rbp-70h]
  __int64 v130; // [rsp+110h] [rbp-60h] BYREF
  __int64 v131; // [rsp+118h] [rbp-58h]
  __int64 v132; // [rsp+120h] [rbp-50h]
  __int64 v133; // [rsp+128h] [rbp-48h]
  _QWORD *v134; // [rsp+130h] [rbp-40h]
  __int64 v135; // [rsp+138h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v2 + 64) != 1 )
    BUG();
  v114 = **(_QWORD **)(**(_QWORD **)(v2 + 56) + 56LL);
  v3 = sub_2AAFF80(a1);
  if ( !*(_DWORD *)(v3 + 56) )
    BUG();
  v4 = *(unsigned int *)(a1 + 24);
  v5 = **(_QWORD **)(v3 + 48);
  v130 = 0;
  v6 = *(_QWORD **)(*(_QWORD *)(v5 + 40) + 8LL);
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = v6;
  v135 = *v6;
  v7 = *(__int64 **)(a1 + 16);
  v79 = &v7[v4];
  if ( v7 == v79 )
  {
    v14 = 0;
    v15 = 0;
    return sub_C7D6A0(v14, v15, 8);
  }
  v80 = *(__int64 **)(a1 + 16);
  do
  {
    v96 = *v80;
    v95 = *v80 + 112;
    if ( v95 == *(_QWORD *)(*v80 + 120) )
      goto LABEL_18;
    v113 = *(_QWORD *)(*v80 + 120);
    do
    {
      if ( !v113 )
        BUG();
      if ( **(_BYTE **)(v113 + 72) != 84 )
        break;
      v8 = 0;
      v9 = *(__int64 **)(v96 + 56);
      v10 = v9;
      v115 = &v9[*(unsigned int *)(v96 + 64)];
      if ( v115 != v9 )
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( *v10 != v114 )
            goto LABEL_11;
          v12 = *(_QWORD *)(*(_QWORD *)(v113 + 24) + 8LL * v8);
          LODWORD(v122) = 64;
          v97 = v12;
          v121 = 1;
          sub_9865C0((__int64)&v123, (__int64)&v121);
          sub_9865C0((__int64)&v126, (__int64)&v123);
          v128 = &v116;
          sub_969240((__int64 *)&v123);
          v13 = sub_2BF04A0(v97);
          if ( !v13 )
            goto LABEL_16;
          if ( *(_BYTE *)(v13 + 8) != 4 )
            goto LABEL_16;
          if ( *(_BYTE *)(v13 + 160) != 82 )
            goto LABEL_16;
          v98 = v13;
          v17 = **(_QWORD **)(v13 + 48);
          if ( !v17 )
            goto LABEL_16;
          *v128 = v17;
          sub_9865C0((__int64)&v123, (__int64)&v126);
          if ( !sub_2C2C640((__int64)&v123, *(_QWORD *)(*(_QWORD *)(v98 + 48) + 8LL)) )
          {
            sub_969240((__int64 *)&v123);
LABEL_16:
            sub_969240((__int64 *)&v126);
            sub_969240(&v121);
            goto LABEL_11;
          }
          sub_969240((__int64 *)&v123);
          sub_969240((__int64 *)&v126);
          sub_969240(&v121);
          v99 = v116;
          v18 = sub_2BF04A0(v116);
          if ( v18 && (unsigned int)*(unsigned __int8 *)(v18 + 8) - 33 <= 1 && v99 )
          {
            v19 = v99 - 96;
            v20 = v99;
            if ( *(_BYTE *)(v99 - 88) == 33 && *(_QWORD *)(v99 + 64) )
              goto LABEL_11;
            goto LABEL_29;
          }
          v44 = sub_2BF0490(v99);
          if ( !v44 || *(_DWORD *)(v44 + 56) != 2 )
            goto LABEL_11;
          v85 = v44;
          v81 = **(_QWORD **)(v44 + 48);
          v45 = sub_2BF04A0(v81);
          v46 = v99;
          if ( !v45 || (unsigned int)*(unsigned __int8 *)(v45 + 8) - 33 > 1 || (v47 = v81 - 96, !v81) )
          {
            v89 = *(_QWORD *)(*(_QWORD *)(v85 + 48) + 8LL);
            v65 = sub_2BF04A0(v89);
            if ( !v65 || (unsigned int)*(unsigned __int8 *)(v65 + 8) - 33 > 1 || !v89 )
              goto LABEL_11;
            v46 = v99;
            v47 = v89 - 96;
          }
          v48 = *(_QWORD *)(v47 + 152);
          v49 = *(unsigned __int8 **)(v48 + 40);
          if ( !v49 )
            goto LABEL_85;
          v50 = *v49 - 29;
          if ( v50 != 15 )
          {
            if ( v50 > 0xF )
            {
              if ( v50 == 16 )
              {
                v87 = *(_QWORD *)(*(_QWORD *)(v47 + 48) + 8LL);
                v102 = v47;
                v57 = sub_2BF04A0(v46);
                if ( !v57 )
                  goto LABEL_11;
                v58 = *(_BYTE *)(v57 + 8);
                v19 = v102;
                v59 = v87;
                switch ( v58 )
                {
                  case 23:
LABEL_73:
                    if ( *(_DWORD *)(v57 + 160) != 16 )
                      goto LABEL_11;
                    break;
                  case 9:
                    if ( **(_BYTE **)(v57 + 136) != 45 )
                      goto LABEL_11;
                    break;
                  case 16:
                    goto LABEL_73;
                  default:
                    if ( v58 != 4 || *(_BYTE *)(v57 + 160) != 16 )
                      goto LABEL_11;
                    break;
                }
              }
              else
              {
LABEL_85:
                if ( *(_DWORD *)(v48 + 24) != 2 )
                  goto LABEL_11;
                v88 = *(_QWORD *)(*(_QWORD *)(v47 + 48) + 8LL);
                v103 = v47;
                v57 = sub_2BF04A0(v46);
                if ( !v57 )
                  goto LABEL_11;
                v64 = *(_BYTE *)(v57 + 8);
                v19 = v103;
                v59 = v88;
                if ( v64 == 23 )
                {
                  if ( *(_DWORD *)(v57 + 160) != 34 )
                    goto LABEL_11;
                }
                else if ( v64 == 9 )
                {
                  if ( **(_BYTE **)(v57 + 136) != 63 )
                    goto LABEL_11;
                }
                else if ( v64 != 17 && (v64 != 4 || *(_BYTE *)(v57 + 160) != 34) )
                {
                  goto LABEL_11;
                }
              }
              v60 = *(_QWORD **)(v57 + 48);
              v20 = v19 + 96;
              if ( v19 + 96 != *v60 || v59 != v60[1] )
                goto LABEL_11;
              goto LABEL_29;
            }
            if ( v50 == 13 )
            {
              v93 = *(_QWORD *)(*(_QWORD *)(v47 + 48) + 8LL);
              v108 = v47;
              v51 = sub_2BF04A0(v46);
              if ( !v51 )
                goto LABEL_11;
              v77 = *(_BYTE *)(v51 + 8);
              v19 = v108;
              v53 = v93;
              if ( v77 != 23 )
              {
                if ( v77 == 9 )
                {
                  if ( **(_BYTE **)(v51 + 136) != 42 )
                    goto LABEL_11;
                  goto LABEL_64;
                }
                if ( v77 != 16 )
                {
                  if ( v77 != 4 || *(_BYTE *)(v51 + 160) != 13 )
                    goto LABEL_11;
                  goto LABEL_64;
                }
              }
              if ( *(_DWORD *)(v51 + 160) != 13 )
                goto LABEL_11;
              v54 = *(_QWORD **)(v51 + 48);
              v20 = v108 + 96;
              if ( v108 + 96 != *v54 )
              {
LABEL_65:
                v55 = *(_DWORD *)(v51 + 56);
                if ( v20 != v54[v55 - 1] || v53 != v54[v55 - 2] )
                  goto LABEL_11;
                goto LABEL_29;
              }
            }
            else
            {
              if ( v50 != 14 )
                goto LABEL_85;
              v86 = *(_QWORD *)(*(_QWORD *)(v47 + 48) + 8LL);
              v101 = v47;
              v51 = sub_2BF04A0(v46);
              if ( !v51 )
                goto LABEL_11;
              v52 = *(_BYTE *)(v51 + 8);
              v19 = v101;
              v53 = v86;
              switch ( v52 )
              {
                case 23:
LABEL_63:
                  if ( *(_DWORD *)(v51 + 160) != 14 )
                    goto LABEL_11;
                  break;
                case 9:
                  if ( **(_BYTE **)(v51 + 136) != 43 )
                    goto LABEL_11;
                  break;
                case 16:
                  goto LABEL_63;
                default:
                  if ( v52 != 4 || *(_BYTE *)(v51 + 160) != 14 )
                    goto LABEL_11;
                  break;
              }
LABEL_64:
              v54 = *(_QWORD **)(v51 + 48);
              v20 = v19 + 96;
              if ( v19 + 96 != *v54 )
                goto LABEL_65;
            }
            if ( v53 == v54[1] )
              goto LABEL_29;
            goto LABEL_65;
          }
          v90 = *(_QWORD *)(*(_QWORD *)(v47 + 48) + 8LL);
          v104 = v47;
          v66 = sub_2BF04A0(v46);
          if ( !v66 )
            goto LABEL_11;
          v67 = *(_BYTE *)(v66 + 8);
          if ( v67 == 23 )
            break;
          if ( v67 == 9 )
          {
            if ( **(_BYTE **)(v66 + 136) != 44 )
              goto LABEL_11;
          }
          else
          {
            if ( v67 == 16 )
              break;
            if ( v67 != 4 || *(_BYTE *)(v66 + 160) != 15 )
              goto LABEL_11;
          }
LABEL_98:
          v68 = *(_QWORD *)(v66 + 48);
          v82 = v90;
          v91 = v104;
          if ( !*(_QWORD *)(v68 + 8) )
            goto LABEL_11;
          v105 = *(_QWORD *)(v68 + 8);
          v69 = sub_2BF04A0(v105);
          v70 = v82;
          if ( v69 )
            goto LABEL_11;
          v71 = v82;
          v83 = v105;
          v106 = v70;
          if ( sub_2BF04A0(v71) )
            goto LABEL_11;
          v72 = v91;
          if ( **(_BYTE **)(v83 + 40) != 17 )
            goto LABEL_11;
          v73 = *(_BYTE **)(v106 + 40);
          if ( *v73 != 17 )
            goto LABEL_11;
          v92 = *(_QWORD *)(v83 + 40);
          v107 = v72;
          sub_9865C0((__int64)&v123, (__int64)(v73 + 24));
          sub_C47170((__int64)&v123, 0xFFFFFFFFFFFFFFFFLL);
          v74 = v124;
          LODWORD(v124) = 0;
          v75 = v107;
          v127 = v74;
          v126 = v123;
          if ( *(_DWORD *)(v92 + 32) <= 0x40u )
          {
            if ( v123 != *(const char **)(v92 + 24) )
            {
LABEL_105:
              sub_969240((__int64 *)&v126);
              sub_969240((__int64 *)&v123);
              goto LABEL_11;
            }
          }
          else
          {
            v76 = sub_C43C50(v92 + 24, (const void **)&v126);
            v75 = v107;
            if ( !v76 )
              goto LABEL_105;
          }
          v109 = v75;
          sub_969240((__int64 *)&v126);
          sub_969240((__int64 *)&v123);
          v19 = v109;
          v20 = v109 + 96;
LABEL_29:
          v21 = *(_DWORD *)(a2 + 24);
          v22 = *(_QWORD *)(a2 + 8);
          if ( !v21 )
            goto LABEL_42;
          v23 = v21 - 1;
          v24 = v23 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v25 = (__int64 *)(v22 + 16LL * v24);
          v26 = *v25;
          if ( *v25 != v20 )
          {
            v36 = 1;
            while ( v26 != -4096 )
            {
              v56 = v36 + 1;
              v24 = v23 & (v36 + v24);
              v25 = (__int64 *)(v22 + 16LL * v24);
              v26 = *v25;
              if ( *v25 == v20 )
                goto LABEL_31;
              v36 = v56;
            }
LABEL_42:
            if ( v116 != v20 )
              goto LABEL_11;
            v27 = 0;
            goto LABEL_44;
          }
LABEL_31:
          v27 = (const char *)v25[1];
          if ( v116 != v20 )
          {
            if ( v27 )
              goto LABEL_33;
            goto LABEL_11;
          }
LABEL_44:
          v84 = v27;
          v100 = v19;
          v37 = sub_2BF0A50(v11);
          v38 = *(_QWORD *)(v37 + 80);
          v120[1] = v37 + 24;
          v39 = *(_QWORD *)(v100 + 48);
          v120[0] = v38;
          v40 = *(_QWORD *)(v39 + 8);
          v41 = *(_BYTE *)(sub_2BFD6A0((__int64)&v130, v20) + 8);
          if ( v41 == 12 )
          {
            v123 = v84;
            v126 = "ind.escape";
            v129 = 259;
            v124 = v40;
            if ( sub_2AB07D0(v120, 15, (__int64 *)&v123, 2, 0, (void **)&v126) )
              goto LABEL_33;
          }
          else if ( v41 == 14 )
          {
            v61 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(v40 + 40) + 8LL), 0, 0);
            v62 = sub_2AC42A0(a1, v61);
            v122 = v40;
            v126 = "ind.escape";
            v125 = 257;
            v129 = 259;
            v118 = 0;
            v117 = 0;
            v121 = v62;
            v63 = sub_2C27AE0(v120, 15, &v121, 2, v119, 0, &v117, (void **)&v123);
            if ( v63 )
              v63 += 12;
            if ( sub_2C286A0(v120, (__int64)v84, (__int64)v63, &v118, (void **)&v126) )
            {
              sub_9C6650(&v117);
              sub_9C6650(&v118);
              goto LABEL_33;
            }
            sub_9C6650(&v117);
            sub_9C6650(&v118);
          }
          else
          {
            if ( v41 > 3u && v41 != 5 && (v41 & 0xFD) != 4 )
              BUG();
            v42 = *(_QWORD *)(v100 + 152);
            v119 = 0;
            v129 = 257;
            v111 = *(_BYTE **)(v42 + 40);
            v43 = sub_B45210((__int64)v111);
            v124 = v40;
            LODWORD(v121) = v43;
            v123 = v84;
            if ( sub_2C27AE0(v120, 2 * (*v111 == 43) + 14, (__int64 *)&v123, 2, v43, 1, &v119, (void **)&v126) )
            {
              sub_9C6650(&v119);
LABEL_33:
              v28 = *(_QWORD *)(*(_QWORD *)(v113 + 24) + 8LL * v8);
              v126 = (const char *)(v113 + 16);
              v29 = *(_QWORD **)(v28 + 16);
              v30 = (__int64)&v29[*(unsigned int *)(v28 + 24)];
              v31 = sub_2C25810(v29, v30, (__int64 *)&v126);
              if ( (_QWORD *)v30 != v31 )
              {
                if ( (_QWORD *)v30 != v31 + 1 )
                {
                  v110 = v34;
                  memmove(v31, v31 + 1, v30 - (_QWORD)(v31 + 1));
                  LODWORD(v32) = *(_DWORD *)(v28 + 24);
                  v34 = v110;
                }
                v32 = (unsigned int)(v32 - 1);
                *(_DWORD *)(v28 + 24) = v32;
                v33 = (_QWORD *)(*(_QWORD *)(v113 + 24) + 8LL * v8);
              }
              *v33 = v34;
              v35 = *(unsigned int *)(v34 + 24);
              if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v34 + 28) )
              {
                v112 = v34;
                sub_C8D5F0(v34 + 16, (const void *)(v34 + 32), v35 + 1, 8u, v32, (__int64)v33);
                v34 = v112;
                v35 = *(unsigned int *)(v112 + 24);
              }
              *(_QWORD *)(*(_QWORD *)(v34 + 16) + 8 * v35) = v113 + 16;
              ++*(_DWORD *)(v34 + 24);
              goto LABEL_11;
            }
            sub_9C6650(&v119);
          }
LABEL_11:
          ++v8;
          if ( v115 == ++v10 )
            goto LABEL_17;
        }
        if ( *(_DWORD *)(v66 + 160) != 15 )
          goto LABEL_11;
        goto LABEL_98;
      }
LABEL_17:
      v113 = *(_QWORD *)(v113 + 8);
    }
    while ( v95 != v113 );
LABEL_18:
    ++v80;
  }
  while ( v79 != v80 );
  v14 = v131;
  v15 = 16LL * (unsigned int)v133;
  return sub_C7D6A0(v14, v15, 8);
}
