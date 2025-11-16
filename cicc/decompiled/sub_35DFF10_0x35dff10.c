// Function: sub_35DFF10
// Address: 0x35dff10
//
void __fastcall sub_35DFF10(_QWORD *a1)
{
  _QWORD *v1; // r14
  __int64 v2; // rdx
  __int64 v3; // rdx
  unsigned __int8 *v4; // rbx
  unsigned __int8 v5; // al
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // esi
  unsigned __int64 v16; // r13
  __int64 v17; // r8
  int v18; // r11d
  unsigned int v19; // edi
  unsigned __int8 **v20; // rdx
  unsigned __int8 **v21; // rax
  unsigned __int8 *v22; // rcx
  unsigned __int8 *v23; // rdx
  __int64 v24; // rdi
  __int64 **v25; // r15
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  int v30; // eax
  int v31; // edx
  unsigned int v32; // eax
  __int64 v33; // rsi
  int v34; // edx
  int v35; // ecx
  int v36; // edx
  int v37; // edi
  _QWORD *v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r10
  unsigned __int64 v42; // rdi
  __int64 (__fastcall *v43)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v44; // rax
  _QWORD *v45; // r9
  __int64 v46; // rdi
  _QWORD *v47; // rax
  __int64 v48; // rax
  unsigned __int8 *v49; // r12
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned int v52; // esi
  __int64 v53; // r8
  int v54; // r13d
  __int64 v55; // rcx
  unsigned __int8 **v56; // rdx
  unsigned __int8 **v57; // rax
  unsigned __int8 *v58; // rdi
  __int64 ***v59; // rdx
  _QWORD *v60; // r12
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 *v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // rbx
  __int64 v73; // r15
  __int64 v74; // r13
  __int64 v75; // rdx
  unsigned int v76; // esi
  int v77; // ecx
  int v78; // ecx
  __int64 v79; // r8
  __int64 v80; // rdi
  unsigned __int8 *v81; // rsi
  int v82; // r10d
  unsigned __int8 **v83; // r9
  __int64 v84; // rax
  int v85; // ecx
  int v86; // ecx
  __int64 v87; // r8
  int v88; // r10d
  __int64 v89; // rdi
  unsigned __int8 *v90; // rsi
  __int64 v91; // r12
  __int64 v92; // r13
  _QWORD *v93; // r15
  unsigned __int8 *v94; // rsi
  _BYTE *v95; // rax
  unsigned __int8 *v96; // rdx
  unsigned __int8 *v97; // r14
  __int64 v98; // rdx
  __int64 v99; // rdx
  unsigned int v100; // esi
  __int64 v101; // r9
  int v102; // r11d
  unsigned int v103; // r8d
  unsigned __int8 **v104; // rdx
  unsigned __int8 **v105; // rax
  unsigned __int8 *v106; // rdi
  unsigned __int8 *v107; // rdx
  __int64 **v108; // rdx
  int v109; // ecx
  int v110; // edx
  int v111; // esi
  int v112; // esi
  __int64 v113; // r11
  __int64 v114; // r8
  unsigned __int8 *v115; // rdi
  int v116; // r10d
  unsigned __int8 **v117; // r9
  int v118; // esi
  int v119; // esi
  __int64 v120; // r11
  int v121; // r10d
  __int64 v122; // r8
  unsigned __int8 *v123; // rdi
  int v124; // ecx
  int v125; // edx
  int v126; // r11d
  int v127; // r11d
  __int64 v128; // r9
  unsigned int v129; // ecx
  unsigned __int8 *v130; // r8
  int v131; // edi
  unsigned __int8 **v132; // rsi
  int v133; // r10d
  int v134; // r10d
  __int64 v135; // r8
  unsigned __int8 **v136; // rcx
  unsigned int v137; // r12d
  int v138; // esi
  unsigned __int8 *v139; // rdi
  __int64 v140; // [rsp+18h] [rbp-178h]
  __int64 v141; // [rsp+20h] [rbp-170h]
  __int64 *v142; // [rsp+28h] [rbp-168h]
  unsigned __int8 **v143; // [rsp+30h] [rbp-160h]
  unsigned int v144; // [rsp+38h] [rbp-158h]
  __int64 v145; // [rsp+38h] [rbp-158h]
  unsigned __int8 **v146; // [rsp+40h] [rbp-150h]
  __int64 v147; // [rsp+48h] [rbp-148h]
  _QWORD *v148; // [rsp+48h] [rbp-148h]
  _QWORD *v149; // [rsp+48h] [rbp-148h]
  __int64 *v150; // [rsp+48h] [rbp-148h]
  _QWORD *v151; // [rsp+48h] [rbp-148h]
  unsigned __int8 *v152; // [rsp+48h] [rbp-148h]
  __int64 v153; // [rsp+48h] [rbp-148h]
  unsigned int v154; // [rsp+48h] [rbp-148h]
  __int64 v155; // [rsp+50h] [rbp-140h]
  unsigned int v156; // [rsp+58h] [rbp-138h]
  _BYTE *v157; // [rsp+58h] [rbp-138h]
  _QWORD *v158; // [rsp+60h] [rbp-130h] BYREF
  unsigned __int64 *v159; // [rsp+68h] [rbp-128h]
  char v160[32]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v161; // [rsp+90h] [rbp-100h]
  char v162[32]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v163; // [rsp+C0h] [rbp-D0h]
  unsigned __int64 v164[2]; // [rsp+D0h] [rbp-C0h] BYREF
  _BYTE v165[32]; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v166; // [rsp+100h] [rbp-90h]
  __int64 v167; // [rsp+108h] [rbp-88h]
  __int16 v168; // [rsp+110h] [rbp-80h]
  __int64 v169; // [rsp+118h] [rbp-78h]
  void **v170; // [rsp+120h] [rbp-70h]
  void **v171; // [rsp+128h] [rbp-68h]
  __int64 v172; // [rsp+130h] [rbp-60h]
  int v173; // [rsp+138h] [rbp-58h]
  __int16 v174; // [rsp+13Ch] [rbp-54h]
  char v175; // [rsp+13Eh] [rbp-52h]
  __int64 v176; // [rsp+140h] [rbp-50h]
  __int64 v177; // [rsp+148h] [rbp-48h]
  void *v178; // [rsp+150h] [rbp-40h] BYREF
  void *v179; // [rsp+158h] [rbp-38h] BYREF

  v1 = a1;
  v2 = *a1;
  v164[0] = (unsigned __int64)v165;
  v164[1] = 0x200000000LL;
  v170 = &v178;
  v171 = &v179;
  v168 = 0;
  v169 = v2;
  v172 = 0;
  v178 = &unk_49DA100;
  v173 = 0;
  v174 = 512;
  v179 = &unk_49DA0B0;
  v3 = a1[4];
  v175 = 7;
  v176 = 0;
  v177 = 0;
  v166 = 0;
  v167 = 0;
  v158 = a1;
  v159 = v164;
  v146 = *(unsigned __int8 ***)(v3 + 32);
  v143 = &v146[*(unsigned int *)(v3 + 40)];
  if ( v143 != v146 )
  {
    while ( 1 )
    {
      v4 = *v146;
      v5 = **v146;
      if ( v5 == 85 )
      {
        v156 = 0;
        v144 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
LABEL_4:
        v6 = 0;
LABEL_5:
        v7 = 32 * v6;
        if ( (v4[7] & 0x80u) != 0 )
        {
LABEL_6:
          v8 = sub_BD2BC0((__int64)v4);
          v10 = v8 + v9;
          if ( (v4[7] & 0x80u) == 0 )
          {
            if ( !(unsigned int)(v10 >> 4) )
              goto LABEL_28;
          }
          else
          {
            if ( !(unsigned int)((v10 - sub_BD2BC0((__int64)v4)) >> 4) )
              goto LABEL_28;
            if ( (v4[7] & 0x80u) != 0 )
            {
              v11 = *(_DWORD *)(sub_BD2BC0((__int64)v4) + 8);
              if ( (v4[7] & 0x80u) == 0 )
                BUG();
              v12 = sub_BD2BC0((__int64)v4);
              v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
              goto LABEL_11;
            }
          }
          BUG();
        }
        while ( 1 )
        {
LABEL_28:
          v14 = 0;
LABEL_11:
          if ( v156 >= (unsigned int)((32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF) - 32 - v7 - v14) >> 5) )
            goto LABEL_78;
          v15 = *((_DWORD *)v1 + 46);
          v16 = *(_QWORD *)&v4[32 * (v156 - (unsigned __int64)(*((_DWORD *)v4 + 1) & 0x7FFFFFF))];
          if ( v15 )
          {
            v17 = v1[21];
            v18 = 1;
            v19 = (v15 - 1) & v144;
            v20 = (unsigned __int8 **)(v17 + 56LL * v19);
            v21 = 0;
            v22 = *v20;
            if ( v4 == *v20 )
            {
LABEL_14:
              v23 = v20[1];
              goto LABEL_15;
            }
            while ( v22 != (unsigned __int8 *)-4096LL )
            {
              if ( v22 == (unsigned __int8 *)-8192LL && !v21 )
                v21 = v20;
              v19 = (v15 - 1) & (v18 + v19);
              v20 = (unsigned __int8 **)(v17 + 56LL * v19);
              v22 = *v20;
              if ( v4 == *v20 )
                goto LABEL_14;
              ++v18;
            }
            v35 = *((_DWORD *)v1 + 44);
            if ( !v21 )
              v21 = v20;
            ++v1[20];
            v36 = v35 + 1;
            if ( 4 * (v35 + 1) < 3 * v15 )
            {
              if ( v15 - *((_DWORD *)v1 + 45) - v36 > v15 >> 3 )
                goto LABEL_39;
              sub_35DF930((__int64)(v1 + 20), v15);
              v85 = *((_DWORD *)v1 + 46);
              if ( !v85 )
              {
LABEL_217:
                ++*((_DWORD *)v1 + 44);
                BUG();
              }
              v86 = v85 - 1;
              v87 = v1[21];
              v83 = 0;
              v88 = 1;
              LODWORD(v89) = v86 & v144;
              v21 = (unsigned __int8 **)(v87 + 56LL * (v86 & v144));
              v90 = *v21;
              v36 = *((_DWORD *)v1 + 44) + 1;
              if ( v4 == *v21 )
                goto LABEL_39;
              while ( v90 != (unsigned __int8 *)-4096LL )
              {
                if ( !v83 && v90 == (unsigned __int8 *)-8192LL )
                  v83 = v21;
                v89 = v86 & (unsigned int)(v89 + v88);
                v21 = (unsigned __int8 **)(v87 + 56 * v89);
                v90 = *v21;
                if ( v4 == *v21 )
                  goto LABEL_39;
                ++v88;
              }
              goto LABEL_105;
            }
          }
          else
          {
            ++v1[20];
          }
          sub_35DF930((__int64)(v1 + 20), 2 * v15);
          v77 = *((_DWORD *)v1 + 46);
          if ( !v77 )
            goto LABEL_217;
          v78 = v77 - 1;
          v79 = v1[21];
          LODWORD(v80) = v78 & v144;
          v21 = (unsigned __int8 **)(v79 + 56LL * (v78 & v144));
          v81 = *v21;
          v36 = *((_DWORD *)v1 + 44) + 1;
          if ( v4 == *v21 )
            goto LABEL_39;
          v82 = 1;
          v83 = 0;
          while ( v81 != (unsigned __int8 *)-4096LL )
          {
            if ( v81 == (unsigned __int8 *)-8192LL && !v83 )
              v83 = v21;
            v80 = v78 & (unsigned int)(v80 + v82);
            v21 = (unsigned __int8 **)(v79 + 56 * v80);
            v81 = *v21;
            if ( v4 == *v21 )
              goto LABEL_39;
            ++v82;
          }
LABEL_105:
          if ( v83 )
            v21 = v83;
LABEL_39:
          *((_DWORD *)v1 + 44) = v36;
          if ( *v21 != (unsigned __int8 *)-4096LL )
            --*((_DWORD *)v1 + 45);
          v23 = (unsigned __int8 *)(v21 + 3);
          *v21 = v4;
          v21[1] = (unsigned __int8 *)(v21 + 3);
          v21[2] = (unsigned __int8 *)0x400000000LL;
LABEL_15:
          if ( *(_BYTE *)v16 <= 0x1Cu || *(_BYTE *)(*(_QWORD *)(v16 + 8) + 8LL) != 12 )
            goto LABEL_24;
          v24 = (__int64)v158;
          v25 = *(__int64 ***)&v23[8 * v156];
          if ( *((_BYTE *)v158 + 220) )
          {
            v26 = (_QWORD *)v158[25];
            v27 = &v26[*((unsigned int *)v158 + 53)];
            if ( v26 != v27 )
            {
              while ( v16 != *v26 )
              {
                if ( v27 == ++v26 )
                  goto LABEL_83;
              }
              goto LABEL_22;
            }
          }
          else
          {
            v65 = sub_C8CA60((__int64)(v158 + 24), v16);
            v24 = (__int64)v158;
            if ( v65 )
              goto LABEL_22;
          }
LABEL_83:
          if ( *(_BYTE *)(v24 + 92) )
          {
            v66 = *(_QWORD **)(v24 + 72);
            v67 = &v66[*(unsigned int *)(v24 + 84)];
            if ( v66 == v67 )
              goto LABEL_24;
            while ( v16 != *v66 )
            {
              if ( v67 == ++v66 )
                goto LABEL_24;
            }
          }
          else
          {
            if ( !sub_C8CA60(v24 + 64, v16) )
              goto LABEL_24;
            v24 = (__int64)v158;
          }
LABEL_22:
          v28 = *(_QWORD *)(v24 + 24);
          v29 = *(_QWORD *)(v28 + 8);
          v30 = *(_DWORD *)(v28 + 24);
          if ( v30 )
          {
            v31 = v30 - 1;
            v32 = (v30 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v33 = *(_QWORD *)(v29 + 8LL * v32);
            if ( v16 == v33 )
              goto LABEL_24;
            v37 = 1;
            while ( v33 != -4096 )
            {
              v32 = v31 & (v37 + v32);
              v33 = *(_QWORD *)(v29 + 8LL * v32);
              if ( v16 == v33 )
                goto LABEL_24;
              ++v37;
            }
          }
          sub_D5F1F0((__int64)v159, v16);
          v41 = (__int64)v159;
          v161 = 257;
          if ( v25 == *(__int64 ***)(v16 + 8) )
          {
            v45 = (_QWORD *)v16;
          }
          else
          {
            v42 = v159[10];
            v43 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v42 + 120LL);
            if ( v43 == sub_920130 )
            {
              if ( *(_BYTE *)v16 > 0x15u )
                goto LABEL_90;
              v147 = (__int64)v159;
              if ( (unsigned __int8)sub_AC4810(0x26u) )
                v44 = sub_ADAB70(38, v16, v25, 0);
              else
                v44 = sub_AA93C0(0x26u, v16, (__int64)v25);
              v41 = v147;
              v45 = (_QWORD *)v44;
            }
            else
            {
              v153 = (__int64)v159;
              v84 = v43(v42, 38u, (_BYTE *)v16, (__int64)v25);
              v41 = v153;
              v45 = (_QWORD *)v84;
            }
            if ( !v45 )
            {
LABEL_90:
              v163 = 257;
              v150 = (__int64 *)v41;
              v68 = sub_B51D30(38, v16, (__int64)v25, (__int64)v162, 0, 0);
              v69 = v150[11];
              v142 = v150;
              v70 = v150[7];
              v71 = v150[8];
              v151 = (_QWORD *)v68;
              (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v69 + 16LL))(
                v69,
                v68,
                v160,
                v70,
                v71);
              v45 = v151;
              if ( *v142 != *v142 + 16LL * *((unsigned int *)v142 + 2) )
              {
                v152 = v4;
                v72 = *v142;
                v73 = *v142 + 16LL * *((unsigned int *)v142 + 2);
                v74 = (__int64)v45;
                do
                {
                  v75 = *(_QWORD *)(v72 + 8);
                  v76 = *(_DWORD *)v72;
                  v72 += 16;
                  sub_B99FD0(v74, v76, v75);
                }
                while ( v73 != v72 );
                v4 = v152;
                v45 = (_QWORD *)v74;
              }
            }
          }
          if ( *(_BYTE *)v45 <= 0x1Cu )
            goto LABEL_24;
          v46 = (__int64)v158;
          if ( !*((_BYTE *)v158 + 92) )
            goto LABEL_89;
          v47 = (_QWORD *)v158[9];
          v39 = *((unsigned int *)v158 + 21);
          v38 = &v47[v39];
          if ( v47 != v38 )
          {
            while ( v45 != (_QWORD *)*v47 )
            {
              if ( v38 == ++v47 )
                goto LABEL_94;
            }
            goto LABEL_59;
          }
LABEL_94:
          if ( (unsigned int)v39 < *((_DWORD *)v158 + 20) )
          {
            *((_DWORD *)v158 + 21) = v39 + 1;
            *v38 = v45;
            ++*(_QWORD *)(v46 + 64);
          }
          else
          {
LABEL_89:
            v149 = v45;
            sub_C8CC70((__int64)(v158 + 8), (__int64)v45, (__int64)v38, v39, v40, (__int64)v45);
            v45 = v149;
          }
LABEL_59:
          v48 = v155;
          v148 = v45;
          LOWORD(v48) = 0;
          v155 = v48;
          sub_B444E0(v45, (__int64)(v4 + 24), v48);
          v49 = &v4[32 * (v156 - (unsigned __int64)(*((_DWORD *)v4 + 1) & 0x7FFFFFF))];
          if ( *(_QWORD *)v49 )
          {
            v50 = *((_QWORD *)v49 + 1);
            **((_QWORD **)v49 + 2) = v50;
            if ( v50 )
              *(_QWORD *)(v50 + 16) = *((_QWORD *)v49 + 2);
          }
          *(_QWORD *)v49 = v148;
          v51 = v148[2];
          *((_QWORD *)v49 + 1) = v51;
          if ( v51 )
            *(_QWORD *)(v51 + 16) = v49 + 8;
          *((_QWORD *)v49 + 2) = v148 + 2;
          v148[2] = v49;
LABEL_24:
          v34 = *v4;
          ++v156;
          if ( v34 == 40 )
          {
            v6 = (unsigned int)sub_B491D0((__int64)v4);
            goto LABEL_5;
          }
          if ( v34 == 85 )
            goto LABEL_4;
          if ( v34 != 34 )
            BUG();
          v7 = 64;
          if ( (v4[7] & 0x80u) != 0 )
            goto LABEL_6;
        }
      }
      if ( v5 == 32 )
      {
        v52 = *((_DWORD *)v1 + 46);
        if ( v52 )
        {
          v53 = v1[21];
          v54 = 1;
          LODWORD(v55) = (v52 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v56 = (unsigned __int8 **)(v53 + 56LL * (unsigned int)v55);
          v57 = 0;
          v58 = *v56;
          if ( v4 == *v56 )
          {
LABEL_70:
            v59 = (__int64 ***)v56[1];
LABEL_71:
            v60 = sub_35DEB20((__int64 *)&v158, **((_QWORD **)v4 - 1), *v59);
            if ( v60 )
            {
              v61 = v141;
              LOWORD(v61) = 0;
              v141 = v61;
              sub_B444E0(v60, (__int64)(v4 + 24), v61);
              v62 = *((_QWORD *)v4 - 1);
              if ( *(_QWORD *)v62 )
              {
                v63 = *(_QWORD *)(v62 + 8);
                **(_QWORD **)(v62 + 16) = v63;
                if ( v63 )
                  *(_QWORD *)(v63 + 16) = *(_QWORD *)(v62 + 16);
              }
              *(_QWORD *)v62 = v60;
              v64 = v60[2];
              *(_QWORD *)(v62 + 8) = v64;
              if ( v64 )
                *(_QWORD *)(v64 + 16) = v62 + 8;
              *(_QWORD *)(v62 + 16) = v60 + 2;
              v60[2] = v62;
            }
            goto LABEL_78;
          }
          while ( v58 != (unsigned __int8 *)-4096LL )
          {
            if ( !v57 && v58 == (unsigned __int8 *)-8192LL )
              v57 = v56;
            v55 = (v52 - 1) & ((_DWORD)v55 + v54);
            v56 = (unsigned __int8 **)(v53 + 56 * v55);
            v58 = *v56;
            if ( v4 == *v56 )
              goto LABEL_70;
            ++v54;
          }
          v124 = *((_DWORD *)v1 + 44);
          if ( !v57 )
            v57 = v56;
          ++v1[20];
          v125 = v124 + 1;
          if ( 4 * (v124 + 1) < 3 * v52 )
          {
            if ( v52 - *((_DWORD *)v1 + 45) - v125 <= v52 >> 3 )
            {
              sub_35DF930((__int64)(v1 + 20), v52);
              v133 = *((_DWORD *)v1 + 46);
              if ( !v133 )
              {
LABEL_219:
                ++*((_DWORD *)v1 + 44);
                BUG();
              }
              v134 = v133 - 1;
              v135 = v1[21];
              v136 = 0;
              v137 = v134 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
              v125 = *((_DWORD *)v1 + 44) + 1;
              v138 = 1;
              v57 = (unsigned __int8 **)(v135 + 56LL * v137);
              v139 = *v57;
              if ( v4 != *v57 )
              {
                while ( v139 != (unsigned __int8 *)-4096LL )
                {
                  if ( v139 == (unsigned __int8 *)-8192LL && !v136 )
                    v136 = v57;
                  v137 = v134 & (v138 + v137);
                  v57 = (unsigned __int8 **)(v135 + 56LL * v137);
                  v139 = *v57;
                  if ( v4 == *v57 )
                    goto LABEL_180;
                  ++v138;
                }
                if ( v136 )
                  v57 = v136;
              }
            }
            goto LABEL_180;
          }
        }
        else
        {
          ++v1[20];
        }
        sub_35DF930((__int64)(v1 + 20), 2 * v52);
        v126 = *((_DWORD *)v1 + 46);
        if ( !v126 )
          goto LABEL_219;
        v127 = v126 - 1;
        v128 = v1[21];
        v129 = v127 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v125 = *((_DWORD *)v1 + 44) + 1;
        v57 = (unsigned __int8 **)(v128 + 56LL * v129);
        v130 = *v57;
        if ( v4 != *v57 )
        {
          v131 = 1;
          v132 = 0;
          while ( v130 != (unsigned __int8 *)-4096LL )
          {
            if ( !v132 && v130 == (unsigned __int8 *)-8192LL )
              v132 = v57;
            v129 = v127 & (v131 + v129);
            v57 = (unsigned __int8 **)(v128 + 56LL * v129);
            v130 = *v57;
            if ( v4 == *v57 )
              goto LABEL_180;
            ++v131;
          }
          if ( v132 )
            v57 = v132;
        }
LABEL_180:
        *((_DWORD *)v1 + 44) = v125;
        if ( *v57 != (unsigned __int8 *)-4096LL )
          --*((_DWORD *)v1 + 45);
        v59 = (__int64 ***)(v57 + 3);
        *v57 = v4;
        v57[1] = (unsigned __int8 *)(v57 + 3);
        v57[2] = (unsigned __int8 *)0x400000000LL;
        goto LABEL_71;
      }
      if ( (v5 != 68 || (unsigned int)sub_BCB060(*((_QWORD *)v4 + 1)) < *((_DWORD *)v1 + 2))
        && (*((_DWORD *)v4 + 1) & 0x7FFFFFF) != 0 )
      {
        break;
      }
LABEL_78:
      if ( v143 == ++v146 )
        goto LABEL_79;
    }
    v91 = v140;
    v92 = 0;
    v93 = v1;
    v154 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
    v145 = (__int64)(v1 + 20);
    while ( 1 )
    {
      v100 = *((_DWORD *)v93 + 46);
      if ( !v100 )
        break;
      v101 = v93[21];
      v102 = 1;
      v103 = (v100 - 1) & v154;
      v104 = (unsigned __int8 **)(v101 + 56LL * v103);
      v105 = 0;
      v106 = *v104;
      if ( v4 != *v104 )
      {
        while ( v106 != (unsigned __int8 *)-4096LL )
        {
          if ( v106 == (unsigned __int8 *)-8192LL && !v105 )
            v105 = v104;
          v103 = (v100 - 1) & (v102 + v103);
          v104 = (unsigned __int8 **)(v101 + 56LL * v103);
          v106 = *v104;
          if ( v4 == *v104 )
            goto LABEL_134;
          ++v102;
        }
        v109 = *((_DWORD *)v93 + 44);
        if ( !v105 )
          v105 = v104;
        ++v93[20];
        v110 = v109 + 1;
        if ( 4 * (v109 + 1) < 3 * v100 )
        {
          if ( v100 - *((_DWORD *)v93 + 45) - v110 <= v100 >> 3 )
          {
            sub_35DF930(v145, v100);
            v118 = *((_DWORD *)v93 + 46);
            if ( !v118 )
            {
LABEL_220:
              ++*((_DWORD *)v93 + 44);
              BUG();
            }
            v119 = v118 - 1;
            v117 = 0;
            v120 = v93[21];
            v121 = 1;
            LODWORD(v122) = v119 & v154;
            v110 = *((_DWORD *)v93 + 44) + 1;
            v105 = (unsigned __int8 **)(v120 + 56LL * (v119 & v154));
            v123 = *v105;
            if ( v4 != *v105 )
            {
              while ( v123 != (unsigned __int8 *)-4096LL )
              {
                if ( !v117 && v123 == (unsigned __int8 *)-8192LL )
                  v117 = v105;
                v122 = v119 & (unsigned int)(v122 + v121);
                v105 = (unsigned __int8 **)(v120 + 56 * v122);
                v123 = *v105;
                if ( v4 == *v105 )
                  goto LABEL_147;
                ++v121;
              }
              goto LABEL_157;
            }
          }
          goto LABEL_147;
        }
LABEL_153:
        sub_35DF930(v145, 2 * v100);
        v111 = *((_DWORD *)v93 + 46);
        if ( !v111 )
          goto LABEL_220;
        v112 = v111 - 1;
        v113 = v93[21];
        LODWORD(v114) = v112 & v154;
        v110 = *((_DWORD *)v93 + 44) + 1;
        v105 = (unsigned __int8 **)(v113 + 56LL * (v112 & v154));
        v115 = *v105;
        if ( v4 != *v105 )
        {
          v116 = 1;
          v117 = 0;
          while ( v115 != (unsigned __int8 *)-4096LL )
          {
            if ( !v117 && v115 == (unsigned __int8 *)-8192LL )
              v117 = v105;
            v114 = v112 & (unsigned int)(v114 + v116);
            v105 = (unsigned __int8 **)(v113 + 56 * v114);
            v115 = *v105;
            if ( v4 == *v105 )
              goto LABEL_147;
            ++v116;
          }
LABEL_157:
          if ( v117 )
            v105 = v117;
        }
LABEL_147:
        *((_DWORD *)v93 + 44) = v110;
        if ( *v105 != (unsigned __int8 *)-4096LL )
          --*((_DWORD *)v93 + 45);
        v107 = (unsigned __int8 *)(v105 + 3);
        *v105 = v4;
        v105[1] = (unsigned __int8 *)(v105 + 3);
        v105[2] = (unsigned __int8 *)0x400000000LL;
        goto LABEL_135;
      }
LABEL_134:
      v107 = v104[1];
LABEL_135:
      v108 = *(__int64 ***)&v107[8 * v92];
      if ( (v4[7] & 0x40) != 0 )
        v94 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
      else
        v94 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
      v95 = sub_35DEB20((__int64 *)&v158, *(_QWORD *)&v94[32 * v92], v108);
      if ( v95 )
      {
        LOWORD(v91) = 0;
        v157 = v95;
        sub_B444E0(v95, (__int64)(v4 + 24), v91);
        if ( (v4[7] & 0x40) != 0 )
          v96 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v96 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v97 = &v96[32 * v92];
        if ( *(_QWORD *)v97 )
        {
          v98 = *((_QWORD *)v97 + 1);
          **((_QWORD **)v97 + 2) = v98;
          if ( v98 )
            *(_QWORD *)(v98 + 16) = *((_QWORD *)v97 + 2);
        }
        *(_QWORD *)v97 = v157;
        v99 = *((_QWORD *)v157 + 2);
        *((_QWORD *)v97 + 1) = v99;
        if ( v99 )
          *(_QWORD *)(v99 + 16) = v97 + 8;
        *((_QWORD *)v97 + 2) = v157 + 16;
        *((_QWORD *)v157 + 2) = v97;
      }
      if ( (*((_DWORD *)v4 + 1) & 0x7FFFFFFu) <= (unsigned int)++v92 )
      {
        v140 = v91;
        v1 = v93;
        goto LABEL_78;
      }
    }
    ++v93[20];
    goto LABEL_153;
  }
LABEL_79:
  nullsub_61();
  v178 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v164[0] != v165 )
    _libc_free(v164[0]);
}
