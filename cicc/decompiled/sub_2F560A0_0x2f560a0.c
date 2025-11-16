// Function: sub_2F560A0
// Address: 0x2f560a0
//
__int64 __fastcall sub_2F560A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int v6; // ebx
  unsigned __int8 *v7; // r15
  __int64 v9; // rdi
  unsigned int v10; // ebx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  float v17; // xmm1_4
  int v18; // r14d
  int v19; // r14d
  unsigned int v20; // r13d
  _BYTE *v21; // r10
  unsigned int v22; // eax
  unsigned int v23; // esi
  double v24; // xmm0_8
  int v25; // edi
  unsigned int v26; // edx
  unsigned int v27; // edx
  __int64 v28; // rdx
  double v29; // xmm1_8
  float v30; // xmm1_4
  float v31; // xmm1_4
  int v32; // r11d
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // r13
  unsigned __int16 *v36; // rsi
  int v37; // r11d
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 (__fastcall *v42)(double); // rdx
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 (__fastcall *v45)(double); // rax
  __int64 v46; // rsi
  int v47; // edx
  unsigned int v48; // ecx
  __int64 v49; // rdi
  double v50; // xmm1_8
  unsigned int *v51; // rax
  unsigned int *v52; // rsi
  __int64 v53; // rcx
  __int64 v54; // rbx
  _QWORD **v55; // rcx
  _QWORD **v56; // rdx
  _QWORD **v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r8
  int v63; // edx
  _BYTE *v64; // rdi
  __int64 v65; // r14
  __int64 v66; // rbx
  __int64 v67; // r12
  __int64 v68; // rdx
  __int64 v69; // r13
  unsigned __int64 v70; // rcx
  __int64 v71; // r9
  unsigned int v72; // eax
  __int64 v73; // r15
  __int64 v74; // rdx
  unsigned __int64 v75; // rax
  __int64 v76; // r13
  unsigned int v77; // edx
  __int64 v78; // rdx
  __int64 v79; // rdx
  unsigned int *v80; // rcx
  unsigned int v81; // eax
  __int64 v82; // r15
  __int64 v83; // rcx
  _QWORD *v84; // r9
  __int64 *v85; // r8
  __int64 v86; // r13
  __int64 v87; // r8
  __int64 v88; // r14
  unsigned int v89; // r11d
  _QWORD *v90; // r9
  unsigned __int64 v91; // rcx
  unsigned __int64 v92; // rdx
  bool v93; // r10
  __int64 v94; // rdx
  unsigned int v95; // eax
  __int64 v96; // rdx
  __int64 *v97; // r15
  __int64 v98; // rax
  __int64 *v99; // rax
  unsigned __int64 v100; // rdx
  int v101; // r15d
  __int64 v102; // r9
  unsigned __int64 v103; // r10
  _DWORD *v104; // rax
  unsigned __int64 v105; // rdx
  __int64 v106; // r10
  unsigned __int64 v107; // rax
  _QWORD *v108; // rcx
  _QWORD *v109; // rsi
  unsigned __int8 *v110; // [rsp+18h] [rbp-1F8h]
  unsigned int v111; // [rsp+24h] [rbp-1ECh]
  unsigned int v112; // [rsp+28h] [rbp-1E8h]
  unsigned int v113; // [rsp+30h] [rbp-1E0h]
  int v114; // [rsp+34h] [rbp-1DCh]
  int v115; // [rsp+34h] [rbp-1DCh]
  int v117; // [rsp+40h] [rbp-1D0h]
  float v118; // [rsp+44h] [rbp-1CCh]
  bool v119; // [rsp+44h] [rbp-1CCh]
  float v121; // [rsp+50h] [rbp-1C0h]
  unsigned int v122; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v123; // [rsp+50h] [rbp-1C0h]
  _QWORD *v124; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 v125; // [rsp+58h] [rbp-1B8h]
  __int64 v126; // [rsp+58h] [rbp-1B8h]
  const void *v128; // [rsp+60h] [rbp-1B0h]
  _QWORD *v129; // [rsp+68h] [rbp-1A8h]
  float v130; // [rsp+70h] [rbp-1A0h]
  __int64 v131; // [rsp+70h] [rbp-1A0h]
  int v132; // [rsp+70h] [rbp-1A0h]
  int v133; // [rsp+70h] [rbp-1A0h]
  int v134; // [rsp+78h] [rbp-198h]
  unsigned int *v135; // [rsp+80h] [rbp-190h] BYREF
  __int64 v136; // [rsp+88h] [rbp-188h]
  _BYTE v137[32]; // [rsp+90h] [rbp-180h] BYREF
  _QWORD v138[2]; // [rsp+B0h] [rbp-160h] BYREF
  _BYTE v139[32]; // [rsp+C0h] [rbp-150h] BYREF
  _BYTE *v140; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v141; // [rsp+E8h] [rbp-128h]
  _BYTE v142[32]; // [rsp+F0h] [rbp-120h] BYREF
  _QWORD v143[2]; // [rsp+110h] [rbp-100h] BYREF
  __int64 v144; // [rsp+120h] [rbp-F0h]
  __int64 v145; // [rsp+128h] [rbp-E8h]
  __int64 v146; // [rsp+130h] [rbp-E0h]
  __int64 v147; // [rsp+138h] [rbp-D8h]
  __int64 v148; // [rsp+140h] [rbp-D0h]
  __int64 v149; // [rsp+148h] [rbp-C8h]
  unsigned int v150; // [rsp+150h] [rbp-C0h]
  char v151; // [rsp+154h] [rbp-BCh]
  __int64 v152; // [rsp+158h] [rbp-B8h]
  __int64 v153; // [rsp+160h] [rbp-B0h]
  char *v154; // [rsp+168h] [rbp-A8h]
  __int64 v155; // [rsp+170h] [rbp-A0h]
  int v156; // [rsp+178h] [rbp-98h]
  char v157; // [rsp+17Ch] [rbp-94h]
  char v158; // [rsp+180h] [rbp-90h] BYREF
  __int64 v159; // [rsp+1A0h] [rbp-70h]
  char *v160; // [rsp+1A8h] [rbp-68h]
  __int64 v161; // [rsp+1B0h] [rbp-60h]
  int v162; // [rsp+1B8h] [rbp-58h]
  char v163; // [rsp+1BCh] [rbp-54h]
  char v164; // [rsp+1C0h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a1 + 992);
  v117 = *(_DWORD *)(v4 + 288);
  if ( v117 == 1 )
  {
    v6 = *(_DWORD *)(v4 + 208);
    if ( v6 > 2 )
    {
      v7 = *(unsigned __int8 **)(v4 + 280);
      v9 = *(_QWORD *)(a1 + 40);
      v10 = v6 - 1;
      v124 = *(_QWORD **)(v4 + 200);
      v135 = (unsigned int *)v137;
      v110 = v7;
      v136 = 0x800000000LL;
      if ( !sub_2E21220(v9, a2, 0) )
      {
        v11 = 8LL * *(unsigned int *)(*(_QWORD *)v7 + 24LL);
        goto LABEL_6;
      }
      v79 = *(_QWORD *)(a1 + 32);
      v11 = 8LL * *(unsigned int *)(*(_QWORD *)v7 + 24LL);
      v80 = (unsigned int *)(v11 + *(_QWORD *)(v79 + 344));
      v81 = v80[1];
      v82 = *(_QWORD *)(v79 + 184) + 8LL * *v80;
      v83 = v81;
      if ( v81 )
      {
        v84 = (_QWORD *)v82;
        do
        {
          v85 = &v84[v83 >> 1];
          if ( (*(_DWORD *)((*v85 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v85 >> 1) & 3) < (*(_DWORD *)((*v124 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | 2u) )
          {
            v84 = v85 + 1;
            v83 = v83 - (v83 >> 1) - 1;
          }
          else
          {
            v83 >>= 1;
          }
        }
        while ( v83 > 0 );
        v86 = ((__int64)v84 - v82) >> 3;
      }
      else
      {
        LODWORD(v86) = 0;
      }
      if ( v81 != (_DWORD)v86 )
      {
        LODWORD(v87) = 0;
        v88 = (unsigned int)v86;
        while ( 1 )
        {
          v89 = v87;
          v90 = &v124[(unsigned int)(v87 + 1)];
          v87 = (unsigned int)(v87 + 1);
          v91 = *(_QWORD *)(v82 + 8 * v88) & 0xFFFFFFFFFFFFFFF8LL;
          v92 = *v90 & 0xFFFFFFFFFFFFFFF8LL;
          v93 = (_DWORD)v87 == v10;
          if ( *(_DWORD *)(v91 + 24) <= *(_DWORD *)(v92 + 24) )
          {
            if ( v91 == v92 && (_DWORD)v87 == v10 )
            {
LABEL_118:
              v11 = 8LL * *(unsigned int *)(*(_QWORD *)v110 + 24LL);
              break;
            }
            v94 = (unsigned int)v136;
            if ( (unsigned __int64)(unsigned int)v136 + 1 > HIDWORD(v136) )
            {
              v112 = v81;
              v115 = v87;
              v119 = (_DWORD)v87 == v10;
              v122 = v89;
              v129 = v90;
              sub_C8D5F0((__int64)&v135, v137, (unsigned int)v136 + 1LL, 4u, v87, (__int64)v90);
              v94 = (unsigned int)v136;
              v81 = v112;
              LODWORD(v87) = v115;
              v93 = v119;
              v89 = v122;
              v90 = v129;
            }
            v135[v94] = v89;
            LODWORD(v136) = v136 + 1;
            while ( *(_DWORD *)((*(_QWORD *)(v82 + 8 * v88) & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((*v90 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
            {
              LODWORD(v86) = v86 + 1;
              if ( (_DWORD)v86 == v81 )
                goto LABEL_118;
              v88 = (unsigned int)v86;
            }
          }
          if ( v81 == (_DWORD)v86 || v93 )
            goto LABEL_118;
        }
      }
LABEL_6:
      v134 = *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
      v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 832) + 136LL) + v11);
      if ( v12 < 0 )
      {
        v78 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 832) + 136LL) + v11) & 1LL
            | (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 832) + 136LL) + v11) >> 1);
        v130 = (float)(int)v78 + (float)(int)v78;
      }
      else
      {
        v130 = (float)(int)v12;
      }
      v13 = sub_2E3A080(*(_QWORD *)(a1 + 792));
      if ( v13 < 0 )
        v17 = (float)(v13 & 1 | (unsigned int)((unsigned __int64)v13 >> 1))
            + (float)(v13 & 1 | (unsigned int)((unsigned __int64)v13 >> 1));
      else
        v17 = (float)(int)v13;
      v18 = *(_DWORD *)(a3 + 8);
      v138[0] = v139;
      v19 = -v18;
      v138[1] = 0x800000000LL;
      v114 = *(_DWORD *)(a3 + 72);
      v121 = (float)(1.0 / v17) * v130;
      if ( v114 == v19 )
      {
        v21 = v139;
      }
      else
      {
        v113 = v10;
        v118 = 0.0;
        v111 = 0;
        do
        {
          if ( v19 < 0 )
            v20 = *(unsigned __int16 *)(*(_QWORD *)a3 + 2 * (*(_QWORD *)(a3 + 8) + v19));
          else
            v20 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 56) + 2LL * v19);
          sub_2F558F0((_QWORD *)a1, v20, (__int64)v138, v14, v15, v16);
          if ( sub_2E21220(*(_QWORD *)(a1 + 40), a2, v20) )
          {
            v51 = v135;
            v52 = &v135[(unsigned int)v136];
            if ( v52 != v135 )
            {
              do
              {
                v53 = *v51++;
                *(_DWORD *)(v138[0] + 4 * v53) = unk_44D0BE0;
              }
              while ( v51 != v52 );
            }
          }
          v21 = (_BYTE *)v138[0];
          v22 = 1;
          v23 = 0;
          *(_QWORD *)&v24 = *(unsigned int *)v138[0];
LABEL_16:
          if ( v23 )
          {
LABEL_17:
            if ( v22 == v10 )
            {
              v15 = v110[33];
              v16 = 1;
              v14 = 1;
              v25 = v110[33];
            }
            else
            {
              v15 = 1;
              v16 = 1;
              v14 = 1;
              v25 = 1;
            }
            goto LABEL_19;
          }
          while ( 1 )
          {
            v14 = v110[32];
            if ( v22 == v10 )
            {
              v25 = v110[33];
              if ( (_BYTE)v25 )
              {
                v15 = (unsigned __int8)v14;
                v47 = 1;
              }
              else
              {
                if ( !(_BYTE)v14 )
                  break;
                v15 = 1;
                v47 = 0;
              }
              v26 = v15 + v10 + v47;
              if ( v10 <= v26 && v134 > 2 )
              {
                v27 = 1;
LABEL_56:
                if ( *(float *)(v138[0] + 4LL * v23) < *(float *)&v24
                  || (v15 = v23 + 2, *(_QWORD *)&v24 = *(unsigned int *)(v138[0] + 4LL * v27), v22 == (_DWORD)v15) )
                {
                  v23 = v27;
                }
                else
                {
                  v48 = v23 + 2;
                  do
                  {
                    v49 = v48++;
                    *(_QWORD *)&v50 = *(unsigned int *)(v138[0] + 4 * v49);
                    *(float *)&v50 = fmaxf(*(float *)&v50, *(float *)&v24);
                    v24 = v50;
                  }
                  while ( v22 != v48 );
                  v23 = v27;
                }
                goto LABEL_16;
              }
            }
            else
            {
              v16 = (unsigned __int8)v14;
              v15 = 1;
              v25 = 1;
LABEL_19:
              v26 = v15 + v16 + v22 - v23;
              if ( v134 > 2 && v10 <= v26 )
              {
LABEL_21:
                v27 = v23 + 1;
                if ( v22 <= v23 + 1 )
                {
                  ++v23;
                  v24 = 0.0;
                  goto LABEL_23;
                }
                goto LABEL_56;
              }
            }
            if ( *(float *)&v24 >= INFINITY )
              goto LABEL_21;
            v16 = v124[v23];
            v14 = (unsigned int)(16 * (v25 + v14));
            v15 = (*(_DWORD *)((v124[v22] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v124[v22] >> 1) & 3)
                - (*(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v16 >> 1) & 3);
            v30 = (float)((float)(int)(v26 + 1) * v121) / (float)(v15 + v14 + 400);
            if ( (float)(0.97998047 * v30) < *(float *)&v24 )
              goto LABEL_21;
            v31 = v30 - *(float *)&v24;
            if ( v31 > v118 )
            {
              v111 = v22;
              v113 = v23;
              v118 = v31 * 0.97998047;
            }
LABEL_23:
            if ( v22 >= v10 )
              break;
            v28 = v22++;
            *(_QWORD *)&v29 = *(unsigned int *)(v138[0] + 4 * v28);
            *(float *)&v29 = fmaxf(*(float *)&v29, *(float *)&v24);
            v24 = v29;
            if ( v23 )
              goto LABEL_17;
          }
          v32 = *(_DWORD *)(a3 + 72);
          v19 += v32 > v19;
          if ( v19 < v32 && v19 >= 0 )
          {
            v16 = v19;
            v33 = a3;
            v34 = *(_QWORD *)(a3 + 56);
            v35 = v33;
            do
            {
              v19 = v16;
              if ( (unsigned int)*(unsigned __int16 *)(v34 + 2 * v16) - 1 > 0x3FFFFFFE )
                break;
              LODWORD(v143[0]) = *(unsigned __int16 *)(v34 + 2 * v16);
              v36 = (unsigned __int16 *)(*(_QWORD *)v35 + 2LL * *(_QWORD *)(v35 + 8));
              if ( v36 == sub_2F4C810(*(unsigned __int16 **)v35, (__int64)v36, (int *)v143) )
                break;
              ++v16;
              ++v19;
            }
            while ( v37 > (int)v16 );
            a3 = v35;
          }
        }
        while ( v19 != v114 );
        if ( v113 != v10 )
        {
          v38 = *(_QWORD *)(a1 + 24);
          v39 = *(_QWORD *)(a1 + 32);
          v40 = *(_QWORD *)(a1 + 768);
          v143[0] = &unk_4A388F0;
          v143[1] = a2;
          v144 = a4;
          v41 = *(_QWORD *)(v40 + 32);
          v146 = v39;
          v145 = v41;
          v147 = v38;
          v42 = *(__int64 (__fastcall **)(double))(**(_QWORD **)(v40 + 16) + 128LL);
          v43 = 0;
          if ( (char *)v42 != (char *)sub_2DAC790 )
          {
            v43 = v42(v24);
            v41 = v145;
          }
          v148 = v43;
          v152 = a1 + 400;
          v44 = *(_DWORD *)(a4 + 8);
          v163 = 1;
          v149 = a1 + 760;
          v150 = v44;
          v154 = &v158;
          v151 = 0;
          v153 = 0;
          v155 = 4;
          v156 = 0;
          v157 = 1;
          v159 = 0;
          v160 = &v164;
          v161 = 4;
          v162 = 0;
          if ( !*(_BYTE *)(v41 + 36) )
            goto LABEL_84;
          v45 = *(__int64 (__fastcall **)(double))(v41 + 16);
          v46 = *(unsigned int *)(v41 + 28);
          v42 = (__int64 (__fastcall *)(double))((char *)v45 + 8 * v46);
          if ( v45 == v42 )
          {
LABEL_127:
            if ( (unsigned int)v46 < *(_DWORD *)(v41 + 24) )
            {
              *(_DWORD *)(v41 + 28) = v46 + 1;
              *(_QWORD *)v42 = v143;
              ++*(_QWORD *)(v41 + 8);
              goto LABEL_85;
            }
LABEL_84:
            sub_C8CC70(v41 + 8, (__int64)v143, (__int64)v42, v41, v15, v16);
            goto LABEL_85;
          }
          while ( *(_QWORD **)v45 != v143 )
          {
            v45 = (__int64 (__fastcall *)(double))((char *)v45 + 8);
            if ( v42 == v45 )
              goto LABEL_127;
          }
LABEL_85:
          sub_2FB3410(*(_QWORD *)(a1 + 1000), v143, 0);
          sub_2FB2500(*(_QWORD *)(a1 + 1000));
          v59 = sub_2FBA5C0(*(_QWORD *)(a1 + 1000), v124[v113]);
          v60 = sub_2FBA740(*(_QWORD *)(a1 + 1000), v124[v111]);
          sub_2FBD930(*(_QWORD *)(a1 + 1000), v59, v60);
          v61 = *(_QWORD *)(a1 + 1000);
          v140 = v142;
          v141 = 0x800000000LL;
          sub_2FBB760(v61, &v140);
          sub_2E01430(
            *(__int64 **)(a1 + 840),
            *(_DWORD *)(a2 + 112),
            (unsigned int *)(*(_QWORD *)v144 + 4LL * v150),
            *(unsigned int *)(v144 + 8) - (unsigned __int64)v150);
          v63 = 1;
          if ( !v113 )
            v63 = v110[32];
          if ( v111 == v10 )
            v117 = v110[33];
          v64 = v140;
          if ( v10 <= v117 + v63 + v111 - v113 )
          {
            v65 = (unsigned int)v141;
            if ( (_DWORD)v141 )
            {
              v66 = 0;
              v67 = a1;
              v128 = (const void *)(a1 + 936);
              while ( *(_DWORD *)&v64[4 * v66] != 1 )
              {
LABEL_92:
                if ( ++v66 == v65 )
                  goto LABEL_67;
              }
              v68 = v150 + (unsigned int)v66;
              v69 = *(_QWORD *)(v67 + 32);
              v70 = *(unsigned int *)(v69 + 160);
              v71 = *(unsigned int *)(*(_QWORD *)v144 + 4 * v68);
              v72 = *(_DWORD *)(*(_QWORD *)v144 + 4 * v68) & 0x7FFFFFFF;
              v73 = 8LL * v72;
              if ( v72 < (unsigned int)v70 )
              {
                v74 = *(_QWORD *)(*(_QWORD *)(v69 + 152) + 8LL * v72);
                if ( v74 )
                {
LABEL_96:
                  v75 = *(unsigned int *)(v67 + 928);
                  v76 = *(_DWORD *)(v74 + 112) & 0x7FFFFFFF;
                  v77 = v76 + 1;
                  if ( (int)v76 + 1 > (unsigned int)v75 && v77 != v75 )
                  {
                    if ( v77 >= v75 )
                    {
                      v101 = *(_DWORD *)(v67 + 936);
                      v102 = *(unsigned int *)(v67 + 940);
                      v103 = v77 - v75;
                      if ( v77 > (unsigned __int64)*(unsigned int *)(v67 + 932) )
                      {
                        v125 = v77 - v75;
                        v132 = *(_DWORD *)(v67 + 940);
                        sub_C8D5F0(v67 + 920, v128, v77, 8u, v62, v102);
                        v75 = *(unsigned int *)(v67 + 928);
                        v103 = v125;
                        LODWORD(v102) = v132;
                      }
                      v104 = (_DWORD *)(*(_QWORD *)(v67 + 920) + 8 * v75);
                      v105 = v103;
                      do
                      {
                        if ( v104 )
                        {
                          *v104 = v101;
                          v104[1] = v102;
                        }
                        v104 += 2;
                        --v105;
                      }
                      while ( v105 );
                      *(_DWORD *)(v67 + 928) += v103;
                    }
                    else
                    {
                      *(_DWORD *)(v67 + 928) = v77;
                    }
                  }
                  *(_DWORD *)(*(_QWORD *)(v67 + 920) + 8 * v76) = 3;
                  v64 = v140;
                  goto LABEL_92;
                }
              }
              v95 = v72 + 1;
              if ( (unsigned int)v70 < v95 )
              {
                v100 = v95;
                if ( v95 != v70 )
                {
                  if ( v95 >= v70 )
                  {
                    v106 = *(_QWORD *)(v69 + 168);
                    v107 = v95 - v70;
                    if ( v100 > *(unsigned int *)(v69 + 164) )
                    {
                      v123 = v107;
                      v126 = *(_QWORD *)(v69 + 168);
                      v133 = v71;
                      sub_C8D5F0(v69 + 152, (const void *)(v69 + 168), v100, 8u, v62, v71);
                      v70 = *(unsigned int *)(v69 + 160);
                      v107 = v123;
                      v106 = v126;
                      LODWORD(v71) = v133;
                    }
                    v96 = *(_QWORD *)(v69 + 152);
                    v108 = (_QWORD *)(v96 + 8 * v70);
                    v109 = &v108[v107];
                    if ( v108 != v109 )
                    {
                      do
                        *v108++ = v106;
                      while ( v109 != v108 );
                      v96 = *(_QWORD *)(v69 + 152);
                    }
                    *(_DWORD *)(v69 + 160) += v107;
                    goto LABEL_124;
                  }
                  *(_DWORD *)(v69 + 160) = v95;
                }
              }
              v96 = *(_QWORD *)(v69 + 152);
LABEL_124:
              v97 = (__int64 *)(v96 + v73);
              v98 = sub_2E10F30(v71);
              *v97 = v98;
              v131 = v98;
              sub_2E11E80((_QWORD *)v69, v98);
              v74 = v131;
              goto LABEL_96;
            }
          }
LABEL_67:
          if ( v64 != v142 )
            _libc_free((unsigned __int64)v64);
          v54 = v145;
          v143[0] = &unk_4A388F0;
          if ( *(_BYTE *)(v145 + 36) )
          {
            v55 = *(_QWORD ***)(v145 + 16);
            v56 = &v55[*(unsigned int *)(v145 + 28)];
            v57 = v55;
            if ( v55 != v56 )
            {
              while ( *v57 != v143 )
              {
                if ( v56 == ++v57 )
                  goto LABEL_75;
              }
              v58 = (unsigned int)(*(_DWORD *)(v145 + 28) - 1);
              *(_DWORD *)(v145 + 28) = v58;
              *v57 = v55[v58];
              ++*(_QWORD *)(v54 + 8);
            }
          }
          else
          {
            v99 = sub_C8CA60(v145 + 8, (__int64)v143);
            if ( v99 )
            {
              *v99 = -2;
              ++*(_DWORD *)(v54 + 32);
              ++*(_QWORD *)(v54 + 8);
            }
          }
LABEL_75:
          if ( !v163 )
            _libc_free((unsigned __int64)v160);
          if ( !v157 )
            _libc_free((unsigned __int64)v154);
          v21 = (_BYTE *)v138[0];
        }
      }
      if ( v21 != v139 )
        _libc_free((unsigned __int64)v21);
      if ( v135 != (unsigned int *)v137 )
        _libc_free((unsigned __int64)v135);
    }
  }
  return 0;
}
