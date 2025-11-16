// Function: sub_7EC960
// Address: 0x7ec960
//
void __fastcall sub_7EC960(const __m128i *a1)
{
  __m128i *v1; // r10
  __int64 v2; // r15
  int v3; // r14d
  __m128i *v4; // rsi
  __int16 v5; // cx
  __int64 *v6; // r11
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdi
  unsigned int v21; // eax
  const __m128i *v22; // r10
  int v23; // eax
  __int64 v24; // r8
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rax
  __int64 *v28; // rsi
  __m128i *v29; // rax
  __int64 v31; // rdi
  __m128i *v32; // r13
  _QWORD *k; // rcx
  __int64 v34; // rdi
  __m128i *v35; // rax
  __int64 *v36; // r9
  __int64 v37; // rcx
  __int64 *v38; // r9
  __int64 v39; // r8
  __int64 v40; // rdi
  __int64 v41; // rcx
  _QWORD *m; // rcx
  __m128i *v43; // rax
  __int64 v44; // rdi
  _QWORD *v45; // r8
  __m128i *v46; // rax
  unsigned int v47; // esi
  __int64 v48; // rdi
  __int64 v49; // rdi
  _QWORD *v50; // rax
  __int64 *v51; // rax
  __int64 v52; // rdi
  _DWORD *v53; // rcx
  __int64 *v54; // rax
  __int64 v55; // rdx
  _QWORD *v56; // rax
  _BYTE *v57; // rax
  __int64 j; // r9
  _QWORD *v59; // rax
  __int64 v60; // rax
  _BYTE *v61; // rax
  _BYTE *v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  _QWORD *v65; // rax
  int v66; // eax
  __int64 v67; // rdx
  __int64 *v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // [rsp+10h] [rbp-190h]
  __int64 v71; // [rsp+10h] [rbp-190h]
  __int64 v72; // [rsp+18h] [rbp-188h]
  _DWORD *v73; // [rsp+18h] [rbp-188h]
  __int64 i; // [rsp+20h] [rbp-180h]
  __int64 v75; // [rsp+20h] [rbp-180h]
  __int64 v76; // [rsp+28h] [rbp-178h]
  __int64 v77; // [rsp+28h] [rbp-178h]
  int v78; // [rsp+30h] [rbp-170h]
  __int64 v79; // [rsp+30h] [rbp-170h]
  __int64 *v80; // [rsp+30h] [rbp-170h]
  __int64 *v81; // [rsp+30h] [rbp-170h]
  __int64 v82; // [rsp+30h] [rbp-170h]
  __int64 *v83; // [rsp+30h] [rbp-170h]
  __int64 *v84; // [rsp+38h] [rbp-168h]
  __int64 v85; // [rsp+38h] [rbp-168h]
  __int64 *v86; // [rsp+38h] [rbp-168h]
  __int64 v87; // [rsp+38h] [rbp-168h]
  __int64 v88; // [rsp+38h] [rbp-168h]
  const __m128i *v89; // [rsp+38h] [rbp-168h]
  __int64 v90; // [rsp+38h] [rbp-168h]
  _QWORD *v91; // [rsp+38h] [rbp-168h]
  __m128i *v92; // [rsp+38h] [rbp-168h]
  __int64 *v93; // [rsp+40h] [rbp-160h]
  const __m128i *v94; // [rsp+40h] [rbp-160h]
  __int64 v95; // [rsp+40h] [rbp-160h]
  __int64 *v96; // [rsp+40h] [rbp-160h]
  __int64 v97; // [rsp+40h] [rbp-160h]
  __int64 *v98; // [rsp+40h] [rbp-160h]
  const __m128i *v99; // [rsp+40h] [rbp-160h]
  __int64 v100; // [rsp+40h] [rbp-160h]
  unsigned int v101; // [rsp+40h] [rbp-160h]
  const __m128i *v102; // [rsp+40h] [rbp-160h]
  __m128i *v103; // [rsp+40h] [rbp-160h]
  __int64 v104; // [rsp+40h] [rbp-160h]
  _QWORD *v105; // [rsp+48h] [rbp-158h]
  __int64 *v106; // [rsp+48h] [rbp-158h]
  __m128i *v107; // [rsp+48h] [rbp-158h]
  unsigned int v108; // [rsp+48h] [rbp-158h]
  const __m128i *v109; // [rsp+48h] [rbp-158h]
  __int64 v110; // [rsp+48h] [rbp-158h]
  __m128i *v111; // [rsp+48h] [rbp-158h]
  __int64 v112; // [rsp+48h] [rbp-158h]
  __int64 v113; // [rsp+48h] [rbp-158h]
  __int64 v114; // [rsp+48h] [rbp-158h]
  _QWORD *v115; // [rsp+48h] [rbp-158h]
  __int64 v116; // [rsp+48h] [rbp-158h]
  __int64 v117; // [rsp+48h] [rbp-158h]
  __int64 v118; // [rsp+48h] [rbp-158h]
  _QWORD *v119; // [rsp+48h] [rbp-158h]
  _QWORD *v120; // [rsp+48h] [rbp-158h]
  _QWORD *v121; // [rsp+48h] [rbp-158h]
  _QWORD *v122; // [rsp+48h] [rbp-158h]
  __int64 v123; // [rsp+48h] [rbp-158h]
  const __m128i *v124; // [rsp+48h] [rbp-158h]
  int v125; // [rsp+50h] [rbp-150h]
  __int16 v126; // [rsp+54h] [rbp-14Ch]
  __int16 v127; // [rsp+56h] [rbp-14Ah]
  __m128i *v128; // [rsp+58h] [rbp-148h] BYREF
  __m128i *v129; // [rsp+60h] [rbp-140h] BYREF
  __m128i *v130; // [rsp+68h] [rbp-138h] BYREF
  _DWORD v131[8]; // [rsp+70h] [rbp-130h] BYREF
  __m128i *v132[4]; // [rsp+90h] [rbp-110h] BYREF
  int v133[24]; // [rsp+B0h] [rbp-F0h] BYREF
  _BYTE v134[144]; // [rsp+110h] [rbp-90h] BYREF

  v128 = (__m128i *)a1;
  if ( !a1 )
    return;
  v1 = (__m128i *)a1;
  v2 = unk_4D03EB0;
  v3 = dword_4D03F38[0];
  unk_4D03EB0 = 0;
  v4 = (__m128i *)dword_4F07508[0];
  v5 = dword_4F07508[1];
  v6 = (__int64 *)a1[3].m128i_i64[0];
  v127 = dword_4D03F38[1];
  v125 = dword_4F07508[0];
  *(_QWORD *)dword_4D03F38 = a1->m128i_i64[0];
  v126 = v5;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)dword_4D03F38;
  switch ( a1[2].m128i_i8[8] )
  {
    case 0:
      goto LABEL_12;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 0x10:
      sub_7F3E10(a1);
      v1 = v128;
      goto LABEL_5;
    case 6:
      sub_7E7540(a1);
      v1 = v128;
      goto LABEL_5;
    case 7:
      goto LABEL_11;
    case 8:
      v16 = *(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 152LL);
      for ( i = *(_QWORD *)(qword_4F04C50 + 32LL); *(_BYTE *)(v16 + 140) == 12; v16 = *(_QWORD *)(v16 + 160) )
        ;
      v131[0] = 0;
      v76 = *(_QWORD *)(v16 + 160);
      if ( v6 )
      {
        v106 = v6;
        sub_7F2600(v6, 0);
        v6 = v106;
        v1 = (__m128i *)a1;
      }
      else if ( unk_4F06890 && *(_BYTE *)(i + 174) == 1 )
      {
        v68 = sub_73E830(*(_QWORD *)(qword_4F04C50 + 64LL));
        v1 = (__m128i *)a1;
        v6 = v68;
        a1[3].m128i_i64[0] = (__int64)v68;
        v76 = *v68;
      }
      else if ( unk_4F0688C && *(_BYTE *)(i + 174) == 2 )
      {
        v97 = *(_QWORD *)(qword_4F04C50 + 64LL);
        v116 = sub_7E1C10();
        v50 = sub_73E830(v97);
        v51 = (__int64 *)sub_73E110((__int64)v50, v116);
        v1 = (__m128i *)a1;
        v6 = v51;
        a1[3].m128i_i64[0] = (__int64)v51;
        v76 = *v51;
      }
      v132[0] = v1;
      v17 = v1[4].m128i_i64[1];
      v1[4].m128i_i64[1] = 0;
      v18 = *(_QWORD *)(v16 + 168);
      if ( v17 )
      {
        v19 = qword_4F04C50;
        if ( *(_QWORD *)(qword_4F04C50 + 72LL) )
        {
          v78 = 1;
          v20 = *(_QWORD *)(v17 + 40);
          if ( v20 )
          {
            v93 = v6;
            v107 = v1;
            sub_733650(v20);
            v1 = v107;
            v6 = v93;
            v19 = qword_4F04C50;
          }
        }
        else
        {
          if ( (*(_BYTE *)(v18 + 16) & 0x40) != 0 )
          {
            v81 = v6;
            v102 = v1;
            sub_7F90D0(qword_4D03F58, v134);
            sub_7E7190(v102, (__int64)v133, v132);
            sub_7FEC50(v17, (unsigned int)v134, 0, 0, 1, 0, (__int64)v133, 0, 0);
            v1 = (__m128i *)v102;
            v6 = v81;
          }
          else
          {
            v52 = v76;
            if ( *(_BYTE *)(v76 + 140) == 12 )
            {
              do
                v52 = *(_QWORD *)(v52 + 160);
              while ( *(_BYTE *)(v52 + 140) == 12 );
            }
            else
            {
              v52 = v76;
            }
            v73 = *(_DWORD **)(qword_4F04C50 + 72LL);
            v80 = v6;
            v89 = v1;
            v100 = sub_7E9260(v52, v17, &v130);
            v53 = v73;
            if ( !(_DWORD)v130 )
              v53 = v131;
            v71 = (__int64)v53;
            sub_7F9080(v100, v134);
            *(_QWORD *)(v17 + 8) = v100;
            sub_7E7190(v89, (__int64)v133, v132);
            sub_7FEC50(v17, (unsigned int)v134, 0, 0, 1, 0, (__int64)v133, v71, 0);
            v1 = (__m128i *)v89;
            v6 = v80;
            if ( v100 )
            {
              v54 = sub_73E830(v100);
              v1 = (__m128i *)v89;
              v6 = v54;
              v132[0][3].m128i_i64[0] = (__int64)v54;
              if ( v131[0] )
              {
                v83 = v54;
                sub_7FCA60(v100, v17);
                v6 = v83;
                v1 = (__m128i *)v89;
              }
            }
          }
          v78 = 0;
          v19 = qword_4F04C50;
        }
      }
      else
      {
        if ( (*(_BYTE *)(v18 + 16) & 0x60) == 0x40 )
        {
          v1[3].m128i_i64[0] = 0;
          v90 = (__int64)v6;
          v103 = v1;
          sub_7E7190(v1, (__int64)v133, v132);
          v123 = sub_8D46C0(*(_QWORD *)(qword_4D03F58 + 120));
          v57 = sub_731250(qword_4D03F58);
          for ( j = v123; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v124 = v103;
          v104 = v90;
          v91 = v57;
          v59 = (_QWORD *)sub_72D2E0((_QWORD *)j);
          v60 = sub_72D2E0(v59);
          v61 = sub_73E130(v91, v60);
          v62 = sub_73DCD0(v61);
          v65 = sub_7E6A80(v62, 0x49u, v104, v133, v63, v64);
          if ( v65 )
          {
            *v65 = *(_QWORD *)dword_4D03F38;
            v65[1] = *(_QWORD *)dword_4D03F38;
          }
          v66 = sub_7E71E0(*(_QWORD *)(qword_4F04C50 + 88LL), 0, 1);
          v22 = v124;
          v78 = v66;
          if ( v66 )
          {
            v27 = qword_4F04C50;
            if ( !unk_4D045B0 )
              goto LABEL_50;
            v67 = *(_QWORD *)(qword_4F04C50 + 32LL);
            v108 = v78;
            goto LABEL_149;
          }
LABEL_105:
          if ( !unk_4D045B0
            || !*(_QWORD *)(qword_4F04C50 + 64LL)
            || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 174LL) - 1) <= 1u )
          {
            goto LABEL_51;
          }
          if ( v78 )
          {
            sub_7E7190(v22, (__int64)v133, v132);
            sub_825C40(v133, *(_QWORD *)v132[0]);
            goto LABEL_51;
          }
          v108 = 0;
          goto LABEL_110;
        }
        v78 = 1;
        v19 = qword_4F04C50;
      }
      v84 = v6;
      v94 = v1;
      v21 = sub_7E71E0(*(_QWORD *)(v19 + 88), 0, 1);
      v22 = v94;
      v108 = v21;
      if ( !v84 )
      {
        if ( v21 )
        {
LABEL_47:
          if ( !unk_4D045B0 )
            goto LABEL_48;
          v27 = qword_4F04C50;
          v55 = *(_QWORD *)(qword_4F04C50 + 32LL);
          goto LABEL_128;
        }
        goto LABEL_105;
      }
      v23 = sub_8D2600(v76);
      v22 = v94;
      if ( v23 )
      {
        v94[3].m128i_i64[0] = 0;
        sub_7E7190(v94, (__int64)v133, v132);
        v56 = sub_7E69E0(v84, v133);
        v22 = v94;
        if ( v56 )
        {
          *v56 = *(_QWORD *)dword_4D03F38;
          v56[1] = *(_QWORD *)dword_4D03F38;
        }
        v27 = qword_4F04C50;
        if ( !unk_4D045B0 )
        {
          if ( v108 )
            goto LABEL_50;
          goto LABEL_51;
        }
        v78 = 0;
        v55 = *(_QWORD *)(qword_4F04C50 + 32LL);
LABEL_128:
        if ( !*(_QWORD *)(v27 + 64) || (unsigned __int8)(*(_BYTE *)(v55 + 174) - 1) <= 1u )
          goto LABEL_130;
        if ( v78 )
        {
          v99 = v22;
          sub_7E7190(v22, (__int64)v133, v132);
LABEL_111:
          sub_825C40(v133, *(_QWORD *)v132[0]);
          if ( v108 )
          {
            v22 = v99;
            v27 = qword_4F04C50;
            goto LABEL_50;
          }
          goto LABEL_51;
        }
LABEL_110:
        v99 = v22;
        goto LABEL_111;
      }
      if ( !v108 )
        goto LABEL_105;
      v26 = sub_731920((__int64)v84, 1, 0, v108, v24, v25);
      v22 = v94;
      if ( v26 || (unsigned __int8)(*(_BYTE *)(i + 174) - 1) <= 1u )
        goto LABEL_47;
      v82 = (__int64)v84;
      v92 = sub_7E7CA0(*v84);
      v94[3].m128i_i64[0] = (__int64)sub_73E830((__int64)v92);
      sub_7E7190(v94, (__int64)v133, v132);
      v69 = sub_7E6AB0((__int64)v92, v82, v133);
      v22 = v94;
      if ( v69 )
      {
        *v69 = *(_QWORD *)dword_4D03F38;
        v69[1] = *(_QWORD *)dword_4D03F38;
      }
      v27 = qword_4F04C50;
      if ( !unk_4D045B0 )
        goto LABEL_50;
      v67 = *(_QWORD *)(qword_4F04C50 + 32LL);
LABEL_149:
      if ( !*(_QWORD *)(v27 + 64) )
        goto LABEL_161;
      if ( (unsigned __int8)(*(_BYTE *)(v67 + 174) - 1) > 1u )
        goto LABEL_110;
      v78 = 0;
LABEL_130:
      if ( v108 )
      {
LABEL_48:
        if ( v78 )
        {
          v109 = v22;
          sub_7E7190(v22, (__int64)v133, v132);
          v22 = v109;
          v27 = qword_4F04C50;
LABEL_50:
          v110 = (__int64)v22;
          sub_7E7530(*(_QWORD *)(v27 + 88), (__int64)v133);
          sub_7E2D10(v110);
          goto LABEL_51;
        }
LABEL_161:
        v27 = qword_4F04C50;
        goto LABEL_50;
      }
LABEL_51:
      sub_7E17A0((__int64)v132[0]->m128i_i64);
      v1 = v128;
LABEL_5:
      sub_7FAF20(v1);
      dword_4D03F38[0] = v3;
      unk_4D03EB0 = v2;
      dword_4F07508[0] = v125;
      LOWORD(dword_4F07508[1]) = v126;
      LOWORD(dword_4D03F38[1]) = v127;
      return;
    case 9:
      v10 = (_QWORD *)a1[4].m128i_i64[1];
      v11 = v10[2];
      if ( v11 && (*(_BYTE *)(v11 - 8) & 8) == 0 )
      {
        v119 = v10;
        sub_7EC5C0(v11, v4);
        v10 = v119;
      }
      v12 = v10[1];
      if ( v12 && (*(_BYTE *)(v12 - 8) & 8) == 0 )
      {
        v122 = v10;
        sub_7EC5C0(v12, v4);
        v10 = v122;
      }
      v13 = v10[3];
      if ( v13 && (*(_BYTE *)(v13 - 8) & 8) == 0 )
      {
        v120 = v10;
        sub_7EC5C0(v13, v4);
        v10 = v120;
      }
      v14 = v10[4];
      if ( v14 && (*(_BYTE *)(v14 - 8) & 8) == 0 )
      {
        v121 = v10;
        sub_7EC5C0(v14, v4);
        v10 = v121;
      }
      v15 = v10[5];
      if ( v15 )
      {
        v105 = v10;
        do
        {
          if ( (*(_BYTE *)(v15 - 8) & 8) == 0 )
            sub_7EC5C0(v15, v4);
          v15 = *(_QWORD *)(v15 + 112);
        }
        while ( v15 );
        v10 = v105;
      }
      v48 = v10[10];
      if ( v48 )
      {
        v115 = v10;
        sub_7F2600(v48, 0);
        v10 = v115;
      }
      v49 = v10[11];
      if ( !v49 )
        goto LABEL_4;
      sub_7F2600(v49, 0);
      v1 = v128;
      goto LABEL_5;
    case 0xA:
    case 0x17:
      sub_7F2600(v6, 0);
      goto LABEL_4;
    case 0xB:
      sub_7EDF20(a1, 0, 0, 0, 0);
      v1 = v128;
      goto LABEL_5;
    case 0xC:
      sub_7F2A70(v6, 1);
      sub_7EC960(v128[4].m128i_i64[1]);
      v1 = v128;
      goto LABEL_5;
    case 0xD:
      v36 = (__int64 *)a1[5].m128i_i64[0];
      v132[0] = (__m128i *)a1;
      if ( v36[2] )
      {
        v79 = *v36;
        v86 = v36;
        v95 = v36[2];
        sub_7E18E0((__int64)v134, v95, 0);
        v111 = v132[0];
        sub_7E7090(v132[0], (__int64)v133, v132);
        v37 = v95;
        v38 = v86;
        v39 = v79;
        *(_QWORD *)(v111[5].m128i_i64[0] + 8) = v95;
        v86[2] = 0;
        v40 = *(_QWORD *)(v95 + 88);
        *(_QWORD *)(v95 + 80) = v111;
        if ( v40 )
        {
          sub_7E9190(v40, (__int64)v133);
          v38 = v86;
          v39 = v79;
          v37 = v95;
        }
        if ( v39
          && ((v87 = v37, v96 = v38, v112 = v39, sub_7EC960(v39), v37 = v87, *(_BYTE *)(v112 + 40))
           || *(_QWORD *)(v112 + 16)) )
        {
          *v96 = 0;
          sub_7E7620(v112, (__int64)v133);
          sub_7F3E10(v132[0]);
          v41 = v87;
        }
        else
        {
          v118 = v37;
          sub_7F3E10(v132[0]);
          v41 = v118;
        }
        v113 = v41;
        if ( *(_QWORD *)(v41 + 88) )
        {
          sub_7E1720((__int64)v132[0]->m128i_i64, (__int64)v133);
          sub_7E7530(*(_QWORD *)(v113 + 88), (__int64)v133);
        }
        sub_7E1AA0();
      }
      else if ( *v36 )
      {
        v98 = v36;
        v117 = *v36;
        sub_7EC960(*v36);
        if ( *(_BYTE *)(v117 + 40) || *(_QWORD *)(v117 + 16) )
        {
          *v98 = 0;
          sub_7E7090(v132[0], (__int64)v133, v132);
          sub_7E7620(v117, (__int64)v133);
          sub_7F3E10(v132[0]);
        }
        else
        {
          sub_7F3E10(v132[0]);
        }
      }
      else
      {
        sub_7F3E10(a1);
      }
LABEL_4:
      v1 = v128;
      goto LABEL_5;
    case 0xE:
      v28 = (__int64 *)a1[5].m128i_i64[0];
      v29 = (__m128i *)a1[4].m128i_i64[1];
      v129 = (__m128i *)a1;
      v130 = v29;
      v77 = v28[3];
      v85 = v28[4];
      sub_7E18E0((__int64)v133, v77, 0);
      sub_7E7090(a1, (__int64)v131, &v129);
      *(_QWORD *)(a1[5].m128i_i64[0] + 8) = v77;
      v31 = *(_QWORD *)(v77 + 88);
      *(_QWORD *)(v77 + 80) = a1;
      if ( v31 )
        sub_7E9190(v31, (__int64)v131);
      if ( *v28 )
      {
        sub_7EC960(*v28);
        sub_7E6810(*v28, (__int64)v131, 1);
      }
      sub_7EC8E0(v28[2], (__int64)v131);
      sub_7EC8E0(v28[5], (__int64)v131);
      sub_7EC8E0(v28[6], (__int64)v131);
      v75 = v28[8];
      v72 = v28[7];
      sub_7F2A70(v72, 1);
      sub_7F2600(v75, 0);
      sub_7FAF20(a1);
      sub_7E18E0((__int64)v134, v85, 0);
      v32 = v130;
      v70 = qword_4D03F68[1];
      sub_7E7090(v130, (__int64)v132, &v130);
      if ( (v32[2].m128i_i8[9] & 1) != 0 )
      {
        for ( k = 0; ; k[3] = v130 )
        {
          k = sub_732D20((__int64)v32, v70, 0, k);
          if ( !k )
            break;
        }
        v32[2].m128i_i8[9] &= ~1u;
        v130[2].m128i_i8[9] |= 1u;
      }
      *(_QWORD *)(v32[5].m128i_i64[0] + 8) = v85;
      v34 = *(_QWORD *)(v85 + 88);
      *(_QWORD *)(v85 + 80) = v32;
      if ( v34 )
        sub_7E9190(v34, (__int64)v132);
      sub_7EC8E0(v28[1], (__int64)v132);
      sub_7EC960(v130);
      if ( *(_QWORD *)(v85 + 88) )
      {
        sub_7E1720((__int64)v130, (__int64)v132);
        sub_7E7530(*(_QWORD *)(v85 + 88), (__int64)v132);
      }
      sub_7E1AA0();
      sub_7268E0((__int64)v129, 13);
      v35 = v129;
      *(_QWORD *)(v129[5].m128i_i64[0] + 8) = v75;
      v35[4].m128i_i64[1] = (__int64)v32;
      v35[3].m128i_i64[0] = v72;
      sub_7304E0(v75);
      if ( *(_QWORD *)(v77 + 88) )
      {
        sub_7E1720((__int64)v129, (__int64)v131);
        sub_7E7530(*(_QWORD *)(v77 + 88), (__int64)v131);
      }
      sub_7E1AA0();
      v1 = v128;
      goto LABEL_5;
    case 0xF:
      v7 = a1[5].m128i_i64[0];
      v8 = *(_QWORD *)(v7 + 8);
      if ( v8 )
      {
        sub_7EB190(v8, v4);
        v9 = *(_QWORD *)(v7 + 16);
        if ( v9 )
          sub_7EB190(v9, v4);
      }
LABEL_11:
      sub_7E18B0();
      v1 = v128;
      goto LABEL_5;
    case 0x11:
      sub_804750(a1, 0);
      v1 = v128;
      goto LABEL_5;
    case 0x12:
      sub_7F2990(a1);
      v1 = v128;
      goto LABEL_5;
    case 0x13:
      sub_7DE0F0((__int64)a1, 0, 0);
      v1 = v128;
      goto LABEL_5;
    case 0x14:
    case 0x18:
      goto LABEL_5;
    case 0x15:
      sub_7DA010((__int64)a1);
      v1 = v128;
      goto LABEL_5;
    case 0x16:
      sub_7D76F0(a1);
      v1 = v128;
      goto LABEL_5;
    case 0x19:
      if ( v6 )
      {
LABEL_12:
        sub_7F2600(v6, a1);
        v1 = v128;
      }
      else
      {
        v114 = a1[4].m128i_i64[1];
        v88 = qword_4D03F68[1];
        sub_7E7090(a1, (__int64)v133, &v128);
        if ( (a1[2].m128i_i8[9] & 1) != 0 )
        {
          for ( m = 0; ; m[3] = v128 )
          {
            m = sub_732D20((__int64)a1, v88, 0, m);
            if ( !m )
              break;
          }
          v43 = v128;
          a1[2].m128i_i8[9] &= ~1u;
          v43[2].m128i_i8[9] |= 1u;
        }
        v44 = unk_4D03F50;
        if ( unk_4D03F50 )
        {
          *(_QWORD *)(v114 + 8) = unk_4D03F50;
          v45 = sub_73E830(v44);
          v46 = v128;
          v128[3].m128i_i64[0] = (__int64)v45;
          v46[4].m128i_i64[1] = 0;
          sub_7F9080(*(_QWORD *)(v114 + 8), v134);
          v47 = (unsigned int)v134;
        }
        else
        {
          v101 = unk_4D03F48;
          sub_7F8B60(v128);
          v47 = v101;
        }
        sub_7FEC50(v114, v47, 0, 0, 0, 0, (__int64)v133, (__int64)v132, 0);
        v1 = v128;
      }
      goto LABEL_5;
    default:
      sub_721090();
  }
}
