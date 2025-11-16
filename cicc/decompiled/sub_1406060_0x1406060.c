// Function: sub_1406060
// Address: 0x1406060
//
__int64 *__fastcall sub_1406060(__int64 *a1, __int64 **a2, unsigned __int64 a3, __int64 *a4)
{
  __int64 v5; // r11
  __int64 *v7; // rcx
  __int64 *v8; // r12
  __int64 v9; // r13
  __int64 **v10; // r14
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 **v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r9
  __int64 v21; // r9
  unsigned __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r10
  __int64 v25; // rsi
  __int64 v26; // r15
  __int64 **v27; // rax
  __int64 *v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 *result; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 *v34; // rdi
  __int64 v35; // rsi
  __int64 *v36; // rdx
  __int64 v37; // r8
  __int64 v38; // rcx
  __int64 **k; // rdi
  __int64 *v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // r10
  unsigned __int64 v43; // rcx
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rdi
  __int64 *v48; // r12
  __int64 **v49; // r14
  __int64 *v50; // r9
  __int64 v51; // r8
  __int64 v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 **v55; // r10
  __int64 v56; // r8
  __int64 *v57; // rsi
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 **i; // rdi
  __int64 v61; // rcx
  __int64 v62; // rcx
  unsigned __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 *v68; // r14
  __int64 v69; // r15
  __int64 v70; // r12
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rdi
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 *v79; // r14
  __int64 v80; // r15
  __int64 v81; // r13
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 *v84; // rcx
  __int64 v85; // r11
  __int64 v86; // rcx
  __int64 **j; // rdi
  __int64 *v88; // rdx
  __int64 v89; // rsi
  __int64 v90; // rcx
  __int64 **m; // r14
  __int64 *v92; // rdx
  __int64 v93; // rsi
  __int64 v94; // rdx
  __int64 v95; // rdx
  __int64 v96; // [rsp+0h] [rbp-150h]
  __int64 v97; // [rsp+8h] [rbp-148h]
  __int64 *v98; // [rsp+8h] [rbp-148h]
  __int64 *v99; // [rsp+10h] [rbp-140h]
  __int64 v100; // [rsp+10h] [rbp-140h]
  __int64 v101; // [rsp+18h] [rbp-138h]
  __int64 v102; // [rsp+18h] [rbp-138h]
  __int64 v103; // [rsp+20h] [rbp-130h]
  __int64 *v104; // [rsp+20h] [rbp-130h]
  __int64 v105; // [rsp+20h] [rbp-130h]
  __int64 v106; // [rsp+28h] [rbp-128h]
  __int64 v107; // [rsp+28h] [rbp-128h]
  __int64 *v108; // [rsp+30h] [rbp-120h]
  __int64 v109; // [rsp+38h] [rbp-118h]
  __int64 *v110; // [rsp+40h] [rbp-110h]
  __int64 v111; // [rsp+48h] [rbp-108h]
  __int64 v112; // [rsp+48h] [rbp-108h]
  __int64 v113; // [rsp+50h] [rbp-100h]
  __int64 v114; // [rsp+50h] [rbp-100h]
  __int64 *v115; // [rsp+50h] [rbp-100h]
  __int64 v116; // [rsp+58h] [rbp-F8h]
  __int64 v117; // [rsp+58h] [rbp-F8h]
  __int64 v118; // [rsp+58h] [rbp-F8h]
  __int64 v119; // [rsp+58h] [rbp-F8h]
  __int64 v120; // [rsp+60h] [rbp-F0h]
  __int64 *v121; // [rsp+60h] [rbp-F0h]
  __int64 *v122; // [rsp+60h] [rbp-F0h]
  __int64 *v123; // [rsp+60h] [rbp-F0h]
  __int64 *v124; // [rsp+68h] [rbp-E8h]
  __int64 v125; // [rsp+68h] [rbp-E8h]
  __int64 v126; // [rsp+70h] [rbp-E0h]
  __int64 v127; // [rsp+70h] [rbp-E0h]
  __int64 v128; // [rsp+78h] [rbp-D8h]
  __int64 *v129; // [rsp+78h] [rbp-D8h]
  __int64 v130; // [rsp+78h] [rbp-D8h]
  __int64 v131; // [rsp+80h] [rbp-D0h]
  __int64 **v132; // [rsp+80h] [rbp-D0h]
  __int64 v133; // [rsp+80h] [rbp-D0h]
  __int64 v134; // [rsp+80h] [rbp-D0h]
  __int64 *v135; // [rsp+80h] [rbp-D0h]
  __int64 *v136; // [rsp+80h] [rbp-D0h]
  __int64 v137; // [rsp+88h] [rbp-C8h]
  __int64 v138; // [rsp+88h] [rbp-C8h]
  __int64 *v139; // [rsp+88h] [rbp-C8h]
  __int64 v140; // [rsp+88h] [rbp-C8h]
  __int64 v141; // [rsp+88h] [rbp-C8h]
  __int64 v143; // [rsp+88h] [rbp-C8h]
  __int64 *v145; // [rsp+98h] [rbp-B8h] BYREF
  __m128i v146; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v147; // [rsp+B0h] [rbp-A0h]
  __int64 *v148; // [rsp+B8h] [rbp-98h]
  __int64 *v149; // [rsp+C0h] [rbp-90h] BYREF
  __int64 *v150; // [rsp+C8h] [rbp-88h]
  __int64 v151; // [rsp+D0h] [rbp-80h]
  _QWORD *v152; // [rsp+D8h] [rbp-78h]
  __int64 *v153; // [rsp+E0h] [rbp-70h] BYREF
  __int64 *v154; // [rsp+E8h] [rbp-68h]
  __int64 v155; // [rsp+F0h] [rbp-60h]
  __int64 *v156; // [rsp+F8h] [rbp-58h]
  __m128i v157; // [rsp+100h] [rbp-50h] BYREF
  __int64 v158; // [rsp+110h] [rbp-40h]
  __int64 *v159; // [rsp+118h] [rbp-38h]

  v5 = a3;
  v7 = *a2;
  v8 = (__int64 *)a1[2];
  if ( *a2 == v8 )
  {
    v75 = a1[3];
    v76 = ((__int64)v8 - v75) >> 3;
    if ( a3 > v76 )
    {
      v136 = a4;
      sub_1404DC0(a1, a3 - v76);
      v8 = (__int64 *)a1[2];
      v75 = a1[3];
      a4 = v136;
      v5 = a3;
      v76 = ((__int64)v8 - v75) >> 3;
    }
    v77 = a1[5];
    v78 = v76 - v5;
    if ( v78 < 0 )
    {
      v95 = ~((unsigned __int64)~v78 >> 6);
    }
    else
    {
      if ( v78 <= 63 )
      {
        v79 = (__int64 *)a1[5];
        v80 = a1[4];
        v81 = v75;
        v82 = (__int64)&v8[-v5];
LABEL_80:
        v158 = a1[4];
        v159 = (__int64 *)v77;
        v153 = (__int64 *)v82;
        v143 = v82;
        v157.m128i_i64[0] = (__int64)v8;
        v157.m128i_i64[1] = v75;
        v154 = (__int64 *)v81;
        v155 = v80;
        v156 = v79;
        sub_1403E90((__int64)&v153, &v157, a4);
        a1[3] = v81;
        a1[4] = v80;
        a1[2] = v143;
        a1[5] = (__int64)v79;
        return (__int64 *)v143;
      }
      v95 = v78 >> 6;
    }
    v79 = (__int64 *)(v77 + 8 * v95);
    v81 = *v79;
    v80 = *v79 + 512;
    v82 = *v79 + 8 * (v78 - (v95 << 6));
    goto LABEL_80;
  }
  v9 = a1[6];
  if ( v7 == (__int64 *)v9 )
  {
    v62 = a1[8];
    v63 = ((v62 - v9) >> 3) - 1;
    if ( a3 > v63 )
    {
      v135 = a4;
      sub_1404E90(a1, a3 - v63);
      v9 = a1[6];
      v62 = a1[8];
      a4 = v135;
      v5 = a3;
    }
    v64 = a1[7];
    v65 = a1[9];
    v66 = v5 + ((v9 - v64) >> 3);
    if ( v66 < 0 )
    {
      v74 = ~((unsigned __int64)~v66 >> 6);
    }
    else
    {
      if ( v66 <= 63 )
      {
        v67 = v9 + 8 * v5;
        v68 = (__int64 *)a1[9];
        v69 = v62;
        v70 = a1[7];
LABEL_60:
        v154 = (__int64 *)a1[7];
        v156 = (__int64 *)v65;
        v157.m128i_i64[0] = v67;
        v139 = (__int64 *)v67;
        v157.m128i_i64[1] = v70;
        v158 = v69;
        v159 = v68;
        v153 = (__int64 *)v9;
        v155 = v62;
        sub_1403E90((__int64)&v153, &v157, a4);
        result = v139;
        a1[7] = v70;
        a1[8] = v69;
        a1[6] = (__int64)v139;
        a1[9] = (__int64)v68;
        return result;
      }
      v74 = v66 >> 6;
    }
    v68 = (__int64 *)(v65 + 8 * v74);
    v70 = *v68;
    v69 = *v68 + 512;
    v67 = *v68 + 8 * (v66 - (v74 << 6));
    goto LABEL_60;
  }
  v10 = (__int64 **)a1[5];
  v11 = a2[3];
  v12 = (char *)v7 - (char *)a2[1];
  v13 = a1[4];
  v14 = (__int64 *)a1[7];
  v145 = (__int64 *)*a4;
  v15 = (v9 - (__int64)v14) >> 3;
  v16 = (v13 - (__int64)v8) >> 3;
  v17 = (((((char *)v11 - (char *)v10) >> 3) - 1) << 6) + (v12 >> 3);
  v18 = (__int64 **)a1[9];
  v19 = v17 + v16;
  v20 = v15 + ((v18 - v10 - 1) << 6);
  if ( (__int64)((unsigned __int64)(v20 + v16) >> 1) <= v17 + v16 )
  {
    v42 = a1[8];
    v43 = ((v42 - v9) >> 3) - 1;
    if ( v5 > v43 )
    {
      v130 = v17;
      v134 = v15 + ((((a1[9] - (__int64)v10) >> 3) - 1) << 6);
      v141 = v5;
      sub_1404E90(a1, v5 - v43);
      v9 = a1[6];
      v14 = (__int64 *)a1[7];
      v42 = a1[8];
      v18 = (__int64 **)a1[9];
      v17 = v130;
      v20 = v134;
      v5 = v141;
      v15 = (v9 - (__int64)v14) >> 3;
    }
    v44 = v15 + v5;
    if ( v15 + v5 < 0 )
    {
      v73 = ~((unsigned __int64)~v44 >> 6);
    }
    else
    {
      if ( v44 <= 63 )
      {
        v132 = v18;
        v125 = v9 + 8 * v5;
        v127 = v42;
        v129 = v14;
        goto LABEL_33;
      }
      v73 = v44 >> 6;
    }
    v132 = &v18[v73];
    v129 = *v132;
    v127 = (__int64)(*v132 + 64);
    v125 = (__int64)&(*v132)[v44 - (v73 << 6)];
LABEL_33:
    v45 = v20 - v17;
    v46 = v17 - v20;
    v47 = v15 + v46;
    if ( v15 + v46 < 0 )
    {
      v94 = ~((unsigned __int64)~v47 >> 6);
    }
    else
    {
      if ( v47 <= 63 )
      {
        v138 = v42;
        v48 = (__int64 *)(v9 + 8 * v46);
        v49 = v18;
        v50 = v14;
        goto LABEL_36;
      }
      v94 = v47 >> 6;
    }
    v49 = &v18[v94];
    v50 = *v49;
    v48 = &(*v49)[v47 - (v94 << 6)];
    v138 = (__int64)(*v49 + 64);
LABEL_36:
    v51 = v48 - v50 + v5;
    if ( v45 > v5 )
    {
      v52 = v15 - v5;
      if ( v52 < 0 )
      {
        v53 = ~((unsigned __int64)~v52 >> 6);
      }
      else
      {
        if ( v52 <= 63 )
        {
          v121 = (__int64 *)v18;
          v114 = v42;
          v117 = (__int64)v14;
          v111 = v9 - 8 * v5;
LABEL_41:
          v96 = v5;
          v152 = v18;
          v156 = (__int64 *)v18;
          v104 = (__int64 *)v18;
          v157.m128i_i64[1] = v117;
          v158 = v114;
          v159 = v121;
          v100 = v48 - v50 + v5;
          v98 = v50;
          v151 = v42;
          v155 = v42;
          v102 = v42;
          v157.m128i_i64[0] = v111;
          v149 = (__int64 *)v9;
          v150 = v14;
          v153 = (__int64 *)v9;
          v154 = v14;
          sub_1405F80(&v146, v157.m128i_i64, (__int64 *)&v153, &v149);
          v157.m128i_i64[0] = v9;
          v157.m128i_i64[1] = (__int64)v14;
          a1[6] = v125;
          v159 = v104;
          a1[7] = (__int64)v129;
          v153 = (__int64 *)v111;
          a1[8] = v127;
          v155 = v114;
          a1[9] = (__int64)v132;
          v154 = (__int64 *)v117;
          v156 = v121;
          v158 = v102;
          v152 = v49;
          v149 = v48;
          v150 = v98;
          v151 = v138;
          result = sub_1405B90(v146.m128i_i64, (__int64 *)&v149, (__int64)&v153, &v157);
          if ( v100 < 0 )
          {
            v54 = ~((unsigned __int64)~v100 >> 6);
          }
          else
          {
            if ( v100 <= 63 )
            {
              v56 = (__int64)v145;
              v58 = (__int64)&v48[v96];
              goto LABEL_70;
            }
            v54 = v100 >> 6;
          }
          v55 = &v49[v54];
          result = (__int64 *)(v54 << 6);
          v56 = (__int64)v145;
          v57 = *v55;
          v58 = (__int64)&(*v55)[v100 - (_QWORD)result];
          v59 = (__int64)v145;
          if ( v49 != v55 )
          {
            result = (__int64 *)v138;
            if ( v48 != (__int64 *)v138 )
            {
              do
                *v48++ = v56;
              while ( v48 != (__int64 *)v138 );
              v59 = (__int64)v145;
            }
            for ( i = v49 + 1; v55 > i; v59 = (__int64)v145 )
            {
              result = *i;
              v61 = (__int64)(*i + 64);
              do
                *result++ = v59;
              while ( (__int64 *)v61 != result );
              ++i;
            }
            for ( ; (__int64 *)v58 != v57; ++v57 )
              *v57 = v59;
            return result;
          }
LABEL_70:
          while ( v48 != (__int64 *)v58 )
            *v48++ = v56;
          return result;
        }
        v53 = v52 >> 6;
      }
      v121 = (__int64 *)&v18[v53];
      v117 = *v121;
      v114 = *v121 + 512;
      v111 = *v121 + 8 * (v52 - (v53 << 6));
      goto LABEL_41;
    }
    if ( v51 < 0 )
    {
      v83 = ~((unsigned __int64)~v51 >> 6);
    }
    else
    {
      if ( v51 <= 63 )
      {
        v122 = (__int64 *)v49;
        v85 = (__int64)&v48[v5];
        v84 = v50;
        v118 = v138;
LABEL_85:
        v105 = (__int64)v50;
        v155 = v118;
        v156 = v122;
        v153 = (__int64 *)v85;
        v107 = v85;
        v154 = v84;
        v108 = v84;
        v158 = v42;
        v109 = v42;
        v159 = (__int64 *)v18;
        v110 = (__int64 *)v18;
        v157.m128i_i64[0] = v9;
        v157.m128i_i64[1] = (__int64)v14;
        sub_1403E90((__int64)&v157, &v153, (__int64 *)&v145);
        v153 = (__int64 *)v9;
        v154 = v14;
        v152 = v122;
        v150 = v108;
        v151 = v118;
        v158 = v138;
        v156 = v110;
        v149 = (__int64 *)v107;
        v155 = v109;
        v157.m128i_i64[0] = (__int64)v48;
        v157.m128i_i64[1] = v105;
        v159 = (__int64 *)v49;
        sub_1405F80(&v146, v157.m128i_i64, (__int64 *)&v153, &v149);
        a1[6] = v125;
        a1[7] = (__int64)v129;
        a1[8] = v127;
        a1[9] = (__int64)v132;
        result = v110;
        if ( v49 == (__int64 **)v110 )
        {
          for ( result = v145; v48 != (__int64 *)v9; ++v48 )
            *v48 = (__int64)result;
        }
        else
        {
          v86 = (__int64)v145;
          if ( v48 != (__int64 *)v138 )
          {
            do
              *v48++ = v86;
            while ( v48 != (__int64 *)v138 );
            v86 = (__int64)v145;
          }
          for ( j = v49 + 1; j < (__int64 **)v110; v86 = (__int64)v145 )
          {
            v88 = *j;
            v89 = (__int64)(*j + 64);
            do
              *v88++ = v86;
            while ( (__int64 *)v89 != v88 );
            ++j;
          }
          while ( v14 != (__int64 *)v9 )
            *v14++ = v86;
        }
        return result;
      }
      v83 = v51 >> 6;
    }
    v84 = v49[v83];
    v122 = (__int64 *)&v49[v83];
    v85 = (__int64)&v84[v51 - (v83 << 6)];
    v118 = (__int64)(v84 + 64);
    goto LABEL_85;
  }
  v21 = a1[3];
  v22 = ((__int64)v8 - v21) >> 3;
  if ( v5 > v22 )
  {
    v133 = v19;
    v140 = v5;
    sub_1404DC0(a1, v5 - v22);
    v8 = (__int64 *)a1[2];
    v21 = a1[3];
    v13 = a1[4];
    v10 = (__int64 **)a1[5];
    v19 = v133;
    v5 = v140;
    v22 = ((__int64)v8 - v21) >> 3;
  }
  v23 = v22 - v5;
  if ( (__int64)(v22 - v5) < 0 )
  {
    v71 = ~((unsigned __int64)~v23 >> 6);
  }
  else
  {
    if ( v23 <= 63 )
    {
      v137 = (__int64)v10;
      v24 = v21;
      v131 = v13;
      v128 = (__int64)&v8[-v5];
      v25 = v22 + v19;
      if ( (__int64)(v22 + v19) >= 0 )
        goto LABEL_9;
LABEL_64:
      v72 = ~((unsigned __int64)~v25 >> 6);
LABEL_110:
      v27 = &v10[v72];
      v28 = *v27;
      v126 = (__int64)(*v27 + 64);
      v26 = (__int64)&(*v27)[v25 - (v72 << 6)];
      goto LABEL_11;
    }
    v71 = v23 >> 6;
  }
  v24 = (__int64)v10[v71];
  v137 = (__int64)&v10[v71];
  v131 = v24 + 512;
  v128 = v24 + 8 * (v23 - (v71 << 6));
  v25 = v22 + v19;
  if ( (__int64)(v22 + v19) < 0 )
    goto LABEL_64;
LABEL_9:
  if ( v25 > 63 )
  {
    v72 = v25 >> 6;
    goto LABEL_110;
  }
  v126 = v13;
  v26 = (__int64)&v8[v19];
  v27 = v10;
  v28 = (__int64 *)v21;
LABEL_11:
  if ( v19 >= v5 )
  {
    v29 = v5 + v22;
    if ( v29 < 0 )
    {
      v30 = ~((unsigned __int64)~v29 >> 6);
    }
    else
    {
      if ( v29 <= 63 )
      {
        v124 = (__int64 *)v10;
        v113 = (__int64)&v8[v5];
        v116 = v13;
        v120 = v21;
LABEL_16:
        v97 = v5;
        v99 = (__int64 *)v27;
        v150 = (__int64 *)v24;
        v101 = v24;
        v158 = v13;
        v106 = v13;
        v151 = v131;
        v152 = (_QWORD *)v137;
        v153 = (__int64 *)v113;
        v154 = (__int64 *)v120;
        v149 = (__int64 *)v128;
        v155 = v116;
        v156 = v124;
        v157.m128i_i64[1] = v21;
        v103 = v21;
        v157.m128i_i64[0] = (__int64)v8;
        v159 = (__int64 *)v10;
        sub_1405F80(&v146, v157.m128i_i64, (__int64 *)&v153, &v149);
        v157.m128i_i64[0] = (__int64)v8;
        v159 = (__int64 *)v10;
        a1[2] = v128;
        a1[3] = v101;
        a1[5] = v137;
        a1[4] = v131;
        v156 = v99;
        v151 = v116;
        v157.m128i_i64[1] = v103;
        v149 = (__int64 *)v113;
        v158 = v106;
        v150 = (__int64 *)v120;
        v152 = v124;
        v153 = (__int64 *)v26;
        v154 = v28;
        v155 = v126;
        sub_1405E20(&v146, (__int64)&v149, (__int64 *)&v153, (__int64)&v157);
        result = v99;
        v32 = ((v26 - (__int64)v28) >> 3) - v97;
        if ( v32 < 0 )
        {
          v33 = ~((unsigned __int64)~v32 >> 6);
        }
        else
        {
          if ( v32 <= 63 )
          {
            v37 = (__int64)v145;
            v36 = (__int64 *)(v26 - 8 * v97);
LABEL_122:
            if ( (__int64 *)v26 != v36 )
            {
              do
                *v36++ = v37;
              while ( v36 != (__int64 *)v26 );
            }
            return result;
          }
          v33 = v32 >> 6;
        }
        v34 = &v99[v33];
        v35 = *v34 + 512;
        v36 = (__int64 *)(*v34 + 8 * (v32 - (v33 << 6)));
        v37 = (__int64)v145;
        v38 = (__int64)v145;
        if ( v34 != v99 )
        {
          if ( (__int64 *)v35 != v36 )
          {
            do
              *v36++ = v37;
            while ( (__int64 *)v35 != v36 );
            v38 = (__int64)v145;
          }
          for ( k = (__int64 **)(v34 + 1); k < (__int64 **)v99; v38 = (__int64)v145 )
          {
            v40 = *k;
            v41 = (__int64)(*k + 64);
            do
              *v40++ = v38;
            while ( (__int64 *)v41 != v40 );
            ++k;
          }
          while ( v28 != (__int64 *)v26 )
            *v28++ = v38;
          return result;
        }
        goto LABEL_122;
      }
      v30 = v29 >> 6;
    }
    v124 = (__int64 *)&v10[v30];
    v120 = *v124;
    v116 = *v124 + 512;
    v113 = *v124 + 8 * (v29 - (v30 << 6));
    goto LABEL_16;
  }
  v156 = (__int64 *)v27;
  v115 = (__int64 *)v27;
  v155 = v126;
  v149 = (__int64 *)v128;
  v151 = v131;
  v152 = (_QWORD *)v137;
  v150 = (__int64 *)v24;
  v112 = v24;
  v157.m128i_i64[1] = v21;
  v119 = v21;
  v158 = v13;
  v123 = (__int64 *)v13;
  v153 = (__int64 *)v26;
  v154 = v28;
  v157.m128i_i64[0] = (__int64)v8;
  v159 = (__int64 *)v10;
  sub_1405F80(&v146, v157.m128i_i64, (__int64 *)&v153, &v149);
  v153 = v8;
  v156 = (__int64 *)v10;
  v157 = v146;
  v154 = (__int64 *)v119;
  v155 = (__int64)v123;
  v158 = v147;
  v159 = v148;
  sub_1403E90((__int64)&v157, &v153, (__int64 *)&v145);
  result = v115;
  a1[2] = v128;
  a1[3] = v112;
  a1[4] = v131;
  a1[5] = v137;
  if ( v115 == (__int64 *)v10 )
  {
    result = v145;
    if ( (__int64 *)v26 != v8 )
    {
      do
        *v8++ = (__int64)result;
      while ( v8 != (__int64 *)v26 );
    }
  }
  else
  {
    v90 = (__int64)v145;
    if ( v123 != v8 )
    {
      do
        *v8++ = v90;
      while ( v8 != v123 );
      v90 = (__int64)v145;
    }
    for ( m = v10 + 1; m < (__int64 **)v115; v90 = (__int64)v145 )
    {
      v92 = *m;
      v93 = (__int64)(*m + 64);
      do
        *v92++ = v90;
      while ( (__int64 *)v93 != v92 );
      ++m;
    }
    while ( v28 != (__int64 *)v26 )
      *v28++ = v90;
  }
  return result;
}
