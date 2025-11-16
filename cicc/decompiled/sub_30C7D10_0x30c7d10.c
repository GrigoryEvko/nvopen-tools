// Function: sub_30C7D10
// Address: 0x30c7d10
//
__int64 *__fastcall sub_30C7D10(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r12
  unsigned __int64 *v7; // r13
  unsigned __int64 *v8; // r9
  __int64 v9; // r10
  __int64 v10; // r11
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r15
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int64 *v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rcx
  unsigned __int64 *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 *v39; // rsi
  __int64 v40; // rax
  unsigned __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // rsi
  unsigned __int64 v44; // r14
  __int64 v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rax
  unsigned __int64 *v50; // r11
  unsigned __int64 v51; // r13
  __int64 v52; // r15
  __int64 v53; // rax
  __int64 v54; // rcx
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rdi
  unsigned __int64 *v57; // rsi
  __int64 v58; // rax
  unsigned __int64 v59; // r11
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r13
  __int64 v63; // rbx
  __int64 v64; // rdi
  unsigned __int64 v65; // rsi
  __int64 v66; // rcx
  unsigned __int64 *v67; // rdx
  unsigned __int64 v68; // r12
  __int64 v70; // r8
  unsigned __int64 v71; // r15
  __int64 v72; // r12
  unsigned __int64 v73; // r8
  __int64 v74; // rdi
  unsigned __int64 v75; // r9
  __int64 v76; // rcx
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // r10
  __int64 v79; // r13
  __int64 v80; // rdx
  __int64 v81; // [rsp+0h] [rbp-150h]
  __int64 v82; // [rsp+8h] [rbp-148h]
  unsigned __int64 v83; // [rsp+10h] [rbp-140h]
  unsigned __int64 v84; // [rsp+18h] [rbp-138h]
  __int64 v85; // [rsp+18h] [rbp-138h]
  __int64 v86; // [rsp+18h] [rbp-138h]
  unsigned __int64 *v87; // [rsp+20h] [rbp-130h]
  unsigned __int64 *v88; // [rsp+20h] [rbp-130h]
  __int64 v89; // [rsp+28h] [rbp-128h]
  unsigned __int64 *v90; // [rsp+28h] [rbp-128h]
  unsigned __int64 *v91; // [rsp+30h] [rbp-120h]
  char *v92; // [rsp+30h] [rbp-120h]
  __int64 v93; // [rsp+38h] [rbp-118h]
  unsigned __int64 v94; // [rsp+38h] [rbp-118h]
  __int64 v95; // [rsp+38h] [rbp-118h]
  __int64 v96; // [rsp+38h] [rbp-118h]
  __int64 v97; // [rsp+40h] [rbp-110h]
  unsigned __int64 v98; // [rsp+40h] [rbp-110h]
  unsigned __int64 v99; // [rsp+40h] [rbp-110h]
  unsigned __int64 v100; // [rsp+48h] [rbp-108h]
  unsigned __int64 v101; // [rsp+48h] [rbp-108h]
  __int64 v102; // [rsp+48h] [rbp-108h]
  __int64 v103; // [rsp+50h] [rbp-100h]
  unsigned __int64 v104; // [rsp+50h] [rbp-100h]
  char *v105; // [rsp+50h] [rbp-100h]
  __int64 v106; // [rsp+50h] [rbp-100h]
  unsigned __int64 v107; // [rsp+58h] [rbp-F8h]
  __int64 v108; // [rsp+58h] [rbp-F8h]
  __int64 v109; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v110; // [rsp+60h] [rbp-F0h]
  unsigned __int64 *v111; // [rsp+60h] [rbp-F0h]
  __int64 *v112; // [rsp+60h] [rbp-F0h]
  __int64 v113; // [rsp+68h] [rbp-E8h]
  __int64 v114; // [rsp+68h] [rbp-E8h]
  __int64 v115; // [rsp+68h] [rbp-E8h]
  __int64 v116; // [rsp+68h] [rbp-E8h]
  __int64 v119[4]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v120; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int64 v121; // [rsp+A8h] [rbp-A8h]
  unsigned __int64 v122; // [rsp+B0h] [rbp-A0h]
  unsigned __int64 *v123; // [rsp+B8h] [rbp-98h]
  __int64 v124; // [rsp+C0h] [rbp-90h] BYREF
  unsigned __int64 v125; // [rsp+C8h] [rbp-88h]
  __int64 v126; // [rsp+D0h] [rbp-80h]
  unsigned __int64 *v127; // [rsp+D8h] [rbp-78h]
  __int64 v128; // [rsp+E0h] [rbp-70h] BYREF
  unsigned __int64 v129; // [rsp+E8h] [rbp-68h]
  __int64 v130; // [rsp+F0h] [rbp-60h]
  unsigned __int64 *v131; // [rsp+F8h] [rbp-58h]
  __int64 v132; // [rsp+100h] [rbp-50h] BYREF
  unsigned __int64 v133; // [rsp+108h] [rbp-48h]
  __int64 v134; // [rsp+110h] [rbp-40h]
  unsigned __int64 *v135; // [rsp+118h] [rbp-38h]

  v5 = a2;
  v7 = (unsigned __int64 *)a1[5];
  v8 = (unsigned __int64 *)a1[9];
  v9 = a1[6];
  v10 = a1[4];
  v11 = a1[2];
  v110 = a1[7];
  v12 = (__int64)(v9 - v110) >> 3;
  v13 = ((((a2[3] - (__int64)v7) >> 3) - 1) << 6) + ((*a2 - a2[1]) >> 3);
  v14 = (v10 - v11) >> 3;
  v15 = v13 + v14;
  v16 = v12 + ((v8 - v7 - 1) << 6);
  if ( (unsigned __int64)(v16 + v14) >> 1 <= v13 + v14 )
  {
    v108 = a1[8];
    v41 = ((v108 - v9) >> 3) - 1;
    if ( a5 > v41 )
    {
      v102 = v13;
      v106 = v12 + ((v8 - v7 - 1) << 6);
      v116 = a5;
      sub_30C7760(a1, a5 - v41);
      v9 = a1[6];
      a5 = v116;
      v108 = a1[8];
      v8 = (unsigned __int64 *)a1[9];
      v110 = a1[7];
      v16 = v106;
      v12 = (__int64)(v9 - v110) >> 3;
      v13 = v102;
      v42 = v116 + v12;
      if ( v116 + v12 >= 0 )
        goto LABEL_19;
    }
    else
    {
      v42 = a5 + v12;
      if ( a5 + v12 >= 0 )
      {
LABEL_19:
        if ( v42 <= 63 )
        {
          v101 = (unsigned __int64)v8;
          v44 = v110;
          v104 = v9 + 8 * a5;
          v46 = v16 - v13;
          v47 = v13 - v16;
          v98 = v108;
          v48 = v47 + v12;
          if ( v48 >= 0 )
            goto LABEL_22;
          goto LABEL_45;
        }
        v43 = v42 >> 6;
LABEL_21:
        v44 = v8[v43];
        v45 = v42 - (v43 << 6);
        v101 = (unsigned __int64)&v8[v43];
        v46 = v16 - v13;
        v47 = v13 - v16;
        v98 = v44 + 512;
        v104 = v44 + 8 * v45;
        v48 = v47 + v12;
        if ( v48 >= 0 )
        {
LABEL_22:
          if ( v48 <= 63 )
          {
            v52 = v108;
            v51 = v110;
            v53 = v9 + 8 * v47;
            v50 = v8;
LABEL_25:
            *v5 = v53;
            v5[1] = v51;
            v5[2] = v52;
            v5[3] = (__int64)v50;
            if ( v46 <= a5 )
            {
              v86 = v53;
              v76 = a1[6];
              v77 = a1[7];
              v88 = v50;
              v78 = a1[9];
              v134 = a1[8];
              v109 = v134;
              v113 = v76;
              v132 = v76;
              v133 = v77;
              v96 = a3 + 8 * v46;
              v135 = (unsigned __int64 *)v78;
              v90 = (unsigned __int64 *)v78;
              sub_30C5F60(&v120, v96, a4, &v132);
              v133 = v51;
              v134 = v52;
              v124 = v120;
              v128 = v113;
              v125 = v121;
              v129 = v77;
              v126 = v122;
              v132 = v86;
              v127 = v123;
              v130 = v109;
              v131 = v90;
              v135 = v88;
              sub_30C7AD0(&v119, &v132, &v128, &v124);
              a1[7] = v44;
              a1[6] = v104;
              a1[8] = v98;
              a1[9] = v101;
              v128 = *v5;
              v129 = v5[1];
              v130 = v5[2];
              v131 = (unsigned __int64 *)v5[3];
              return sub_30C5F60(&v132, a3, v96, &v128);
            }
            v54 = a1[6];
            v55 = a1[7];
            v56 = a1[8];
            v57 = (unsigned __int64 *)a1[9];
            v58 = ((__int64)(v54 - v55) >> 3) - a5;
            if ( v58 < 0 )
            {
              v79 = ~((unsigned __int64)~v58 >> 6);
            }
            else
            {
              if ( v58 <= 63 )
              {
                v94 = a1[9];
                v59 = a1[8];
                v60 = v54 - 8 * a5;
                v61 = a1[7];
LABEL_29:
                v130 = a1[8];
                v126 = v56;
                v120 = v60;
                v92 = (char *)v60;
                v128 = v54;
                v129 = v55;
                v131 = v57;
                v124 = v54;
                v125 = v55;
                v127 = v57;
                v87 = v8;
                v89 = v9;
                v123 = (unsigned __int64 *)v94;
                v121 = v61;
                v85 = v61;
                v122 = v59;
                sub_30C7AD0(&v132, &v120, &v124, &v128);
                a1[7] = v44;
                a1[6] = v104;
                a1[8] = v98;
                a1[9] = v101;
                v62 = *v5;
                v105 = (char *)v5[2];
                v63 = v5[3];
                if ( v63 == v94 )
                {
                  v132 = v89;
                  v133 = v110;
                  v135 = v87;
                  v134 = v108;
                  sub_30C7BB0(&v128, v62, v92, &v132);
                }
                else
                {
                  v128 = v89;
                  v131 = v87;
                  v129 = v110;
                  v130 = v108;
                  sub_30C7BB0(&v132, v85, v92, &v128);
                  v64 = v132;
                  v65 = v133;
                  v66 = v134;
                  v67 = v135;
                  if ( v63 != v94 - 8 )
                  {
                    v112 = v5;
                    v68 = v94 - 8;
                    do
                    {
                      v128 = v64;
                      v68 -= 8LL;
                      v130 = v66;
                      v131 = v67;
                      v129 = v65;
                      sub_30C7BB0(&v132, *(_QWORD *)(v68 + 8), (char *)(*(_QWORD *)(v68 + 8) + 512LL), &v128);
                      v64 = v132;
                      v65 = v133;
                      v66 = v134;
                      v67 = v135;
                    }
                    while ( v63 != v68 );
                    v5 = v112;
                  }
                  v132 = v64;
                  v135 = v67;
                  v133 = v65;
                  v134 = v66;
                  sub_30C7BB0(&v128, v62, v105, &v132);
                }
                v128 = *v5;
                v129 = v5[1];
                v130 = v5[2];
                v131 = (unsigned __int64 *)v5[3];
                return sub_30C5F60(&v132, a3, a4, &v128);
              }
              v79 = v58 >> 6;
            }
            v61 = v57[v79];
            v94 = (unsigned __int64)&v57[v79];
            v60 = v61 + 8 * (v58 - (v79 << 6));
            v59 = v61 + 512;
            goto LABEL_29;
          }
          v49 = v48 >> 6;
LABEL_24:
          v50 = &v8[v49];
          v51 = *v50;
          v52 = *v50 + 512;
          v53 = *v50 + 8 * (v48 - (v49 << 6));
          goto LABEL_25;
        }
LABEL_45:
        v49 = ~((unsigned __int64)~v48 >> 6);
        goto LABEL_24;
      }
    }
    v43 = ~((unsigned __int64)~v42 >> 6);
    goto LABEL_21;
  }
  v100 = a1[3];
  v17 = (__int64)(v11 - v100) >> 3;
  if ( a5 > v17 )
  {
    v114 = a5;
    sub_30C7690(a1, a5 - v17);
    v11 = a1[2];
    a5 = v114;
    v10 = a1[4];
    v100 = a1[3];
    v7 = (unsigned __int64 *)a1[5];
    v17 = (__int64)(v11 - v100) >> 3;
    v18 = v17 - v114;
    if ( (__int64)(v17 - v114) >= 0 )
      goto LABEL_4;
LABEL_40:
    v19 = ~((unsigned __int64)~v18 >> 6);
    goto LABEL_6;
  }
  v18 = v17 - a5;
  if ( (__int64)(v17 - a5) < 0 )
    goto LABEL_40;
LABEL_4:
  if ( v18 <= 63 )
  {
    v111 = v7;
    v20 = v100;
    v107 = v10;
    v103 = v11 - 8 * a5;
    v21 = v15 + v17;
    if ( v21 >= 0 )
      goto LABEL_7;
LABEL_42:
    v22 = ~((unsigned __int64)~v21 >> 6);
    goto LABEL_9;
  }
  v19 = v18 >> 6;
LABEL_6:
  v20 = v7[v19];
  v111 = &v7[v19];
  v107 = v20 + 512;
  v103 = v20 + 8 * (v18 - (v19 << 6));
  v21 = v15 + v17;
  if ( v21 < 0 )
    goto LABEL_42;
LABEL_7:
  if ( v21 <= 63 )
  {
    v24 = v100;
    v26 = v11 + 8 * v15;
    v23 = v7;
    v25 = v10;
    goto LABEL_10;
  }
  v22 = v21 >> 6;
LABEL_9:
  v23 = &v7[v22];
  v24 = *v23;
  v25 = *v23 + 512;
  v26 = *v23 + 8 * (v21 - (v22 << 6));
LABEL_10:
  *v5 = v26;
  v5[1] = v24;
  v5[2] = v25;
  v5[3] = (__int64)v23;
  if ( v15 >= a5 )
  {
    v27 = a1[2];
    v28 = a1[3];
    v29 = a1[4];
    v30 = a1[5];
    v31 = a5 + ((__int64)(v27 - v28) >> 3);
    if ( v31 < 0 )
    {
      v115 = ~((unsigned __int64)~v31 >> 6);
    }
    else
    {
      if ( v31 <= 63 )
      {
        v91 = (unsigned __int64 *)a1[5];
        v93 = a1[4];
        v97 = v27 + 8 * a5;
        v32 = a1[3];
        goto LABEL_14;
      }
      v115 = v31 >> 6;
    }
    v91 = (unsigned __int64 *)(v30 + 8 * v115);
    v32 = *v91;
    v93 = *v91 + 512;
    v97 = *v91 + 8 * (v31 - (v115 << 6));
LABEL_14:
    v125 = v32;
    v84 = v32;
    v81 = a5;
    v82 = v10;
    v129 = v20;
    v83 = v20;
    v131 = v111;
    v127 = v91;
    v122 = v29;
    v123 = (unsigned __int64 *)v30;
    v128 = v103;
    v120 = v27;
    v121 = v28;
    v130 = v107;
    v124 = v97;
    v126 = v93;
    sub_30C7AD0(&v132, &v120, &v124, &v128);
    v132 = v11;
    v135 = v7;
    a1[2] = v103;
    a1[3] = v83;
    a1[5] = (unsigned __int64)v111;
    v134 = v82;
    a1[4] = v107;
    v33 = *v5;
    v34 = v5[1];
    v126 = v93;
    v35 = v5[2];
    v36 = (unsigned __int64 *)v5[3];
    v133 = v100;
    v125 = v84;
    v128 = v33;
    v127 = v91;
    v129 = v34;
    v130 = v35;
    v131 = v36;
    v124 = v97;
    sub_30C7970(&v120, (__int64)&v124, &v128, (__int64)&v132);
    v37 = *v5;
    v38 = v5[1];
    v39 = (unsigned __int64 *)v5[3];
    v130 = v5[2];
    v128 = v37;
    v129 = v38;
    v131 = v39;
    v40 = ((v37 - v38) >> 3) - v81;
    if ( v40 < 0 )
    {
      v80 = ~((unsigned __int64)~v40 >> 6);
    }
    else
    {
      if ( v40 <= 63 )
      {
        v128 = v37 - 8 * v81;
        return sub_30C5F60(&v132, a3, a4, &v128);
      }
      v80 = v40 >> 6;
    }
    v131 = &v39[v80];
    v129 = *v131;
    v130 = v129 + 512;
    v128 = v129 + 8 * (v40 - (v80 << 6));
    return sub_30C5F60(&v132, a3, a4, &v128);
  }
  v70 = a5 - v15;
  v71 = a1[5];
  v125 = v20;
  v99 = v20;
  v72 = a3 + 8 * v70;
  v73 = a1[3];
  v74 = a1[2];
  v135 = (unsigned __int64 *)v71;
  v75 = a1[4];
  v95 = v10;
  v126 = v107;
  v133 = v73;
  v129 = v24;
  v130 = v25;
  v131 = v23;
  v132 = v74;
  v124 = v103;
  v127 = v111;
  v128 = v26;
  v134 = v75;
  sub_30C7AD0(&v120, &v132, &v128, &v124);
  v132 = v120;
  v133 = v121;
  v134 = v122;
  v135 = v123;
  sub_30C5F60(&v128, a3, v72, &v132);
  v128 = v11;
  a1[3] = v99;
  a1[2] = v103;
  a1[4] = v107;
  v129 = v100;
  a1[5] = (unsigned __int64)v111;
  v130 = v95;
  v131 = v7;
  return sub_30C5F60(&v132, v72, a4, &v128);
}
