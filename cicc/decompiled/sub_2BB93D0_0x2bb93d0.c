// Function: sub_2BB93D0
// Address: 0x2bb93d0
//
void __fastcall sub_2BB93D0(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rax
  unsigned __int64 *v9; // r15
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // r10
  int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned __int64 *v25; // r12
  __int64 v26; // r14
  unsigned __int64 *v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 *v31; // r12
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int64 *v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // r12
  __int64 v42; // rbx
  unsigned __int64 *v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // r14
  __int64 v47; // rbx
  unsigned __int64 *v48; // rsi
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rax
  unsigned __int64 *v52; // rsi
  __int64 v53; // rax
  unsigned __int64 *v54; // rbx
  __int64 v55; // r14
  unsigned __int64 *v56; // r12
  unsigned __int64 *v57; // rbx
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned __int64 *v62; // r14
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rbx
  __int64 v66; // r15
  unsigned __int64 *v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // r14
  unsigned __int64 *v70; // r15
  __int64 v71; // rbx
  __int64 v72; // r14
  unsigned __int64 *v73; // rsi
  __int64 v74; // rdi
  __int64 v75; // r12
  __int64 v76; // r14
  __int64 v77; // r10
  __int64 v78; // rdx
  unsigned __int64 *v79; // r15
  __int64 v80; // rbx
  __int64 v81; // r14
  unsigned __int64 *v82; // rsi
  __int64 v83; // rdi
  __int64 v84; // rax
  __int64 v85; // r14
  __int64 v86; // rbx
  __int64 v87; // r15
  unsigned __int64 *v88; // r12
  __int64 v89; // r15
  __int64 v90; // r14
  unsigned __int64 *v91; // rsi
  __int64 v92; // rdi
  __int64 v93; // r10
  unsigned __int64 *v94; // rbx
  __int64 v95; // r12
  __int64 v96; // r13
  __int64 v97; // r13
  __int64 v98; // r12
  unsigned __int64 *i; // rbx
  __int64 v100; // [rsp-8h] [rbp-98h]
  __int64 v101; // [rsp+8h] [rbp-88h]
  __int64 v102; // [rsp+10h] [rbp-80h]
  __int64 v103; // [rsp+10h] [rbp-80h]
  __int64 v104; // [rsp+18h] [rbp-78h]
  __int64 v105; // [rsp+18h] [rbp-78h]
  __int64 v106; // [rsp+18h] [rbp-78h]
  __int64 v107; // [rsp+18h] [rbp-78h]
  __int64 v108; // [rsp+20h] [rbp-70h]
  __int64 v109; // [rsp+20h] [rbp-70h]
  int v110; // [rsp+28h] [rbp-68h]
  __int64 v112; // [rsp+38h] [rbp-58h]
  __int64 v113; // [rsp+38h] [rbp-58h]
  __int64 v114; // [rsp+38h] [rbp-58h]
  __int64 v115; // [rsp+40h] [rbp-50h]
  __int64 v116; // [rsp+48h] [rbp-48h]
  __int64 v117; // [rsp+50h] [rbp-40h]
  unsigned __int64 *v118; // [rsp+50h] [rbp-40h]
  unsigned __int64 *v119; // [rsp+50h] [rbp-40h]
  __int64 v120; // [rsp+58h] [rbp-38h]
  __int64 v121; // [rsp+58h] [rbp-38h]
  __int64 v122; // [rsp+58h] [rbp-38h]

  v8 = a5;
  v9 = (unsigned __int64 *)a1;
  v10 = (unsigned __int64 *)a6;
  v11 = a2;
  if ( a7 <= a5 )
    v8 = a7;
  v116 = a3;
  v120 = a4;
  if ( a4 <= v8 )
  {
LABEL_17:
    v22 = (char *)v11 - (char *)v9;
    v121 = (char *)v11 - (char *)v9;
    v23 = ((char *)v11 - (char *)v9) >> 6;
    if ( (char *)v11 - (char *)v9 > 0 )
    {
      v118 = v11;
      v24 = (__int64)v10;
      v25 = v9;
      v26 = v23;
      do
      {
        v27 = v25;
        v28 = v24;
        v25 += 8;
        v24 += 64;
        sub_2BB7BD0(v28, v27, a3, v22, a5, a6);
        --v26;
      }
      while ( v26 );
      v29 = v121;
      v30 = 64;
      v31 = v118;
      if ( v121 > 0 )
        v30 = v121;
      v32 = (unsigned __int64 *)((char *)v10 + v30);
      if ( (unsigned __int64 *)((char *)v10 + v30) != v10 && (unsigned __int64 *)v116 != v118 )
      {
        do
        {
          if ( (unsigned __int8)sub_2B1D420(
                                  *(unsigned __int8 **)(*v31 + 8),
                                  *(unsigned __int8 **)(*v10 + 8),
                                  a3,
                                  v29,
                                  a5,
                                  a6) )
          {
            v33 = v31;
            v34 = (__int64)v9;
            v31 += 8;
            v9 += 8;
            sub_2BB7BD0(v34, v33, v35, v36, v37, v38);
            if ( v32 == v10 )
              break;
          }
          else
          {
            v39 = v10;
            v40 = (__int64)v9;
            v10 += 8;
            v9 += 8;
            sub_2BB7BD0(v40, v39, v35, v36, v37, v38);
            if ( v32 == v10 )
              break;
          }
        }
        while ( (unsigned __int64 *)v116 != v31 );
      }
      if ( v10 != v32 )
      {
        v41 = (char *)v32 - (char *)v10;
        v42 = ((char *)v32 - (char *)v10) >> 6;
        if ( v41 > 0 )
        {
          do
          {
            v43 = v10;
            v44 = (__int64)v9;
            v10 += 8;
            v9 += 8;
            sub_2BB7BD0(v44, v43, a3, v29, a5, a6);
            --v42;
          }
          while ( v42 );
        }
      }
    }
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v13 = a4;
      v110 = a1;
      v14 = a1;
      v117 = (__int64)a2;
      if ( a5 >= a4 )
        goto LABEL_15;
LABEL_6:
      v112 = v13 / 2;
      v115 = v14 + ((v13 / 2) << 6);
      v117 = sub_2B1D790(v117, v116, v115, a8, a5, a6);
      v17 = (v117 - (__int64)v11) >> 6;
      while ( 1 )
      {
        v120 -= v112;
        if ( v120 > v17 && v17 <= a7 )
        {
          v18 = v115;
          v19 = v115;
          if ( !v17 )
            goto LABEL_10;
          v78 = v117 - (_QWORD)v11;
          v106 = (__int64)v11 - v115;
          if ( v117 - (__int64)v11 <= 0 )
          {
            if ( v106 <= 0 )
              goto LABEL_10;
            v109 = 0;
            v85 = 0;
LABEL_72:
            v107 = v12;
            v86 = v117;
            v87 = ((__int64)v11 - v115) >> 6;
            do
            {
              v11 -= 8;
              v86 -= 64;
              sub_2BB7BD0(v86, v11, v78, v18, v15, v16);
              --v87;
            }
            while ( v87 );
            v12 = v107;
          }
          else
          {
            v101 = v12;
            v79 = v11;
            v80 = a6;
            v81 = (v117 - (__int64)v11) >> 6;
            do
            {
              v82 = v79;
              v83 = v80;
              v79 += 8;
              v80 += 64;
              sub_2BB7BD0(v83, v82, v78, v18, v15, v16);
              --v81;
            }
            while ( v81 );
            v84 = v117 - (_QWORD)v11;
            v12 = v101;
            if ( v117 - (__int64)v11 <= 0 )
              v84 = 64;
            v85 = v84;
            v109 = v84 >> 6;
            if ( v106 > 0 )
              goto LABEL_72;
          }
          if ( v85 <= 0 )
          {
            v19 = v115;
          }
          else
          {
            v88 = (unsigned __int64 *)a6;
            v89 = v115;
            v90 = v109;
            do
            {
              v91 = v88;
              v92 = v89;
              v88 += 8;
              v89 += 64;
              sub_2BB7BD0(v92, v91, v78, v18, v15, v16);
              --v90;
            }
            while ( v90 );
            v93 = v109 << 6;
            if ( v109 <= 0 )
              v93 = 64;
            v19 = v115 + v93;
          }
          goto LABEL_10;
        }
        if ( v120 > a7 )
        {
          v19 = sub_2BB8FF0(v115, (__int64)v11, v117, v112, v15, v16);
          goto LABEL_10;
        }
        v19 = v117;
        if ( !v120 )
          goto LABEL_10;
        v62 = (unsigned __int64 *)v115;
        v102 = v117 - (_QWORD)v11;
        v63 = (v117 - (__int64)v11) >> 6;
        v64 = (__int64)v11 - v115;
        if ( (__int64)v11 - v115 <= 0 )
        {
          if ( v102 <= 0 )
            goto LABEL_10;
          v108 = 0;
          v70 = (unsigned __int64 *)a6;
          v105 = 0;
        }
        else
        {
          v104 = v12;
          v65 = a6;
          v66 = ((__int64)v11 - v115) >> 6;
          do
          {
            v67 = v62;
            v68 = v65;
            v62 += 8;
            v65 += 64;
            sub_2BB7BD0(v68, v67, v63, v64, v15, v16);
            --v66;
          }
          while ( v66 );
          v69 = 64;
          v12 = v104;
          if ( (__int64)v11 - v115 > 0 )
            v69 = (__int64)v11 - v115;
          v70 = (unsigned __int64 *)(a6 + v69);
          v105 = v69;
          v108 = v69 >> 6;
          if ( v102 <= 0 )
            goto LABEL_60;
        }
        v103 = v12;
        v71 = v115;
        v72 = (v117 - (__int64)v11) >> 6;
        do
        {
          v73 = v11;
          v74 = v71;
          v11 += 8;
          v71 += 64;
          sub_2BB7BD0(v74, v73, v63, v64, v15, v16);
          --v72;
        }
        while ( v72 );
        v12 = v103;
LABEL_60:
        if ( v105 <= 0 )
        {
          v19 = v117;
        }
        else
        {
          v75 = v117;
          v76 = v108;
          do
          {
            v70 -= 8;
            v75 -= 64;
            sub_2BB7BD0(v75, v70, v63, v64, v15, v16);
            --v76;
          }
          while ( v76 );
          v77 = -64 * v108;
          if ( v108 <= 0 )
            v77 = -64;
          v19 = v117 + v77;
        }
LABEL_10:
        v20 = v112;
        v12 -= v17;
        v113 = v19;
        sub_2BB93D0(v110, v115, v19, v20, v17, a6, a7, a8);
        v21 = v12;
        if ( v12 > a7 )
          v21 = a7;
        v14 = v113;
        a3 = v100;
        if ( v120 <= v21 )
        {
          v11 = (unsigned __int64 *)v117;
          v10 = (unsigned __int64 *)a6;
          v9 = (unsigned __int64 *)v113;
          goto LABEL_17;
        }
        if ( v12 <= a7 )
        {
          v11 = (unsigned __int64 *)v117;
          v10 = (unsigned __int64 *)a6;
          v9 = (unsigned __int64 *)v113;
          break;
        }
        v13 = v120;
        v11 = (unsigned __int64 *)v117;
        v110 = v113;
        if ( v12 < v120 )
          goto LABEL_6;
LABEL_15:
        v114 = v14;
        v17 = v12 / 2;
        v117 += (v12 / 2) << 6;
        v115 = sub_2B1D820(v14, (__int64)v11, v117, a8, a5, a6);
        v112 = (v115 - v114) >> 6;
      }
    }
    v45 = v116 - (_QWORD)v11;
    v122 = v116 - (_QWORD)v11;
    if ( v116 - (__int64)v11 > 0 )
    {
      v119 = v11;
      v46 = (__int64)v10;
      v47 = (v116 - (__int64)v11) >> 6;
      do
      {
        v48 = v11;
        v49 = v46;
        v11 += 8;
        v46 += 64;
        sub_2BB7BD0(v49, v48, v45, a4, a5, a6);
        --v47;
      }
      while ( v47 );
      v50 = v122;
      v51 = v122 - 64;
      if ( v122 <= 0 )
        v51 = 0;
      v52 = (unsigned __int64 *)((char *)v10 + v51);
      v53 = 64;
      if ( v122 > 0 )
        v53 = v122;
      v54 = (unsigned __int64 *)((char *)v10 + v53);
      if ( v119 == v9 )
      {
        v97 = v53 >> 6;
        v98 = v116;
        for ( i = v54 - 8; ; v52 = i )
        {
          v98 -= 64;
          i -= 8;
          sub_2BB7BD0(v98, v52, v50, a4, a5, a6);
          if ( !--v97 )
            break;
        }
      }
      else if ( v10 != v54 )
      {
        v55 = v116;
        v56 = v119 - 8;
        v57 = v54 - 8;
        while ( 1 )
        {
          while ( 1 )
          {
            v55 -= 64;
            if ( (unsigned __int8)sub_2B1D420(
                                    *(unsigned __int8 **)(*v57 + 8),
                                    *(unsigned __int8 **)(*v56 + 8),
                                    v50,
                                    a4,
                                    a5,
                                    a6) )
              break;
            sub_2BB7BD0(v55, v57, v58, v59, v60, v61);
            if ( v10 == v57 )
              return;
            v57 -= 8;
          }
          sub_2BB7BD0(v55, v56, v58, v59, v60, v61);
          if ( v56 == v9 )
            break;
          v56 -= 8;
        }
        v94 = v57 + 8;
        v95 = ((char *)v94 - (char *)v10) >> 6;
        if ( (char *)v94 - (char *)v10 > 0 )
        {
          v96 = v55;
          do
          {
            v94 -= 8;
            v96 -= 64;
            sub_2BB7BD0(v96, v94, v50, a4, a5, a6);
            --v95;
          }
          while ( v95 );
        }
      }
    }
  }
}
