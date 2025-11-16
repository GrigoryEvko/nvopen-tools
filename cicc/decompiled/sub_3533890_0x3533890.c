// Function: sub_3533890
// Address: 0x3533890
//
__int64 __fastcall sub_3533890(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  __int64 *v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // r12
  char *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r15
  __int64 v15; // r10
  char *v16; // r14
  int v17; // r9d
  __int64 v18; // rbx
  __int64 v19; // r15
  char *v20; // r12
  __int64 (__fastcall ***v21)(_QWORD); // rax
  __int64 v22; // rdi
  char *v23; // r14
  char *v24; // r13
  __int64 *v25; // r14
  __int64 *i; // rdx
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 *v29; // rdx
  __int64 v30; // r15
  int v31; // eax
  unsigned int v32; // r12d
  __int64 v33; // r15
  int v34; // eax
  int v35; // ebx
  int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // r12
  __int64 *v39; // r13
  __int64 v40; // rdi
  __int64 *v41; // rbx
  __int64 v42; // r13
  char *v43; // r12
  char *v44; // rbx
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 *v48; // r15
  __int64 *v49; // rcx
  __int64 v50; // rdi
  __int64 *v51; // rcx
  __int64 (__fastcall ***v52)(_QWORD); // r13
  int v53; // eax
  unsigned int v54; // r12d
  __int64 v55; // r13
  int v56; // eax
  int v57; // ebx
  int v58; // eax
  __int64 v59; // rdi
  __int64 v60; // r13
  __int64 v61; // r13
  char *v62; // r12
  char *v63; // rbx
  __int64 v64; // rsi
  __int64 v65; // rdi
  __int64 v66; // rbx
  char *v67; // r13
  __int64 (__fastcall ***v68)(_QWORD); // rdx
  __int64 v69; // rdi
  __int64 v70; // r14
  char *v71; // r13
  char *v72; // rbx
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // r10
  char *v76; // rax
  __int64 v77; // r14
  __int64 v78; // rsi
  char *v79; // rcx
  __int64 v80; // r13
  __int64 *v81; // r12
  char *v82; // rbx
  __int64 v83; // rax
  __int64 v84; // rdi
  __int64 v85; // rdx
  char *v86; // r13
  __int64 v87; // rax
  __int64 v88; // rdi
  char *v89; // r14
  char *v90; // r13
  __int64 v91; // rbx
  __int64 v92; // rdx
  __int64 v93; // rdi
  __int64 v94; // r10
  __int64 v95; // r12
  __int64 *v96; // r14
  __int64 *v97; // rbx
  __int64 v98; // rdi
  __int64 v99; // r12
  char *v100; // r14
  __int64 *v101; // rbx
  __int64 v102; // rdi
  __int64 v103; // [rsp+10h] [rbp-90h]
  __int64 v104; // [rsp+18h] [rbp-88h]
  char *v105; // [rsp+18h] [rbp-88h]
  __int64 v106; // [rsp+20h] [rbp-80h]
  __int64 v107; // [rsp+20h] [rbp-80h]
  __int64 v108; // [rsp+20h] [rbp-80h]
  char *v109; // [rsp+20h] [rbp-80h]
  __int64 v110; // [rsp+20h] [rbp-80h]
  __int64 v111; // [rsp+28h] [rbp-78h]
  __int64 v112; // [rsp+28h] [rbp-78h]
  __int64 v113; // [rsp+28h] [rbp-78h]
  __int64 v114; // [rsp+28h] [rbp-78h]
  __int64 v115; // [rsp+30h] [rbp-70h]
  __int64 v116; // [rsp+30h] [rbp-70h]
  __int64 v117; // [rsp+30h] [rbp-70h]
  __int64 v118; // [rsp+30h] [rbp-70h]
  __int64 v119; // [rsp+38h] [rbp-68h]
  __int64 v120; // [rsp+38h] [rbp-68h]
  __int64 v121; // [rsp+40h] [rbp-60h]
  char *v122; // [rsp+48h] [rbp-58h]
  __int64 *v123; // [rsp+48h] [rbp-58h]
  char *v124; // [rsp+50h] [rbp-50h]
  char *v125; // [rsp+58h] [rbp-48h]
  char *v126; // [rsp+58h] [rbp-48h]
  __int64 *v127; // [rsp+60h] [rbp-40h]
  char *v128; // [rsp+68h] [rbp-38h]
  __int64 *v129; // [rsp+68h] [rbp-38h]
  __int64 *v130; // [rsp+68h] [rbp-38h]

  while ( 1 )
  {
    v7 = (__int64 *)a2;
    v8 = a2;
    v9 = a7;
    v124 = a1;
    v127 = (__int64 *)a3;
    result = a7;
    v125 = a6;
    if ( a5 <= a7 )
      result = a5;
    if ( a4 <= result )
      break;
    v11 = a5;
    if ( a5 <= a7 )
    {
      v18 = a3 - (_QWORD)a2;
      v19 = (a3 - (__int64)a2) >> 3;
      if ( a3 - (__int64)a2 <= 0 )
        return result;
      v20 = a6;
      do
      {
        v21 = (__int64 (__fastcall ***)(_QWORD))*v7;
        *v7 = 0;
        v22 = *(_QWORD *)v20;
        *(_QWORD *)v20 = v21;
        if ( v22 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 24LL))(v22);
        ++v7;
        v20 += 8;
        --v19;
      }
      while ( v19 );
      if ( v18 <= 0 )
        v18 = 8;
      result = (__int64)v125;
      v23 = &v125[v18];
      if ( v124 != a2 )
      {
        if ( v125 == v23 )
          return result;
        v24 = a2 - 8;
        v25 = (__int64 *)(v23 - 8);
        for ( i = v127 - 1; ; i = v29 - 1 )
        {
          v30 = *v25;
          v129 = i;
          v31 = (**(__int64 (__fastcall ***)(__int64))v30)(v30);
          LODWORD(v30) = *(_DWORD *)(v30 + 40);
          v32 = v30 * (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)v24 + 8LL))(*(_QWORD *)v24) * v31;
          v33 = *(_QWORD *)v24;
          v34 = (***(__int64 (__fastcall ****)(_QWORD))v24)(*(_QWORD *)v24);
          LODWORD(v33) = *(_DWORD *)(v33 + 40);
          v35 = v34;
          v36 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)*v25 + 8LL))(*v25);
          v29 = v129;
          if ( v32 > v36 * v35 * (int)v33 )
          {
            v27 = *(_QWORD *)v24;
            *(_QWORD *)v24 = 0;
            v28 = *v129;
            *v129 = v27;
            if ( v28 )
            {
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 24LL))(v28);
              v29 = v129;
            }
            if ( v24 == v124 )
            {
              result = (char *)(v25 + 1) - v125;
              v95 = result >> 3;
              if ( result > 0 )
              {
                v96 = &v25[-v95];
                v97 = &v29[-v95];
                do
                {
                  result = v96[v95];
                  v96[v95] = 0;
                  v98 = v97[v95 - 1];
                  v97[v95 - 1] = result;
                  if ( v98 )
                    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v98 + 24LL))(v98);
                  --v95;
                }
                while ( v95 );
              }
              return result;
            }
            v24 -= 8;
          }
          else
          {
            result = *v25;
            *v25 = 0;
            v37 = *v129;
            *v129 = result;
            if ( v37 )
            {
              result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v37 + 24LL))(v37);
              v29 = v129;
            }
            if ( v125 == (char *)v25 )
              return result;
            --v25;
          }
        }
      }
      v99 = v18 >> 3;
      v100 = &v23[-8 * (v18 >> 3)];
      v101 = &v127[-(v18 >> 3)];
      do
      {
        result = *(_QWORD *)&v100[8 * v99 - 8];
        *(_QWORD *)&v100[8 * v99 - 8] = 0;
        v102 = v101[v99 - 1];
        v101[v99 - 1] = result;
        if ( v102 )
          result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v102 + 24LL))(v102);
        --v99;
      }
      while ( v99 );
      return result;
    }
    v119 = a4;
    if ( a4 <= a5 )
    {
      v14 = a5 / 2;
      v122 = &a2[8 * (a5 / 2)];
      v76 = (char *)sub_3532D60(a1, (__int64)a2, v122);
      v13 = v119;
      v128 = v76;
      v121 = (v76 - a1) >> 3;
    }
    else
    {
      v121 = a4 / 2;
      v128 = &a1[8 * (a4 / 2)];
      v12 = (char *)sub_3532E20(a2, a3, v128);
      v13 = v119;
      v122 = v12;
      v14 = (v12 - a2) >> 3;
    }
    v120 = v13 - v121;
    if ( v13 - v121 > v14 && v14 <= a7 )
    {
      v15 = (__int64)v128;
      if ( !v14 )
        goto LABEL_10;
      v113 = a2 - v128;
      v117 = v122 - a2;
      v77 = (a2 - v128) >> 3;
      v78 = (v122 - a2) >> 3;
      if ( v122 - v8 <= 0 )
      {
        if ( v113 <= 0 )
          goto LABEL_10;
        v118 = 0;
        v110 = 0;
LABEL_84:
        v78 = (__int64)&v8[-8 * v77];
        v86 = &v122[-8 * v77];
        do
        {
          v87 = *(_QWORD *)(v78 + 8 * v77 - 8);
          *(_QWORD *)(v78 + 8 * v77 - 8) = 0;
          v88 = *(_QWORD *)&v86[8 * v77 - 8];
          *(_QWORD *)&v86[8 * v77 - 8] = v87;
          if ( v88 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v88 + 24LL))(v88);
          --v77;
        }
        while ( v77 );
      }
      else
      {
        v79 = v8;
        v109 = v8;
        v80 = (v122 - v8) >> 3;
        v103 = v11;
        v81 = (__int64 *)v79;
        v82 = v125;
        do
        {
          v83 = *v81;
          *v81 = 0;
          v84 = *(_QWORD *)v82;
          *(_QWORD *)v82 = v83;
          if ( v84 )
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v84 + 24LL))(v84, v78);
          ++v81;
          v82 += 8;
          --v80;
        }
        while ( v80 );
        v85 = 8;
        v8 = v109;
        v11 = v103;
        v9 = a7;
        if ( v117 > 0 )
          v85 = v117;
        v110 = v85;
        v118 = v85 >> 3;
        if ( v113 > 0 )
          goto LABEL_84;
      }
      if ( v110 <= 0 )
      {
        v15 = (__int64)v128;
      }
      else
      {
        v89 = v125;
        v114 = v9;
        v90 = v128;
        v91 = v118;
        do
        {
          v92 = *(_QWORD *)v89;
          *(_QWORD *)v89 = 0;
          v93 = *(_QWORD *)v90;
          *(_QWORD *)v90 = v92;
          if ( v93 )
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v93 + 24LL))(v93, v78);
          v89 += 8;
          v90 += 8;
          --v91;
        }
        while ( v91 );
        v9 = v114;
        v94 = 8 * v118;
        if ( v118 <= 0 )
          v94 = 8;
        v15 = (__int64)&v128[v94];
      }
      goto LABEL_10;
    }
    if ( v120 > a7 )
    {
      v15 = sub_35321D0((__int64)v128, (__int64)a2, (__int64)v122);
      goto LABEL_10;
    }
    v15 = (__int64)v122;
    if ( !v120 )
      goto LABEL_10;
    v111 = v122 - a2;
    v115 = a2 - v128;
    v60 = (v122 - a2) >> 3;
    if ( a2 - v128 <= 0 )
    {
      if ( v111 <= 0 )
        goto LABEL_10;
      v116 = 0;
      v107 = 0;
      v105 = v125;
    }
    else
    {
      v106 = (v122 - a2) >> 3;
      v61 = v115 >> 3;
      v104 = v11;
      v62 = v125;
      v63 = v128;
      do
      {
        v64 = *(_QWORD *)v63;
        *(_QWORD *)v63 = 0;
        v65 = *(_QWORD *)v62;
        *(_QWORD *)v62 = v64;
        if ( v65 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v65 + 24LL))(v65);
        v63 += 8;
        v62 += 8;
        --v61;
      }
      while ( v61 );
      v60 = v106;
      v11 = v104;
      v9 = a7;
      v107 = v115;
      v105 = &v125[v115];
      v116 = v115 >> 3;
      if ( v111 <= 0 )
        goto LABEL_67;
    }
    v112 = v9;
    v66 = v60;
    v67 = v128;
    do
    {
      v68 = (__int64 (__fastcall ***)(_QWORD))*v7;
      *v7 = 0;
      v69 = *(_QWORD *)v67;
      *(_QWORD *)v67 = v68;
      if ( v69 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v69 + 24LL))(v69);
      ++v7;
      v67 += 8;
      --v66;
    }
    while ( v66 );
    v9 = v112;
LABEL_67:
    if ( v107 <= 0 )
    {
      v15 = (__int64)v122;
    }
    else
    {
      v70 = v116;
      v108 = v9;
      v71 = &v122[-8 * v116];
      v72 = &v105[-8 * v116];
      do
      {
        v73 = *(_QWORD *)&v72[8 * v70 - 8];
        *(_QWORD *)&v72[8 * v70 - 8] = 0;
        v74 = *(_QWORD *)&v71[8 * v70 - 8];
        *(_QWORD *)&v71[8 * v70 - 8] = v73;
        if ( v74 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v74 + 24LL))(v74);
        --v70;
      }
      while ( v70 );
      v75 = -8;
      v9 = v108;
      if ( v116 > 0 )
        v75 = -8 * v116;
      v15 = (__int64)&v122[v75];
    }
LABEL_10:
    v16 = v125;
    v17 = (int)v125;
    v126 = (char *)v15;
    sub_3533890((_DWORD)v124, (_DWORD)v128, v15, v121, v14, v17, v9);
    a7 = v9;
    a4 = v120;
    a2 = v122;
    a6 = v16;
    a5 = v11 - v14;
    a1 = v126;
    a3 = (__int64)v127;
  }
  v42 = a2 - a1;
  v43 = a1;
  v44 = a6;
  v45 = (a2 - a1) >> 3;
  if ( a2 - a1 > 0 )
  {
    do
    {
      v46 = *(_QWORD *)v43;
      *(_QWORD *)v43 = 0;
      v47 = *(_QWORD *)v44;
      *(_QWORD *)v44 = v46;
      if ( v47 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v47 + 24LL))(v47);
      v43 += 8;
      v44 += 8;
      --v45;
    }
    while ( v45 );
    result = 8;
    v48 = (__int64 *)v125;
    if ( v42 <= 0 )
      v42 = 8;
    v123 = (__int64 *)&v125[v42];
    if ( v125 != &v125[v42] )
    {
      v49 = (__int64 *)v124;
      while ( 1 )
      {
        v130 = v49;
        v41 = v49;
        if ( v7 == v127 )
          break;
        v52 = (__int64 (__fastcall ***)(_QWORD))*v7;
        v53 = (**(__int64 (__fastcall ***)(__int64))*v7)(*v7);
        LODWORD(v52) = *((_DWORD *)v52 + 10);
        v54 = (_DWORD)v52 * (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)*v48 + 8LL))(*v48) * v53;
        v55 = *v48;
        v56 = (**(__int64 (__fastcall ***)(__int64))*v48)(*v48);
        LODWORD(v55) = *(_DWORD *)(v55 + 40);
        v57 = v56;
        v58 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)*v7 + 8LL))(*v7);
        v51 = v130;
        if ( v54 > v58 * v57 * (int)v55 )
        {
          result = *v7;
          *v7 = 0;
          v50 = *v130;
          *v130 = result;
          if ( v50 )
          {
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v50 + 24LL))(v50);
            v51 = v130;
          }
          ++v7;
        }
        else
        {
          result = *v48;
          *v48 = 0;
          v59 = *v130;
          *v130 = result;
          if ( v59 )
          {
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v59 + 24LL))(v59);
            v51 = v130;
          }
          ++v48;
        }
        v49 = v51 + 1;
        if ( v48 == v123 )
          return result;
      }
      v38 = v123 - v48;
      if ( (char *)v123 - (char *)v48 > 0 )
      {
        v39 = v48;
        do
        {
          result = *v39;
          *v39 = 0;
          v40 = *v41;
          *v41 = result;
          if ( v40 )
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v40 + 24LL))(v40);
          ++v39;
          ++v41;
          --v38;
        }
        while ( v38 );
      }
    }
  }
  return result;
}
