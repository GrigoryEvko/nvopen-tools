// Function: sub_1C9C2F0
// Address: 0x1c9c2f0
//
__int64 __fastcall sub_1C9C2F0(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 result; // rax
  __int64 *v6; // rcx
  _QWORD *v7; // rbx
  _QWORD *v8; // rdx
  _QWORD *v9; // r8
  unsigned __int64 v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  _QWORD *v14; // r12
  __int64 v15; // rsi
  __int64 v16; // rax
  unsigned __int64 **v17; // r14
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rcx
  int *v21; // rdx
  __int64 v22; // rax
  int *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // r12
  _QWORD *v29; // rdx
  unsigned __int64 v30; // rcx
  _QWORD *v31; // r8
  _QWORD *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rsi
  _QWORD *v35; // r13
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rcx
  int *v41; // rdx
  __int64 v42; // rax
  int *v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r15
  unsigned __int64 v47; // r13
  __int64 v48; // rdi
  __int64 v49; // r8
  __int64 v50; // rdi
  __int64 v51; // r14
  unsigned __int64 v52; // rdx
  _QWORD *v53; // r12
  _QWORD *v54; // rbx
  _QWORD *v55; // rax
  bool v56; // al
  _QWORD *v57; // rax
  _QWORD *v58; // rdi
  _QWORD *v59; // rdx
  __int64 v60; // rsi
  __int64 v61; // rcx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r15
  __int64 v65; // rbx
  _QWORD *v66; // r8
  unsigned __int64 **v67; // r13
  int v68; // r14d
  _QWORD *v69; // rax
  unsigned __int64 v70; // r11
  int v71; // ecx
  _QWORD *v72; // r10
  __int64 v73; // rsi
  __int64 v74; // rdx
  __int64 v75; // rax
  _QWORD *v76; // r8
  _QWORD *v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // rcx
  unsigned int v80; // r8d
  _QWORD *v81; // rsi
  __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // rax
  char v85; // al
  _BYTE *v86; // rsi
  __int64 v87; // r8
  _QWORD *v88; // rsi
  __int64 v89; // r11
  __int64 v90; // r9
  __int64 v91; // rsi
  _QWORD *v92; // rdi
  __int64 v93; // rax
  _BYTE *v94; // rsi
  _QWORD *v96; // [rsp+8h] [rbp-138h]
  unsigned __int64 v97; // [rsp+18h] [rbp-128h]
  _QWORD *v98; // [rsp+28h] [rbp-118h]
  unsigned int v99; // [rsp+34h] [rbp-10Ch]
  _QWORD *v102; // [rsp+68h] [rbp-D8h]
  _QWORD *v104; // [rsp+78h] [rbp-C8h]
  __int64 v105; // [rsp+80h] [rbp-C0h]
  __int64 v106; // [rsp+88h] [rbp-B8h]
  __int64 *v107; // [rsp+90h] [rbp-B0h]
  __int64 v108; // [rsp+90h] [rbp-B0h]
  _QWORD *v109; // [rsp+98h] [rbp-A8h]
  __int64 v110; // [rsp+98h] [rbp-A8h]
  unsigned __int64 v111; // [rsp+A8h] [rbp-98h] BYREF
  unsigned __int64 v112; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v113; // [rsp+B8h] [rbp-88h] BYREF
  __int64 v114; // [rsp+C0h] [rbp-80h] BYREF
  _BYTE *v115; // [rsp+C8h] [rbp-78h]
  _BYTE *v116; // [rsp+D0h] [rbp-70h]
  unsigned __int64 *v117; // [rsp+E0h] [rbp-60h] BYREF
  int v118; // [rsp+E8h] [rbp-58h] BYREF
  __int64 v119; // [rsp+F0h] [rbp-50h]
  int *v120; // [rsp+F8h] [rbp-48h]
  int *v121; // [rsp+100h] [rbp-40h]
  __int64 v122; // [rsp+108h] [rbp-38h]

  result = (__int64)(a2 + 1);
  v6 = (__int64 *)a2[3];
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v105 = (__int64)v6;
  v107 = a2 + 1;
  if ( v6 != a2 + 1 )
  {
    v7 = a1;
    v109 = a1 + 57;
    v102 = a5 + 1;
    while ( 1 )
    {
      v8 = (_QWORD *)v7[58];
      if ( v8 )
      {
        v9 = v109;
        v10 = *(_QWORD *)(v105 + 32);
        v11 = (_QWORD *)v7[58];
        do
        {
          while ( 1 )
          {
            v12 = v11[2];
            v13 = v11[3];
            if ( v11[4] >= v10 )
              break;
            v11 = (_QWORD *)v11[3];
            if ( !v13 )
              goto LABEL_8;
          }
          v9 = v11;
          v11 = (_QWORD *)v11[2];
        }
        while ( v12 );
LABEL_8:
        if ( v9 != v109 )
        {
          v14 = v109;
          if ( v9[4] <= v10 )
          {
            do
            {
              while ( 1 )
              {
                v15 = v8[2];
                v16 = v8[3];
                if ( v8[4] >= v10 )
                  break;
                v8 = (_QWORD *)v8[3];
                if ( !v16 )
                  goto LABEL_14;
              }
              v14 = v8;
              v8 = (_QWORD *)v8[2];
            }
            while ( v15 );
LABEL_14:
            if ( v14 == v109 || (v17 = &v117, v14[4] > v10) )
            {
              v17 = &v117;
              v117 = (unsigned __int64 *)(v105 + 32);
              v14 = (_QWORD *)sub_1C9B520(v7 + 56, v14, &v117);
            }
            v118 = 0;
            v119 = 0;
            v120 = &v118;
            v121 = &v118;
            v122 = 0;
            v18 = v14[7];
            v104 = v7 + 38;
            if ( v18 )
            {
              v19 = sub_1C95990(v18, (__int64)&v118);
              v20 = v19;
              do
              {
                v21 = (int *)v19;
                v19 = *(_QWORD *)(v19 + 16);
              }
              while ( v19 );
              v120 = v21;
              v22 = v20;
              do
              {
                v23 = (int *)v22;
                v22 = *(_QWORD *)(v22 + 24);
              }
              while ( v22 );
              v121 = v23;
              v24 = v14[10];
              v119 = v20;
              v122 = v24;
            }
            v25 = sub_1C9AD30((__int64)v7, (__int64)&v117, a3, a4, v104);
            v26 = v119;
            v106 = v25;
            while ( v26 )
            {
              sub_1C97220(*(_QWORD *)(v26 + 24));
              v27 = v26;
              v26 = *(_QWORD *)(v26 + 16);
              j_j___libc_free_0(v27, 40);
            }
            if ( v106 )
              break;
          }
        }
      }
LABEL_52:
      v105 = sub_220EF30(v105);
      if ( v107 == (__int64 *)v105 )
        goto LABEL_53;
    }
    v28 = sub_220EF30(v105);
    if ( v107 != (__int64 *)v28 )
    {
      while ( 1 )
      {
        v29 = (_QWORD *)v7[58];
        if ( !v29 )
          goto LABEL_51;
        v30 = *(_QWORD *)(v28 + 32);
        v31 = v109;
        v32 = (_QWORD *)v7[58];
        do
        {
          while ( 1 )
          {
            v33 = v32[2];
            v34 = v32[3];
            if ( v32[4] >= v30 )
              break;
            v32 = (_QWORD *)v32[3];
            if ( !v34 )
              goto LABEL_32;
          }
          v31 = v32;
          v32 = (_QWORD *)v32[2];
        }
        while ( v33 );
LABEL_32:
        if ( v31 == v109 )
          goto LABEL_51;
        v35 = v109;
        if ( v31[4] > v30 )
          goto LABEL_51;
        do
        {
          while ( 1 )
          {
            v36 = v29[2];
            v37 = v29[3];
            if ( v29[4] >= v30 )
              break;
            v29 = (_QWORD *)v29[3];
            if ( !v37 )
              goto LABEL_38;
          }
          v35 = v29;
          v29 = (_QWORD *)v29[2];
        }
        while ( v36 );
LABEL_38:
        if ( v35 == v109 || v35[4] > v30 )
        {
          v117 = (unsigned __int64 *)(v28 + 32);
          v35 = (_QWORD *)sub_1C9B520(v7 + 56, v35, v17);
        }
        v118 = 0;
        v119 = 0;
        v120 = &v118;
        v121 = &v118;
        v122 = 0;
        v38 = v35[7];
        if ( v38 )
        {
          v39 = sub_1C95990(v38, (__int64)&v118);
          v40 = v39;
          do
          {
            v41 = (int *)v39;
            v39 = *(_QWORD *)(v39 + 16);
          }
          while ( v39 );
          v120 = v41;
          v42 = v40;
          do
          {
            v43 = (int *)v42;
            v42 = *(_QWORD *)(v42 + 24);
          }
          while ( v42 );
          v121 = v43;
          v44 = v35[10];
          v119 = v40;
          v122 = v44;
        }
        v45 = sub_1C9AD30((__int64)v7, (__int64)v17, a3, a4, v104);
        v46 = v119;
        v47 = v45;
        while ( v46 )
        {
          sub_1C97220(*(_QWORD *)(v46 + 24));
          v48 = v46;
          v46 = *(_QWORD *)(v46 + 16);
          j_j___libc_free_0(v48, 40);
        }
        if ( !v47 )
          goto LABEL_51;
        v49 = *(_QWORD *)(v47 + 40);
        if ( v49 != *(_QWORD *)(v106 + 40) )
          goto LABEL_51;
        v112 = v106;
        v111 = v47;
        v57 = (_QWORD *)a5[2];
        if ( v57 )
        {
          v58 = v102;
          v59 = (_QWORD *)a5[2];
          do
          {
            while ( 1 )
            {
              v60 = v59[2];
              v61 = v59[3];
              if ( v59[4] >= v47 )
                break;
              v59 = (_QWORD *)v59[3];
              if ( !v61 )
                goto LABEL_74;
            }
            v58 = v59;
            v59 = (_QWORD *)v59[2];
          }
          while ( v60 );
LABEL_74:
          if ( v102 != v58 && v58[4] <= v47 )
            goto LABEL_91;
          v62 = *(_QWORD *)(v49 + 48);
          v63 = v49 + 40;
          if ( v62 == v63 )
            goto LABEL_91;
        }
        else
        {
          v62 = *(_QWORD *)(v49 + 48);
          v63 = v49 + 40;
          if ( v62 == v63 )
            goto LABEL_112;
        }
        v64 = v62;
        v96 = v7;
        v65 = v63;
        v66 = v102;
        v97 = v47;
        v67 = v17;
        v68 = 0;
        do
        {
          if ( !v64 )
            BUG();
          if ( *(_BYTE *)(v64 - 8) == 55 )
          {
            v69 = (_QWORD *)a5[2];
            v70 = v64 - 24;
            v71 = v68 + 1;
            v72 = v66;
            v113 = v64 - 24;
            if ( !v69 )
              goto LABEL_87;
            do
            {
              while ( 1 )
              {
                v73 = v69[2];
                v74 = v69[3];
                if ( v69[4] >= v70 )
                  break;
                v69 = (_QWORD *)v69[3];
                if ( !v74 )
                  goto LABEL_85;
              }
              v72 = v69;
              v69 = (_QWORD *)v69[2];
            }
            while ( v73 );
LABEL_85:
            if ( v66 == v72 || v72[4] > v70 )
            {
LABEL_87:
              v98 = v66;
              v117 = (unsigned __int64 *)&v113;
              v75 = sub_1C9C240(a5, v72, v67);
              v66 = v98;
              v71 = v68 + 1;
              v72 = (_QWORD *)v75;
            }
            *((_DWORD *)v72 + 10) = v68;
            v68 = v71;
          }
          v64 = *(_QWORD *)(v64 + 8);
        }
        while ( v64 != v65 );
        v17 = v67;
        v7 = v96;
        v47 = v97;
        v57 = (_QWORD *)a5[2];
        if ( !v57 )
        {
LABEL_112:
          v76 = v102;
          goto LABEL_113;
        }
LABEL_91:
        v76 = v102;
        v77 = v57;
        do
        {
          while ( 1 )
          {
            v78 = v77[2];
            v79 = v77[3];
            if ( v77[4] >= v111 )
              break;
            v77 = (_QWORD *)v77[3];
            if ( !v79 )
              goto LABEL_95;
          }
          v76 = v77;
          v77 = (_QWORD *)v77[2];
        }
        while ( v78 );
LABEL_95:
        if ( v102 != v76 && v76[4] <= v111 )
        {
          v80 = *((_DWORD *)v76 + 10);
LABEL_98:
          v81 = v102;
          do
          {
            while ( 1 )
            {
              v82 = v57[2];
              v83 = v57[3];
              if ( v57[4] >= v112 )
                break;
              v57 = (_QWORD *)v57[3];
              if ( !v83 )
                goto LABEL_102;
            }
            v81 = v57;
            v57 = (_QWORD *)v57[2];
          }
          while ( v82 );
LABEL_102:
          if ( v102 != v81 && v81[4] <= v112 )
            goto LABEL_105;
          goto LABEL_104;
        }
LABEL_113:
        v117 = &v111;
        v87 = sub_1C9C240(a5, v76, v17);
        v57 = (_QWORD *)a5[2];
        v80 = *(_DWORD *)(v87 + 40);
        if ( v57 )
          goto LABEL_98;
        v81 = v102;
LABEL_104:
        v99 = v80;
        v117 = &v112;
        v84 = sub_1C9C240(a5, v81, v17);
        v80 = v99;
        v81 = (_QWORD *)v84;
LABEL_105:
        if ( *((_DWORD *)v81 + 10) <= v80 )
        {
          if ( !(unsigned __int8)sub_1C9AF20(v7, v106, v47, a3) )
          {
            v94 = v115;
            if ( v115 == v116 )
            {
              sub_1287830((__int64)&v114, v115, (_QWORD *)(v105 + 32));
            }
            else
            {
              if ( v115 )
              {
                *(_QWORD *)v115 = *(_QWORD *)(v105 + 32);
                v94 = v115;
              }
              v115 = v94 + 8;
            }
          }
        }
        else
        {
          v85 = sub_1C9AF20(v7, v47, v106, a3);
          v106 = v47;
          if ( !v85 )
          {
            v86 = v115;
            if ( v115 == v116 )
            {
              sub_1287830((__int64)&v114, v115, (_QWORD *)(v28 + 32));
            }
            else
            {
              if ( v115 )
              {
                *(_QWORD *)v115 = *(_QWORD *)(v28 + 32);
                v86 = v115;
              }
              v106 = v47;
              v115 = v86 + 8;
            }
          }
        }
LABEL_51:
        v28 = sub_220EF30(v28);
        if ( v107 == (__int64 *)v28 )
          goto LABEL_52;
      }
    }
LABEL_53:
    v50 = v114;
    result = (__int64)&v115[-v114] >> 3;
    if ( (_DWORD)result )
    {
      v51 = (__int64)v107;
      v110 = 0;
      v108 = 8LL * (unsigned int)result;
      do
      {
        if ( a2[2] )
        {
          v52 = *(_QWORD *)(v50 + v110);
          v53 = (_QWORD *)v51;
          v54 = (_QWORD *)a2[2];
          while ( 1 )
          {
            while ( v54[4] < v52 )
            {
              v54 = (_QWORD *)v54[3];
              if ( !v54 )
                goto LABEL_61;
            }
            v55 = (_QWORD *)v54[2];
            if ( v54[4] <= v52 )
              break;
            v53 = v54;
            v54 = (_QWORD *)v54[2];
            if ( !v55 )
            {
LABEL_61:
              v56 = v51 == (_QWORD)v53;
              goto LABEL_62;
            }
          }
          v88 = (_QWORD *)v54[3];
          if ( v88 )
          {
            do
            {
              while ( 1 )
              {
                v89 = v88[2];
                v90 = v88[3];
                if ( v88[4] > v52 )
                  break;
                v88 = (_QWORD *)v88[3];
                if ( !v90 )
                  goto LABEL_120;
              }
              v53 = v88;
              v88 = (_QWORD *)v88[2];
            }
            while ( v89 );
          }
LABEL_120:
          while ( v55 )
          {
            while ( 1 )
            {
              v91 = v55[3];
              if ( v55[4] >= v52 )
                break;
              v55 = (_QWORD *)v55[3];
              if ( !v91 )
                goto LABEL_123;
            }
            v54 = v55;
            v55 = (_QWORD *)v55[2];
          }
LABEL_123:
          if ( (_QWORD *)a2[3] != v54 || (_QWORD *)v51 != v53 )
          {
            if ( v53 != v54 )
            {
              do
              {
                v92 = v54;
                v54 = (_QWORD *)sub_220EF30(v54);
                v93 = sub_220F330(v92, v51);
                j_j___libc_free_0(v93, 40);
                --a2[5];
              }
              while ( v53 != v54 );
              v50 = v114;
            }
            goto LABEL_65;
          }
        }
        else
        {
          v53 = (_QWORD *)v51;
          v56 = 1;
LABEL_62:
          if ( (_QWORD *)a2[3] != v53 || !v56 )
            goto LABEL_65;
        }
        sub_1C97470(a2[2]);
        a2[3] = v51;
        v50 = v114;
        a2[2] = 0;
        a2[4] = v51;
        a2[5] = 0;
LABEL_65:
        v110 += 8;
        result = v110;
      }
      while ( v108 != v110 );
    }
    if ( v50 )
      return j_j___libc_free_0(v50, &v116[-v50]);
  }
  return result;
}
