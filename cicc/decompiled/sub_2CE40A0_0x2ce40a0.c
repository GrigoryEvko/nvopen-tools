// Function: sub_2CE40A0
// Address: 0x2ce40a0
//
void __fastcall sub_2CE40A0(_QWORD *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  unsigned __int64 *v5; // rcx
  _QWORD *v6; // rbx
  _QWORD *v7; // rdx
  _QWORD *v8; // r8
  unsigned __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int64 **v16; // r14
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rcx
  int *v20; // rdx
  __int64 v21; // rax
  int *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  __int64 v27; // r12
  _QWORD *v28; // rdx
  unsigned __int64 v29; // rcx
  _QWORD *v30; // r8
  _QWORD *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rsi
  _QWORD *v34; // r13
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rcx
  int *v40; // rdx
  __int64 v41; // rax
  int *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 v45; // r15
  unsigned __int64 v46; // r13
  unsigned __int64 v47; // rdi
  __int64 v48; // r8
  unsigned __int64 v49; // rdi
  __int64 v50; // rax
  unsigned __int64 *v51; // r14
  unsigned __int64 v52; // rdx
  int *v53; // r12
  int *v54; // rbx
  int *v55; // rax
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
  __int64 v66; // r8
  unsigned __int64 **v67; // r13
  int v68; // r14d
  _QWORD *v69; // rax
  unsigned __int64 v70; // r11
  int v71; // ecx
  __int64 v72; // r10
  __int64 v73; // rsi
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // r8
  _QWORD *v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // rcx
  unsigned int v80; // r8d
  __int64 v81; // rsi
  __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // rax
  char v85; // al
  _BYTE *v86; // rsi
  __int64 v87; // r8
  int *v88; // rsi
  __int64 v89; // r11
  __int64 v90; // r9
  __int64 v91; // rsi
  int *v92; // rdi
  int *v93; // rax
  _BYTE *v94; // rsi
  _QWORD *v96; // [rsp+8h] [rbp-138h]
  unsigned __int64 v97; // [rsp+18h] [rbp-128h]
  __int64 v98; // [rsp+28h] [rbp-118h]
  unsigned int v99; // [rsp+34h] [rbp-10Ch]
  _QWORD *v102; // [rsp+68h] [rbp-D8h]
  _QWORD *v104; // [rsp+78h] [rbp-C8h]
  __int64 v105; // [rsp+80h] [rbp-C0h]
  __int64 v106; // [rsp+88h] [rbp-B8h]
  unsigned __int64 *v107; // [rsp+90h] [rbp-B0h]
  __int64 v108; // [rsp+90h] [rbp-B0h]
  _QWORD *v109; // [rsp+98h] [rbp-A8h]
  __int64 v110; // [rsp+98h] [rbp-A8h]
  unsigned __int64 v111; // [rsp+A8h] [rbp-98h] BYREF
  unsigned __int64 v112; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v113; // [rsp+B8h] [rbp-88h] BYREF
  unsigned __int64 v114; // [rsp+C0h] [rbp-80h] BYREF
  _BYTE *v115; // [rsp+C8h] [rbp-78h]
  _BYTE *v116; // [rsp+D0h] [rbp-70h]
  unsigned __int64 *v117; // [rsp+E0h] [rbp-60h] BYREF
  int v118; // [rsp+E8h] [rbp-58h] BYREF
  unsigned __int64 v119; // [rsp+F0h] [rbp-50h]
  int *v120; // [rsp+F8h] [rbp-48h]
  int *v121; // [rsp+100h] [rbp-40h]
  __int64 v122; // [rsp+108h] [rbp-38h]

  v5 = (unsigned __int64 *)a2[3];
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v105 = (__int64)v5;
  v107 = a2 + 1;
  if ( v5 != a2 + 1 )
  {
    v6 = a1;
    v109 = a1 + 57;
    v102 = a5 + 1;
    while ( 1 )
    {
      v7 = (_QWORD *)v6[58];
      if ( v7 )
      {
        v8 = v109;
        v9 = *(_QWORD *)(v105 + 32);
        v10 = (_QWORD *)v6[58];
        do
        {
          while ( 1 )
          {
            v11 = v10[2];
            v12 = v10[3];
            if ( v10[4] >= v9 )
              break;
            v10 = (_QWORD *)v10[3];
            if ( !v12 )
              goto LABEL_8;
          }
          v8 = v10;
          v10 = (_QWORD *)v10[2];
        }
        while ( v11 );
LABEL_8:
        if ( v8 != v109 )
        {
          v13 = v109;
          if ( v8[4] <= v9 )
          {
            do
            {
              while ( 1 )
              {
                v14 = v7[2];
                v15 = v7[3];
                if ( v7[4] >= v9 )
                  break;
                v7 = (_QWORD *)v7[3];
                if ( !v15 )
                  goto LABEL_14;
              }
              v13 = v7;
              v7 = (_QWORD *)v7[2];
            }
            while ( v14 );
LABEL_14:
            if ( v13 == v109 || (v16 = &v117, v13[4] > v9) )
            {
              v16 = &v117;
              v117 = (unsigned __int64 *)(v105 + 32);
              v13 = (_QWORD *)sub_2CE3350(v6 + 56, (__int64)v13, &v117);
            }
            v118 = 0;
            v119 = 0;
            v120 = &v118;
            v121 = &v118;
            v122 = 0;
            v17 = v13[7];
            v104 = v6 + 38;
            if ( v17 )
            {
              v18 = sub_2CDD8A0(v17, (__int64)&v118);
              v19 = v18;
              do
              {
                v20 = (int *)v18;
                v18 = *(_QWORD *)(v18 + 16);
              }
              while ( v18 );
              v120 = v20;
              v21 = v19;
              do
              {
                v22 = (int *)v21;
                v21 = *(_QWORD *)(v21 + 24);
              }
              while ( v21 );
              v121 = v22;
              v23 = v13[10];
              v119 = v19;
              v122 = v23;
            }
            v24 = sub_2CE2B60((__int64)v6, (__int64)&v117, a3, a4, v104);
            v25 = v119;
            v106 = v24;
            while ( v25 )
            {
              sub_2CDF0D0(*(_QWORD *)(v25 + 24));
              v26 = v25;
              v25 = *(_QWORD *)(v25 + 16);
              j_j___libc_free_0(v26);
            }
            if ( v106 )
              break;
          }
        }
      }
LABEL_52:
      v105 = sub_220EF30(v105);
      if ( v107 == (unsigned __int64 *)v105 )
        goto LABEL_53;
    }
    v27 = sub_220EF30(v105);
    if ( v107 != (unsigned __int64 *)v27 )
    {
      while ( 1 )
      {
        v28 = (_QWORD *)v6[58];
        if ( !v28 )
          goto LABEL_51;
        v29 = *(_QWORD *)(v27 + 32);
        v30 = v109;
        v31 = (_QWORD *)v6[58];
        do
        {
          while ( 1 )
          {
            v32 = v31[2];
            v33 = v31[3];
            if ( v31[4] >= v29 )
              break;
            v31 = (_QWORD *)v31[3];
            if ( !v33 )
              goto LABEL_32;
          }
          v30 = v31;
          v31 = (_QWORD *)v31[2];
        }
        while ( v32 );
LABEL_32:
        if ( v30 == v109 )
          goto LABEL_51;
        v34 = v109;
        if ( v30[4] > v29 )
          goto LABEL_51;
        do
        {
          while ( 1 )
          {
            v35 = v28[2];
            v36 = v28[3];
            if ( v28[4] >= v29 )
              break;
            v28 = (_QWORD *)v28[3];
            if ( !v36 )
              goto LABEL_38;
          }
          v34 = v28;
          v28 = (_QWORD *)v28[2];
        }
        while ( v35 );
LABEL_38:
        if ( v34 == v109 || v34[4] > v29 )
        {
          v117 = (unsigned __int64 *)(v27 + 32);
          v34 = (_QWORD *)sub_2CE3350(v6 + 56, (__int64)v34, v16);
        }
        v118 = 0;
        v119 = 0;
        v120 = &v118;
        v121 = &v118;
        v122 = 0;
        v37 = v34[7];
        if ( v37 )
        {
          v38 = sub_2CDD8A0(v37, (__int64)&v118);
          v39 = v38;
          do
          {
            v40 = (int *)v38;
            v38 = *(_QWORD *)(v38 + 16);
          }
          while ( v38 );
          v120 = v40;
          v41 = v39;
          do
          {
            v42 = (int *)v41;
            v41 = *(_QWORD *)(v41 + 24);
          }
          while ( v41 );
          v121 = v42;
          v43 = v34[10];
          v119 = v39;
          v122 = v43;
        }
        v44 = sub_2CE2B60((__int64)v6, (__int64)v16, a3, a4, v104);
        v45 = v119;
        v46 = v44;
        while ( v45 )
        {
          sub_2CDF0D0(*(_QWORD *)(v45 + 24));
          v47 = v45;
          v45 = *(_QWORD *)(v45 + 16);
          j_j___libc_free_0(v47);
        }
        if ( !v46 )
          goto LABEL_51;
        v48 = *(_QWORD *)(v46 + 40);
        if ( v48 != *(_QWORD *)(v106 + 40) )
          goto LABEL_51;
        v112 = v106;
        v111 = v46;
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
              if ( v59[4] >= v46 )
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
          if ( v102 != v58 && v58[4] <= v46 )
            goto LABEL_91;
          v62 = *(_QWORD *)(v48 + 56);
          v63 = v48 + 48;
          if ( v62 == v63 )
            goto LABEL_91;
        }
        else
        {
          v62 = *(_QWORD *)(v48 + 56);
          v63 = v48 + 48;
          if ( v62 == v63 )
            goto LABEL_112;
        }
        v64 = v62;
        v96 = v6;
        v65 = v63;
        v66 = (__int64)v102;
        v97 = v46;
        v67 = v16;
        v68 = 0;
        do
        {
          if ( !v64 )
            BUG();
          if ( *(_BYTE *)(v64 - 24) == 62 )
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
              v72 = (__int64)v69;
              v69 = (_QWORD *)v69[2];
            }
            while ( v73 );
LABEL_85:
            if ( v66 == v72 || *(_QWORD *)(v72 + 32) > v70 )
            {
LABEL_87:
              v98 = v66;
              v117 = (unsigned __int64 *)&v113;
              v75 = sub_2CE3FF0(a5, v72, v67);
              v66 = v98;
              v71 = v68 + 1;
              v72 = v75;
            }
            *(_DWORD *)(v72 + 40) = v68;
            v68 = v71;
          }
          v64 = *(_QWORD *)(v64 + 8);
        }
        while ( v64 != v65 );
        v16 = v67;
        v6 = v96;
        v46 = v97;
        v57 = (_QWORD *)a5[2];
        if ( !v57 )
        {
LABEL_112:
          v76 = (__int64)v102;
          goto LABEL_113;
        }
LABEL_91:
        v76 = (__int64)v102;
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
          v76 = (__int64)v77;
          v77 = (_QWORD *)v77[2];
        }
        while ( v78 );
LABEL_95:
        if ( v102 != (_QWORD *)v76 && *(_QWORD *)(v76 + 32) <= v111 )
        {
          v80 = *(_DWORD *)(v76 + 40);
LABEL_98:
          v81 = (__int64)v102;
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
            v81 = (__int64)v57;
            v57 = (_QWORD *)v57[2];
          }
          while ( v82 );
LABEL_102:
          if ( v102 != (_QWORD *)v81 && *(_QWORD *)(v81 + 32) <= v112 )
            goto LABEL_105;
          goto LABEL_104;
        }
LABEL_113:
        v117 = &v111;
        v87 = sub_2CE3FF0(a5, v76, v16);
        v57 = (_QWORD *)a5[2];
        v80 = *(_DWORD *)(v87 + 40);
        if ( v57 )
          goto LABEL_98;
        v81 = (__int64)v102;
LABEL_104:
        v99 = v80;
        v117 = &v112;
        v84 = sub_2CE3FF0(a5, v81, v16);
        v80 = v99;
        v81 = v84;
LABEL_105:
        if ( *(_DWORD *)(v81 + 40) <= v80 )
        {
          if ( !(unsigned __int8)sub_2CE2D50(v6, v106, v46, a3) )
          {
            v94 = v115;
            if ( v115 == v116 )
            {
              sub_9281F0((__int64)&v114, v115, (_QWORD *)(v105 + 32));
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
          v85 = sub_2CE2D50(v6, v46, v106, a3);
          v106 = v46;
          if ( !v85 )
          {
            v86 = v115;
            if ( v115 == v116 )
            {
              sub_9281F0((__int64)&v114, v115, (_QWORD *)(v27 + 32));
            }
            else
            {
              if ( v115 )
              {
                *(_QWORD *)v115 = *(_QWORD *)(v27 + 32);
                v86 = v115;
              }
              v106 = v46;
              v115 = v86 + 8;
            }
          }
        }
LABEL_51:
        v27 = sub_220EF30(v27);
        if ( v107 == (unsigned __int64 *)v27 )
          goto LABEL_52;
      }
    }
LABEL_53:
    v49 = v114;
    v50 = (__int64)&v115[-v114] >> 3;
    if ( (_DWORD)v50 )
    {
      v51 = v107;
      v110 = 0;
      v108 = 8LL * (unsigned int)v50;
      do
      {
        if ( a2[2] )
        {
          v52 = *(_QWORD *)(v49 + v110);
          v53 = (int *)v51;
          v54 = (int *)a2[2];
          while ( 1 )
          {
            while ( *((_QWORD *)v54 + 4) < v52 )
            {
              v54 = (int *)*((_QWORD *)v54 + 3);
              if ( !v54 )
                goto LABEL_61;
            }
            v55 = (int *)*((_QWORD *)v54 + 2);
            if ( *((_QWORD *)v54 + 4) <= v52 )
              break;
            v53 = v54;
            v54 = (int *)*((_QWORD *)v54 + 2);
            if ( !v55 )
            {
LABEL_61:
              v56 = v51 == (unsigned __int64 *)v53;
              goto LABEL_62;
            }
          }
          v88 = (int *)*((_QWORD *)v54 + 3);
          if ( v88 )
          {
            do
            {
              while ( 1 )
              {
                v89 = *((_QWORD *)v88 + 2);
                v90 = *((_QWORD *)v88 + 3);
                if ( *((_QWORD *)v88 + 4) > v52 )
                  break;
                v88 = (int *)*((_QWORD *)v88 + 3);
                if ( !v90 )
                  goto LABEL_120;
              }
              v53 = v88;
              v88 = (int *)*((_QWORD *)v88 + 2);
            }
            while ( v89 );
          }
LABEL_120:
          while ( v55 )
          {
            while ( 1 )
            {
              v91 = *((_QWORD *)v55 + 3);
              if ( *((_QWORD *)v55 + 4) >= v52 )
                break;
              v55 = (int *)*((_QWORD *)v55 + 3);
              if ( !v91 )
                goto LABEL_123;
            }
            v54 = v55;
            v55 = (int *)*((_QWORD *)v55 + 2);
          }
LABEL_123:
          if ( (int *)a2[3] != v54 || v51 != (unsigned __int64 *)v53 )
          {
            if ( v53 != v54 )
            {
              do
              {
                v92 = v54;
                v54 = (int *)sub_220EF30((__int64)v54);
                v93 = sub_220F330(v92, v51);
                j_j___libc_free_0((unsigned __int64)v93);
                --a2[5];
              }
              while ( v53 != v54 );
              v49 = v114;
            }
            goto LABEL_65;
          }
        }
        else
        {
          v53 = (int *)v51;
          v56 = 1;
LABEL_62:
          if ( (int *)a2[3] != v53 || !v56 )
            goto LABEL_65;
        }
        sub_2CDF380(a2[2]);
        a2[3] = (unsigned __int64)v51;
        v49 = v114;
        a2[2] = 0;
        a2[4] = (unsigned __int64)v51;
        a2[5] = 0;
LABEL_65:
        v110 += 8;
      }
      while ( v108 != v110 );
    }
    if ( v49 )
      j_j___libc_free_0(v49);
  }
}
