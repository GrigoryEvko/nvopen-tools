// Function: sub_1DCFC50
// Address: 0x1dcfc50
//
void __fastcall sub_1DCFC50(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int16 v8; // r12
  __int64 v9; // rdx
  _WORD *v10; // r15
  unsigned int v11; // r14d
  _DWORD *v12; // rcx
  _DWORD *v13; // rax
  __int16 v14; // ax
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rdx
  __int16 *v18; // rax
  __int16 v19; // dx
  __int16 *v20; // rax
  unsigned __int16 v21; // r15
  bool v22; // zf
  __int16 *v23; // rdx
  __int16 *k; // r14
  _DWORD *v25; // rax
  _BYTE *v26; // rdx
  __int16 v27; // ax
  __int64 v28; // rax
  int *v29; // r13
  unsigned int v30; // edx
  int *v31; // rax
  _BOOL4 v32; // r8d
  __int64 v33; // rax
  __int64 v34; // r12
  unsigned int *v35; // r14
  unsigned int v36; // r15d
  int *j; // r13
  unsigned int v38; // edx
  int *v39; // rax
  _BOOL4 v40; // r12d
  __int64 v41; // rax
  int v42; // eax
  int *v43; // r8
  unsigned int v44; // r14d
  unsigned int v45; // edx
  int *v46; // rax
  _BOOL4 v47; // r13d
  __int64 v48; // rax
  __int64 v49; // rax
  int *v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rdx
  _WORD *v53; // rax
  __int16 *v54; // r15
  __int16 *v55; // r13
  _DWORD *v56; // rax
  _DWORD *v57; // rcx
  __int16 v58; // ax
  __int64 v59; // rax
  int *v60; // rdi
  __int64 v61; // rsi
  __int64 v62; // rcx
  __int64 v63; // rdx
  unsigned __int16 v64; // r12
  __int64 v65; // r9
  __int16 *v66; // r15
  unsigned int v67; // r14d
  _DWORD *v68; // rcx
  _DWORD *v69; // rax
  __int16 v70; // ax
  int *v71; // r13
  unsigned int v72; // edx
  int *v73; // rax
  _BOOL4 v74; // r8d
  __int64 v75; // rax
  __int64 v76; // r12
  unsigned int *v77; // r14
  unsigned int v78; // r15d
  int *i; // r13
  unsigned int v80; // edx
  int *v81; // rax
  _BOOL4 v82; // r12d
  __int64 v83; // rax
  int v84; // eax
  int *v85; // r8
  unsigned int v86; // r14d
  unsigned int v87; // edx
  int *v88; // rax
  _BOOL4 v89; // r13d
  __int64 v90; // rax
  __int64 v92; // [rsp+18h] [rbp-128h]
  __int16 *v94; // [rsp+28h] [rbp-118h]
  _WORD *v96; // [rsp+38h] [rbp-108h]
  _BOOL4 v97; // [rsp+38h] [rbp-108h]
  unsigned int v98; // [rsp+38h] [rbp-108h]
  int *v99; // [rsp+38h] [rbp-108h]
  int *v100; // [rsp+38h] [rbp-108h]
  unsigned __int16 v102; // [rsp+44h] [rbp-FCh]
  unsigned __int16 v103; // [rsp+46h] [rbp-FAh]
  unsigned __int16 v104; // [rsp+46h] [rbp-FAh]
  _BOOL4 v105; // [rsp+48h] [rbp-F8h]
  unsigned int v106; // [rsp+48h] [rbp-F8h]
  int *v107; // [rsp+48h] [rbp-F8h]
  int *v108; // [rsp+48h] [rbp-F8h]
  __int16 *v109; // [rsp+48h] [rbp-F8h]
  _BYTE *v110; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v111; // [rsp+58h] [rbp-E8h]
  _BYTE v112[136]; // [rsp+60h] [rbp-E0h] BYREF
  int v113; // [rsp+E8h] [rbp-58h] BYREF
  int *v114; // [rsp+F0h] [rbp-50h]
  int *v115; // [rsp+F8h] [rbp-48h]
  int *v116; // [rsp+100h] [rbp-40h]
  __int64 v117; // [rsp+108h] [rbp-38h]

  v110 = v112;
  v111 = 0x2000000000LL;
  v6 = a1[46];
  v113 = 0;
  v114 = 0;
  v115 = &v113;
  v116 = &v113;
  v117 = 0;
  v7 = a1[45];
  v92 = a2;
  if ( !*(_QWORD *)(v6 + 8LL * a2) && !*(_QWORD *)(a1[49] + 8LL * a2) )
  {
    if ( !v7 )
      BUG();
    v53 = (_WORD *)(*(_QWORD *)(v7 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v7 + 8) + 24LL * a2 + 4));
    v54 = v53 + 1;
    if ( !*v53 )
      v54 = 0;
    v104 = *v53 + a2;
    v55 = v54;
LABEL_101:
    v109 = v55;
    while ( 1 )
    {
      if ( !v109 )
        goto LABEL_13;
      if ( v117 )
      {
        v59 = (__int64)v114;
        if ( v114 )
        {
          v60 = &v113;
          do
          {
            while ( 1 )
            {
              v61 = *(_QWORD *)(v59 + 16);
              v62 = *(_QWORD *)(v59 + 24);
              if ( (unsigned int)v104 <= *(_DWORD *)(v59 + 32) )
                break;
              v59 = *(_QWORD *)(v59 + 24);
              if ( !v62 )
                goto LABEL_120;
            }
            v60 = (int *)v59;
            v59 = *(_QWORD *)(v59 + 16);
          }
          while ( v61 );
LABEL_120:
          if ( v60 != &v113 && v104 >= (unsigned int)v60[8] )
            goto LABEL_111;
        }
      }
      else
      {
        v56 = v110;
        v57 = &v110[4 * (unsigned int)v111];
        if ( v110 != (_BYTE *)v57 )
        {
          while ( v104 != *v56 )
          {
            if ( v57 == ++v56 )
              goto LABEL_109;
          }
          if ( v56 != v57 )
            goto LABEL_111;
        }
      }
LABEL_109:
      if ( *(_QWORD *)(a1[46] + 8LL * v104) || *(_QWORD *)(a1[49] + 8LL * v104) )
      {
        v63 = a1[45];
        if ( !v63 )
          BUG();
        v64 = v104;
        v65 = *(_QWORD *)(v63 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v63 + 8) + 24LL * v104 + 4);
        v66 = (__int16 *)v65;
        while ( 1 )
        {
          if ( !v66 )
            goto LABEL_111;
          v67 = v64;
          if ( v117 )
            break;
          v68 = &v110[4 * (unsigned int)v111];
          if ( v110 != (_BYTE *)v68 )
          {
            v69 = v110;
            while ( v64 != *v69 )
            {
              if ( v68 == ++v69 )
                goto LABEL_146;
            }
            if ( v68 != v69 )
              goto LABEL_133;
          }
LABEL_146:
          if ( (unsigned int)v111 <= 0x1FuLL )
          {
            if ( (unsigned int)v111 >= HIDWORD(v111) )
            {
              sub_16CD150((__int64)&v110, v112, 0, 4, (int)v114, v65);
              v68 = &v110[4 * (unsigned int)v111];
            }
            *v68 = v64;
            LODWORD(v111) = v111 + 1;
            goto LABEL_133;
          }
          v102 = v64;
          v76 = (__int64)v114;
          v98 = v67;
          v77 = (unsigned int *)&v110[4 * (unsigned int)v111 - 4];
          v94 = v66;
          if ( v114 )
          {
LABEL_148:
            v78 = *v77;
            for ( i = (int *)v76; ; i = v81 )
            {
              v80 = i[8];
              v81 = (int *)*((_QWORD *)i + 3);
              if ( v78 < v80 )
                v81 = (int *)*((_QWORD *)i + 2);
              if ( !v81 )
                break;
            }
            if ( v78 < v80 )
            {
              if ( v115 != i )
                goto LABEL_163;
            }
            else if ( v78 <= v80 )
            {
              goto LABEL_158;
            }
LABEL_155:
            v82 = 1;
            if ( i != &v113 )
              v82 = v78 < i[8];
LABEL_157:
            v83 = sub_22077B0(40);
            *(_DWORD *)(v83 + 32) = *v77;
            sub_220F040(v82, v83, i, &v113);
            ++v117;
            v76 = (__int64)v114;
            goto LABEL_158;
          }
          while ( 1 )
          {
            i = &v113;
            if ( v115 == &v113 )
            {
              v82 = 1;
              goto LABEL_157;
            }
            v78 = *v77;
LABEL_163:
            if ( v78 > *(_DWORD *)(sub_220EF80(i) + 32) )
              goto LABEL_155;
LABEL_158:
            v22 = (_DWORD)v111 == 1;
            v84 = v111 - 1;
            LODWORD(v111) = v111 - 1;
            if ( v22 )
              break;
            v77 = (unsigned int *)&v110[4 * v84 - 4];
            if ( v76 )
              goto LABEL_148;
          }
          v85 = (int *)v76;
          v86 = v98;
          v66 = v94;
          v64 = v102;
          if ( v85 )
          {
            while ( 1 )
            {
              v87 = v85[8];
              v88 = (int *)*((_QWORD *)v85 + 3);
              if ( v98 < v87 )
                v88 = (int *)*((_QWORD *)v85 + 2);
              if ( !v88 )
                break;
              v85 = v88;
            }
            if ( v98 < v87 )
            {
              if ( v115 == v85 )
                goto LABEL_173;
LABEL_184:
              v100 = v85;
              if ( v86 <= *(_DWORD *)(sub_220EF80(v85) + 32) )
                goto LABEL_133;
              v85 = v100;
              if ( !v100 )
                goto LABEL_133;
              v89 = 1;
              if ( v100 == &v113 )
                goto LABEL_174;
            }
            else
            {
              if ( v98 <= v87 )
                goto LABEL_133;
LABEL_173:
              v89 = 1;
              if ( v85 == &v113 )
                goto LABEL_174;
            }
            v89 = v86 < v85[8];
            goto LABEL_174;
          }
          v85 = &v113;
          if ( v115 != &v113 )
            goto LABEL_184;
          v89 = 1;
LABEL_174:
          v99 = v85;
          v90 = sub_22077B0(40);
          *(_DWORD *)(v90 + 32) = v86;
          sub_220F040(v89, v90, v99, &v113);
          ++v117;
LABEL_133:
          v70 = *v66++;
          if ( v70 )
            v64 += v70;
          else
            v66 = 0;
        }
        v71 = v114;
        if ( v114 )
        {
          while ( 1 )
          {
            v72 = v71[8];
            v73 = (int *)*((_QWORD *)v71 + 3);
            if ( v64 < v72 )
              v73 = (int *)*((_QWORD *)v71 + 2);
            if ( !v73 )
              break;
            v71 = v73;
          }
          if ( v64 < v72 )
          {
            if ( v115 != v71 )
              goto LABEL_176;
          }
          else if ( v64 <= v72 )
          {
            goto LABEL_133;
          }
          v74 = 1;
          if ( v71 != &v113 )
LABEL_179:
            v74 = v64 < (unsigned int)v71[8];
        }
        else
        {
          v71 = &v113;
          if ( v115 == &v113 )
          {
            v74 = 1;
            goto LABEL_145;
          }
LABEL_176:
          if ( *(_DWORD *)(sub_220EF80(v71) + 32) >= (unsigned int)v64 || !v71 )
            goto LABEL_133;
          v74 = 1;
          if ( v71 != &v113 )
            goto LABEL_179;
        }
LABEL_145:
        v97 = v74;
        v75 = sub_22077B0(40);
        *(_DWORD *)(v75 + 32) = v64;
        sub_220F040(v97, v75, v71, &v113);
        ++v117;
        goto LABEL_133;
      }
LABEL_111:
      v58 = *v109++;
      if ( !v58 )
      {
        v55 = 0;
        goto LABEL_101;
      }
      v104 += v58;
    }
  }
  if ( !v7 )
    BUG();
  v8 = a2;
  v9 = *(_QWORD *)(v7 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v7 + 8) + 24LL * a2 + 4);
LABEL_4:
  v10 = (_WORD *)v9;
  if ( v9 )
  {
LABEL_5:
    v11 = v8;
    if ( !v117 )
    {
      v12 = &v110[4 * (unsigned int)v111];
      if ( v110 != (_BYTE *)v12 )
      {
        v13 = v110;
        while ( v8 != *v13 )
        {
          if ( v12 == ++v13 )
            goto LABEL_43;
        }
        if ( v12 != v13 )
          goto LABEL_11;
      }
LABEL_43:
      if ( (unsigned int)v111 <= 0x1FuLL )
      {
        if ( (unsigned int)v111 >= HIDWORD(v111) )
        {
          sub_16CD150((__int64)&v110, v112, 0, 4, (int)v114, a6);
          v12 = &v110[4 * (unsigned int)v111];
        }
        *v12 = v8;
        LODWORD(v111) = v111 + 1;
        goto LABEL_11;
      }
      v103 = v8;
      v34 = (__int64)v114;
      v106 = v11;
      v35 = (unsigned int *)&v110[4 * (unsigned int)v111 - 4];
      v96 = v10;
      if ( v114 )
      {
LABEL_45:
        v36 = *v35;
        for ( j = (int *)v34; ; j = v39 )
        {
          v38 = j[8];
          v39 = (int *)*((_QWORD *)j + 3);
          if ( v36 < v38 )
            v39 = (int *)*((_QWORD *)j + 2);
          if ( !v39 )
            break;
        }
        if ( v36 < v38 )
        {
          if ( j != v115 )
            goto LABEL_60;
        }
        else if ( v36 <= v38 )
        {
          goto LABEL_55;
        }
LABEL_52:
        v40 = 1;
        if ( j != &v113 )
          v40 = v36 < j[8];
      }
      else
      {
        while ( 1 )
        {
          j = &v113;
          if ( v115 == &v113 )
            break;
          v36 = *v35;
LABEL_60:
          if ( v36 > *(_DWORD *)(sub_220EF80(j) + 32) )
            goto LABEL_52;
LABEL_55:
          v22 = (_DWORD)v111 == 1;
          v42 = v111 - 1;
          LODWORD(v111) = v111 - 1;
          if ( v22 )
          {
            v43 = (int *)v34;
            v44 = v106;
            v8 = v103;
            v10 = v96;
            if ( !v43 )
            {
              v43 = &v113;
              if ( v115 != &v113 )
                goto LABEL_89;
              v47 = 1;
LABEL_71:
              v107 = v43;
              v48 = sub_22077B0(40);
              *(_DWORD *)(v48 + 32) = v44;
              sub_220F040(v47, v48, v107, &v113);
              ++v117;
LABEL_11:
              v14 = *v10;
              v9 = 0;
              ++v10;
              if ( !v14 )
                goto LABEL_4;
              v8 += v14;
              if ( !v10 )
                goto LABEL_13;
              goto LABEL_5;
            }
            while ( 1 )
            {
              v45 = v43[8];
              v46 = (int *)*((_QWORD *)v43 + 3);
              if ( v106 < v45 )
                v46 = (int *)*((_QWORD *)v43 + 2);
              if ( !v46 )
                break;
              v43 = v46;
            }
            if ( v106 < v45 )
            {
              if ( v115 == v43 )
                goto LABEL_70;
LABEL_89:
              v108 = v43;
              if ( v44 <= *(_DWORD *)(sub_220EF80(v43) + 32) )
                goto LABEL_11;
              v43 = v108;
              if ( !v108 )
                goto LABEL_11;
              v47 = 1;
              if ( v108 == &v113 )
                goto LABEL_71;
            }
            else
            {
              if ( v106 <= v45 )
                goto LABEL_11;
LABEL_70:
              v47 = 1;
              if ( v43 == &v113 )
                goto LABEL_71;
            }
            v47 = v44 < v43[8];
            goto LABEL_71;
          }
          v35 = (unsigned int *)&v110[4 * v42 - 4];
          if ( v34 )
            goto LABEL_45;
        }
        v40 = 1;
      }
      v41 = sub_22077B0(40);
      *(_DWORD *)(v41 + 32) = *v35;
      sub_220F040(v40, v41, j, &v113);
      ++v117;
      v34 = (__int64)v114;
      goto LABEL_55;
    }
    v29 = v114;
    if ( !v114 )
    {
      v29 = &v113;
      if ( v115 == &v113 )
      {
        v32 = 1;
        goto LABEL_42;
      }
LABEL_73:
      if ( (unsigned int)v8 <= *(_DWORD *)(sub_220EF80(v29) + 32) || !v29 )
        goto LABEL_11;
      v32 = 1;
      if ( v29 != &v113 )
        goto LABEL_76;
      goto LABEL_42;
    }
    while ( 1 )
    {
      v30 = v29[8];
      v31 = (int *)*((_QWORD *)v29 + 3);
      if ( v8 < v30 )
        v31 = (int *)*((_QWORD *)v29 + 2);
      if ( !v31 )
        break;
      v29 = v31;
    }
    if ( v8 < v30 )
    {
      if ( v29 != v115 )
        goto LABEL_73;
    }
    else if ( v8 <= v30 )
    {
      goto LABEL_11;
    }
    v32 = 1;
    if ( v29 != &v113 )
LABEL_76:
      v32 = v8 < (unsigned int)v29[8];
LABEL_42:
    v105 = v32;
    v33 = sub_22077B0(40);
    *(_DWORD *)(v33 + 32) = v8;
    sub_220F040(v105, v33, v29, &v113);
    ++v117;
    goto LABEL_11;
  }
LABEL_13:
  sub_1DCE640((__int64)a1, a2, a3);
  v17 = a1[45];
  if ( !v17 )
    BUG();
  v18 = (__int16 *)(*(_QWORD *)(v17 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v17 + 8) + 24 * v92 + 4));
  v19 = *v18;
  v20 = v18 + 1;
  v21 = v19 + a2;
  v22 = v19 == 0;
  v23 = 0;
  if ( !v22 )
    v23 = v20;
LABEL_16:
  for ( k = v23; k; v21 += v27 )
  {
    if ( v117 )
    {
      v49 = (__int64)v114;
      if ( !v114 )
        goto LABEL_24;
      v50 = &v113;
      do
      {
        while ( 1 )
        {
          v51 = *(_QWORD *)(v49 + 16);
          v52 = *(_QWORD *)(v49 + 24);
          if ( (unsigned int)v21 <= *(_DWORD *)(v49 + 32) )
            break;
          v49 = *(_QWORD *)(v49 + 24);
          if ( !v52 )
            goto LABEL_85;
        }
        v50 = (int *)v49;
        v49 = *(_QWORD *)(v49 + 16);
      }
      while ( v51 );
LABEL_85:
      if ( v50 == &v113 || v21 < (unsigned int)v50[8] )
        goto LABEL_24;
    }
    else
    {
      v25 = v110;
      v26 = &v110[4 * (unsigned int)v111];
      if ( v110 == v26 )
        goto LABEL_24;
      while ( v21 != *v25 )
      {
        if ( v26 == (_BYTE *)++v25 )
          goto LABEL_24;
      }
      if ( v26 == (_BYTE *)v25 )
        goto LABEL_24;
    }
    sub_1DCE640((__int64)a1, v21, a3);
LABEL_24:
    v27 = *k;
    v23 = 0;
    ++k;
    if ( !v27 )
      goto LABEL_16;
  }
  if ( a3 )
  {
    v28 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v28 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 4, v15, v16);
      v28 = *(unsigned int *)(a4 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a4 + 4 * v28) = a2;
    ++*(_DWORD *)(a4 + 8);
  }
  sub_1DCADB0((__int64)v114);
  if ( v110 != v112 )
    _libc_free((unsigned __int64)v110);
}
