// Function: sub_2416F70
// Address: 0x2416f70
//
__int64 __fastcall sub_2416F70(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, __int16 a5)
{
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  bool v8; // dl
  unsigned __int64 v9; // r14
  char v10; // dl
  __int64 v11; // rcx
  __int64 v12; // rsi
  _QWORD *v13; // rbx
  int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rcx
  unsigned int v17; // eax
  char *v18; // rcx
  _QWORD *v19; // r13
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // r12
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rbx
  _QWORD *v25; // rbx
  __int64 v26; // r12
  __int64 v27; // r14
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // r15
  __int64 v30; // r13
  bool v31; // zf
  __int64 *v32; // rax
  int v33; // edx
  unsigned int v34; // esi
  int v35; // edx
  __int64 v36; // rdx
  __int64 v37; // rsi
  _BYTE *v38; // r13
  __int64 v39; // r9
  _BYTE *v40; // r14
  __int64 v41; // rdx
  __int64 v42; // rcx
  _QWORD *v43; // r14
  int *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 *v47; // rdx
  __int64 *v48; // rax
  __int64 *v49; // rdx
  unsigned __int64 v50; // r14
  __int64 v51; // rax
  unsigned __int64 v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // rbx
  __int64 v55; // r14
  char v56; // bl
  __int64 v57; // rax
  _QWORD *v58; // rax
  __int64 *v59; // rdx
  __int64 *v60; // r15
  unsigned int v61; // ebx
  unsigned int v64; // ebx
  int v65; // edi
  int v66; // edi
  _QWORD *v67; // rax
  _QWORD *v68; // rsi
  _QWORD *v69; // rax
  __int64 *v70; // rdx
  __int64 *v71; // r12
  char v72; // r14
  __int64 v73; // rax
  __int64 v74; // r14
  __int64 v75; // rdi
  int v76; // r11d
  __int64 v77; // rsi
  _QWORD *v78; // rdx
  _QWORD *v79; // rax
  __int64 v80; // r8
  unsigned __int64 v81; // rdi
  _QWORD *v82; // rbx
  _QWORD *v83; // r12
  __int64 v84; // rax
  _QWORD *v85; // rax
  _QWORD *v86; // rsi
  __int64 v87; // r9
  _QWORD *v88; // rax
  __int64 *v89; // rdx
  char v90; // r14
  __int64 v91; // rax
  int v92; // ebx
  int v93; // edx
  __int64 v94; // rdx
  int v95; // esi
  __int64 v96; // [rsp+8h] [rbp-188h]
  unsigned __int64 v97; // [rsp+20h] [rbp-170h]
  __int64 v98; // [rsp+28h] [rbp-168h]
  unsigned __int64 v99; // [rsp+28h] [rbp-168h]
  _QWORD *v100; // [rsp+30h] [rbp-160h]
  _QWORD *v101; // [rsp+38h] [rbp-158h]
  _QWORD *v102; // [rsp+40h] [rbp-150h]
  _QWORD *v103; // [rsp+48h] [rbp-148h]
  __int64 v104; // [rsp+48h] [rbp-148h]
  _QWORD *v105; // [rsp+48h] [rbp-148h]
  _QWORD *v107; // [rsp+50h] [rbp-140h]
  __int64 v108; // [rsp+50h] [rbp-140h]
  __int64 v109; // [rsp+50h] [rbp-140h]
  __int64 *v110; // [rsp+50h] [rbp-140h]
  __int64 v112; // [rsp+58h] [rbp-138h]
  char *v113; // [rsp+60h] [rbp-130h] BYREF
  unsigned __int64 v114; // [rsp+68h] [rbp-128h] BYREF
  unsigned __int64 v115; // [rsp+70h] [rbp-120h] BYREF
  unsigned __int64 v116; // [rsp+78h] [rbp-118h]
  _QWORD v117[4]; // [rsp+80h] [rbp-110h] BYREF
  __int64 *v118; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v119; // [rsp+A8h] [rbp-E8h] BYREF
  __int64 v120; // [rsp+B0h] [rbp-E0h]
  __int64 *v121; // [rsp+B8h] [rbp-D8h]
  __int64 *v122; // [rsp+C0h] [rbp-D0h]
  __int64 v123; // [rsp+C8h] [rbp-C8h]
  __int64 *v124[24]; // [rsp+D0h] [rbp-C0h] BYREF

  v6 = a2;
  v7 = *(_QWORD *)(a2 + 8);
  v113 = (char *)a3;
  v114 = a2;
  v8 = *(_BYTE *)a2 == 14;
  if ( (unsigned __int8)(*(_BYTE *)(v7 + 8) - 15) > 1u )
  {
    if ( *(_BYTE *)a2 != 17 )
      goto LABEL_3;
    v64 = *(_DWORD *)(a2 + 32);
    if ( v64 <= 0x40 )
      v8 = *(_QWORD *)(a2 + 24) == 0;
    else
      v8 = v64 == (unsigned int)sub_C444A0(a2 + 24);
  }
  if ( v8 )
    return sub_2415280(a1, (__int64)v113, a4, a5);
LABEL_3:
  v9 = (unsigned __int64)v113;
  v10 = *v113;
  if ( (unsigned __int8)(*(_BYTE *)(*((_QWORD *)v113 + 1) + 8LL) - 15) > 1u )
  {
    if ( v10 == 17 )
    {
      v61 = *((_DWORD *)v113 + 8);
      if ( v61 <= 0x40 ? *((_QWORD *)v113 + 3) == 0 : v61 == (unsigned int)sub_C444A0((__int64)(v113 + 24)) )
        return sub_2415280(a1, v6, a4, a5);
    }
  }
  else if ( v10 == 14 )
  {
    return sub_2415280(a1, v6, a4, a5);
  }
  if ( (char *)a2 == v113 )
    return sub_2415280(a1, v6, a4, a5);
  v11 = *(unsigned int *)(a1 + 472);
  v12 = *(_QWORD *)(a1 + 456);
  v13 = (_QWORD *)(v12 + 56 * v11);
  if ( !(_DWORD)v11 )
  {
LABEL_124:
    v101 = v13;
    goto LABEL_26;
  }
  v14 = *(_DWORD *)(a1 + 472) - 1;
  v15 = v14 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v16 = *(_QWORD *)(v12 + 56LL * v15);
  v101 = (_QWORD *)(v12 + 56LL * v15);
  if ( v6 == v16 )
  {
LABEL_8:
    v17 = v14 & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
    v18 = *(char **)(v12 + 56LL * v17);
    v100 = (_QWORD *)(v12 + 56LL * v17);
    if ( v113 == v18 )
    {
LABEL_9:
      if ( v101 == v13 )
      {
LABEL_97:
        v13 = v101;
        if ( v100 == v101 )
          goto LABEL_26;
        v85 = (_QWORD *)v100[3];
        if ( v85 )
        {
          v86 = v100 + 2;
          do
          {
            if ( v85[4] < v6 )
            {
              v85 = (_QWORD *)v85[3];
            }
            else
            {
              v86 = v85;
              v85 = (_QWORD *)v85[2];
            }
          }
          while ( v85 );
          if ( v100 + 2 != v86 && v86[4] <= v6 )
            return sub_2415280(a1, v9, a4, a5);
        }
        goto LABEL_111;
      }
      v102 = v101 + 2;
      if ( v100 != v13 )
      {
        v19 = (_QWORD *)v100[4];
        v20 = v101[4];
        v103 = v100 + 2;
        if ( v100 + 2 != v19 )
        {
          if ( (_QWORD *)v20 == v101 + 2 )
          {
            if ( v103 != v19 )
              goto LABEL_107;
          }
          else
          {
            v98 = a4;
            v21 = v101[4];
            v97 = v6;
            v22 = v100[4];
            v96 = v21;
            while ( 1 )
            {
              v23 = *(_QWORD *)(v22 + 32);
              v24 = *(_QWORD *)(v21 + 32);
              if ( v23 < v24 )
              {
                v25 = v100 + 2;
                v26 = (__int64)v19;
                v27 = v96;
                v104 = v98;
                v99 = v97;
                goto LABEL_24;
              }
              v21 = sub_220EF30(v21);
              if ( v23 <= v24 )
                v22 = sub_220EF30(v22);
              if ( (_QWORD *)v21 == v102 )
                break;
              if ( v103 == (_QWORD *)v22 )
              {
                a4 = v98;
                v6 = v97;
                return sub_2415280(a1, v6, a4, a5);
              }
            }
            v87 = v22;
            a4 = v98;
            v6 = v97;
            if ( v103 != (_QWORD *)v87 )
            {
              v27 = v96;
              v25 = v100 + 2;
              v99 = v97;
              v26 = (__int64)v19;
              v104 = a4;
              do
              {
LABEL_24:
                v28 = *(_QWORD *)(v27 + 32);
                v29 = *(_QWORD *)(v26 + 32);
                if ( v28 < v29 )
                {
                  v9 = (unsigned __int64)v113;
                  a4 = v104;
                  v6 = v99;
                  v13 = v100;
                  goto LABEL_26;
                }
                v26 = sub_220EF30(v26);
                if ( v28 <= v29 )
                  v27 = sub_220EF30(v27);
              }
              while ( (_QWORD *)v27 != v102 && (_QWORD *)v26 != v25 );
              v20 = v27;
              a4 = v104;
              v9 = (unsigned __int64)v113;
              v6 = v99;
LABEL_107:
              if ( v102 == (_QWORD *)v20 )
                return sub_2415280(a1, v9, a4, a5);
LABEL_111:
              v13 = v100;
              goto LABEL_26;
            }
          }
        }
        return sub_2415280(a1, v6, a4, a5);
      }
      goto LABEL_75;
    }
  }
  else
  {
    v65 = 1;
    while ( v16 != -4096 )
    {
      v15 = v14 & (v65 + v15);
      v16 = *(_QWORD *)(v12 + 56LL * v15);
      if ( v6 == v16 )
      {
        v101 = (_QWORD *)(v12 + 56LL * v15);
        goto LABEL_8;
      }
      ++v65;
    }
    v101 = v13;
    v17 = v14 & (((unsigned int)v113 >> 4) ^ ((unsigned int)v113 >> 9));
    v100 = (_QWORD *)(v12 + 56LL * v17);
    v18 = (char *)*v100;
    if ( v113 == (char *)*v100 )
      goto LABEL_97;
  }
  v66 = 1;
  while ( v18 != (char *)-4096LL )
  {
    v17 = v14 & (v66 + v17);
    v18 = *(char **)(v12 + 56LL * v17);
    if ( v113 == v18 )
    {
      v100 = (_QWORD *)(v12 + 56LL * v17);
      goto LABEL_9;
    }
    ++v66;
  }
  if ( v101 == v13 )
    goto LABEL_124;
  v102 = v101 + 2;
LABEL_75:
  v67 = (_QWORD *)v101[3];
  if ( v67 )
  {
    v68 = v102;
    do
    {
      if ( v67[4] < (unsigned __int64)v113 )
      {
        v67 = (_QWORD *)v67[3];
      }
      else
      {
        v68 = v67;
        v67 = (_QWORD *)v67[2];
      }
    }
    while ( v67 );
    if ( v102 != v68 && v68[4] <= (unsigned __int64)v113 )
      return sub_2415280(a1, v6, a4, a5);
  }
LABEL_26:
  v115 = v6;
  v116 = v9;
  if ( v6 > v9 )
  {
    v115 = v9;
    v116 = v6;
  }
  v30 = a1 + 384;
  v31 = (unsigned __int8)sub_240D670(a1 + 384, (__int64 *)&v115, &v118) == 0;
  v32 = v118;
  if ( v31 )
  {
    v124[0] = v118;
    v33 = *(_DWORD *)(a1 + 400);
    ++*(_QWORD *)(a1 + 384);
    v34 = *(_DWORD *)(a1 + 408);
    v35 = v33 + 1;
    if ( 4 * v35 >= 3 * v34 )
    {
      v34 *= 2;
    }
    else if ( v34 - *(_DWORD *)(a1 + 404) - v35 > v34 >> 3 )
    {
      goto LABEL_31;
    }
    sub_240E560(v30, v34);
    sub_240D670(v30, (__int64 *)&v115, v124);
    v35 = *(_DWORD *)(a1 + 400) + 1;
    v32 = v124[0];
LABEL_31:
    *(_DWORD *)(a1 + 400) = v35;
    if ( *v32 != -4096 || v32[1] != -4096 )
      --*(_DWORD *)(a1 + 404);
    v36 = v115;
    v32[2] = 0;
    v32[3] = 0;
    *v32 = v36;
    v32[1] = v116;
  }
  v105 = v32 + 2;
  v37 = v32[2];
  if ( v37 )
  {
    if ( !a4 )
      BUG();
    if ( (unsigned __int8)sub_B19720(a1 + 16, v37, *(_QWORD *)(a4 + 16)) )
      return v105[1];
    v38 = (_BYTE *)sub_2415280(a1, v114, a4, a5);
    v40 = (_BYTE *)sub_2415280(a1, (__int64)v113, a4, a5);
  }
  else
  {
    v38 = (_BYTE *)sub_2415280(a1, v114, a4, a5);
    v40 = (_BYTE *)sub_2415280(a1, (__int64)v113, a4, a5);
    if ( !a4 )
      BUG();
  }
  sub_2412230((__int64)v124, *(_QWORD *)(a4 + 16), a4, a5, 0, v39, 0, 0);
  *v105 = *(_QWORD *)(a4 + 16);
  LOWORD(v122) = 257;
  v105[1] = sub_A82480((unsigned int **)v124, v38, v40, (__int64)&v118);
  LODWORD(v119) = 0;
  v41 = *(unsigned int *)(a1 + 472);
  v42 = *(_QWORD *)(a1 + 456);
  v120 = 0;
  v121 = &v119;
  LODWORD(a4) = v41;
  v122 = &v119;
  v123 = 0;
  v43 = (_QWORD *)(v42 + 56 * v41);
  if ( v43 == v101 )
  {
    v109 = v42;
    v88 = sub_23FDE00((__int64)&v118, &v114);
    v42 = v109;
    if ( v89 )
    {
      v90 = v88 || v89 == &v119 || v114 < v89[4];
      v110 = v89;
      v91 = sub_22077B0(0x28u);
      *(_QWORD *)(v91 + 32) = v114;
      sub_220F040(v90, v91, v110, &v119);
      ++v123;
      goto LABEL_47;
    }
  }
  else if ( &v118 != v101 + 1 )
  {
    v117[0] = 0;
    v117[2] = &v118;
    v117[1] = 0;
    v44 = (int *)v101[3];
    if ( v44 )
    {
      v45 = sub_240D8D0(v44, (__int64)&v119, v117);
      v46 = v45;
      do
      {
        v47 = (__int64 *)v45;
        v45 = *(_QWORD *)(v45 + 16);
      }
      while ( v45 );
      v121 = v47;
      v48 = (__int64 *)v46;
      do
      {
        v49 = v48;
        v48 = (__int64 *)v48[3];
      }
      while ( v48 );
      v122 = v49;
      v50 = v117[0];
      v51 = v101[6];
      v120 = v46;
      v123 = v51;
      if ( v117[0] )
      {
        do
        {
          sub_240E290(*(_QWORD *)(v50 + 24));
          v52 = v50;
          v50 = *(_QWORD *)(v50 + 16);
          j_j___libc_free_0(v52);
        }
        while ( v50 );
LABEL_47:
        v42 = *(_QWORD *)(a1 + 456);
        LODWORD(a4) = *(_DWORD *)(a1 + 472);
        v43 = (_QWORD *)(v42 + 56LL * (unsigned int)a4);
        goto LABEL_48;
      }
      v42 = *(_QWORD *)(a1 + 456);
      a4 = *(unsigned int *)(a1 + 472);
      v43 = (_QWORD *)(v42 + 56 * a4);
    }
  }
LABEL_48:
  if ( v13 == v43 )
  {
    v108 = v42;
    v69 = sub_23FDE00((__int64)&v118, (unsigned __int64 *)&v113);
    v42 = v108;
    v71 = v70;
    if ( !v70 )
      goto LABEL_90;
    v72 = v69 || v70 == &v119 || (unsigned __int64)v113 < v70[4];
    v73 = sub_22077B0(0x28u);
    *(_QWORD *)(v73 + 32) = v113;
    sub_220F040(v72, v73, v71, &v119);
    ++v123;
  }
  else
  {
    v53 = v13 + 2;
    v54 = v13[4];
    v107 = v53;
    if ( (_QWORD *)v54 == v53 )
      goto LABEL_90;
    v55 = v54;
    do
    {
      v58 = sub_23FE670(&v118, (__int64)&v119, (unsigned __int64 *)(v55 + 32));
      v60 = v59;
      if ( v59 )
      {
        v56 = v58 || v59 == &v119 || *(_QWORD *)(v55 + 32) < (unsigned __int64)v59[4];
        v57 = sub_22077B0(0x28u);
        *(_QWORD *)(v57 + 32) = *(_QWORD *)(v55 + 32);
        sub_220F040(v56, v57, v60, &v119);
        ++v123;
      }
      v55 = sub_220EF30(v55);
    }
    while ( v107 != (_QWORD *)v55 );
  }
  v42 = *(_QWORD *)(a1 + 456);
  LODWORD(a4) = *(_DWORD *)(a1 + 472);
LABEL_90:
  v74 = a1 + 448;
  if ( !(_DWORD)a4 )
  {
    v117[0] = 0;
    ++*(_QWORD *)(a1 + 448);
    goto LABEL_143;
  }
  v75 = v105[1];
  v76 = 1;
  v77 = ((_DWORD)a4 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
  v78 = (_QWORD *)(v42 + 56 * v77);
  v79 = 0;
  v80 = *v78;
  if ( *v78 != v75 )
  {
    while ( v80 != -4096 )
    {
      if ( v80 == -8192 && !v79 )
        v79 = v78;
      v77 = ((_DWORD)a4 - 1) & (unsigned int)(v76 + v77);
      v78 = (_QWORD *)(v42 + 56LL * (unsigned int)v77);
      v80 = *v78;
      if ( v75 == *v78 )
        goto LABEL_92;
      ++v76;
    }
    if ( !v79 )
      v79 = v78;
    ++*(_QWORD *)(a1 + 448);
    v92 = *(_DWORD *)(a1 + 464);
    v117[0] = v79;
    v93 = v92 + 1;
    if ( 4 * (v92 + 1) < (unsigned int)(3 * a4) )
    {
      v77 = (unsigned int)(a4 - (v93 + *(_DWORD *)(a1 + 468)));
      if ( (unsigned int)v77 > (unsigned int)a4 >> 3 )
      {
LABEL_137:
        *(_DWORD *)(a1 + 464) = v93;
        if ( *v79 != -4096 )
          --*(_DWORD *)(a1 + 468);
        v83 = v79 + 2;
        v81 = 0;
        v94 = v105[1];
        *((_DWORD *)v79 + 4) = 0;
        v82 = v79 + 1;
        v79[3] = 0;
        *v79 = v94;
        v79[4] = v79 + 2;
        v79[5] = v79 + 2;
        v79[6] = 0;
        goto LABEL_93;
      }
      v95 = a4;
LABEL_144:
      sub_23FFA90(v74, v95);
      v77 = (__int64)(v105 + 1);
      sub_23FDEA0(v74, v105 + 1, v117);
      v93 = *(_DWORD *)(a1 + 464) + 1;
      v79 = (_QWORD *)v117[0];
      goto LABEL_137;
    }
LABEL_143:
    v95 = 2 * a4;
    goto LABEL_144;
  }
LABEL_92:
  v81 = v78[3];
  v82 = v78 + 1;
  v83 = v78 + 2;
LABEL_93:
  sub_240E290(v81);
  v82[2] = 0;
  v82[3] = v83;
  v82[4] = v83;
  v82[5] = 0;
  if ( v120 )
  {
    *((_DWORD *)v82 + 2) = v119;
    v84 = v120;
    v82[2] = v120;
    v82[3] = v121;
    v82[4] = v122;
    *(_QWORD *)(v84 + 8) = v83;
    v82[5] = v123;
    v120 = 0;
    v121 = &v119;
    v122 = &v119;
    v123 = 0;
  }
  v112 = v105[1];
  sub_240E290(0);
  sub_F94A20(v124, v77);
  return v112;
}
