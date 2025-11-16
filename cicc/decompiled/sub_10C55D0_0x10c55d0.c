// Function: sub_10C55D0
// Address: 0x10c55d0
//
_QWORD *__fastcall sub_10C55D0(const __m128i *a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4)
{
  unsigned __int8 *v5; // r12
  unsigned int v6; // r13d
  __int64 v7; // rbx
  __int64 v8; // r14
  char v9; // al
  unsigned int v10; // r11d
  __int64 v11; // r9
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int8 *v14; // r13
  unsigned __int8 *v15; // rax
  bool v16; // dl
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  const char *v24; // rax
  __int64 v25; // rdi
  _BYTE *v26; // rdx
  __int64 v27; // r14
  __int64 v28; // r14
  __int64 i; // rbx
  _QWORD *v30; // r13
  __int64 v31; // rsi
  __int64 v32; // r9
  _BYTE *v33; // r14
  unsigned __int64 v34; // rsi
  _QWORD *v35; // rax
  int v36; // ecx
  _QWORD *v37; // rdx
  unsigned int **v38; // rdi
  _QWORD *result; // rax
  __int64 v40; // rdx
  _BYTE *v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdx
  _BYTE *v44; // rax
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // r12
  __int64 v49; // rdx
  unsigned int v50; // esi
  int v51; // ecx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // r11d
  _BYTE *v57; // r14
  _BYTE *v58; // rax
  unsigned int **v59; // rdi
  _BYTE *v60; // rbx
  __int64 v61; // rsi
  __int64 v62; // rax
  unsigned int **v63; // r15
  __int64 v64; // r12
  _BYTE *v65; // rax
  unsigned int v66; // eax
  int v67; // ebx
  unsigned int v68; // ebx
  bool v69; // al
  __int64 v70; // r9
  __int64 v71; // rsi
  unsigned int **v72; // r12
  unsigned __int64 v73; // rsi
  bool v74; // al
  bool v75; // al
  bool v76; // al
  __int64 v77; // rax
  __int64 *v78; // rdi
  unsigned int v79; // r11d
  __int64 v80; // r9
  bool v81; // dl
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // r13
  unsigned int **v87; // r12
  __int64 v88; // rdi
  _BYTE *v89; // rax
  _BYTE *v90; // rax
  __int64 v91; // rcx
  __int64 v94; // [rsp+50h] [rbp-190h]
  __int64 v95; // [rsp+50h] [rbp-190h]
  __int64 v96; // [rsp+50h] [rbp-190h]
  __int64 v97; // [rsp+50h] [rbp-190h]
  __int64 v98; // [rsp+50h] [rbp-190h]
  __int64 v99; // [rsp+50h] [rbp-190h]
  __int64 v100; // [rsp+50h] [rbp-190h]
  unsigned int v101; // [rsp+58h] [rbp-188h]
  unsigned int v102; // [rsp+58h] [rbp-188h]
  __int64 v103; // [rsp+58h] [rbp-188h]
  unsigned int v104; // [rsp+58h] [rbp-188h]
  unsigned int v105; // [rsp+58h] [rbp-188h]
  unsigned int v106; // [rsp+58h] [rbp-188h]
  unsigned int v107; // [rsp+58h] [rbp-188h]
  unsigned int v108; // [rsp+58h] [rbp-188h]
  __int64 v109; // [rsp+60h] [rbp-180h]
  __int64 *v110; // [rsp+60h] [rbp-180h]
  int v111; // [rsp+60h] [rbp-180h]
  __int64 v113; // [rsp+68h] [rbp-178h]
  __int64 v114; // [rsp+68h] [rbp-178h]
  __int64 v115; // [rsp+68h] [rbp-178h]
  __int64 v116; // [rsp+68h] [rbp-178h]
  __int64 v117; // [rsp+68h] [rbp-178h]
  __int64 v118; // [rsp+68h] [rbp-178h]
  char v119; // [rsp+7Ah] [rbp-166h] BYREF
  char v120; // [rsp+7Bh] [rbp-165h] BYREF
  unsigned int v121; // [rsp+7Ch] [rbp-164h] BYREF
  __int64 v122; // [rsp+80h] [rbp-160h] BYREF
  int v123; // [rsp+88h] [rbp-158h]
  __int64 v124; // [rsp+90h] [rbp-150h] BYREF
  int v125; // [rsp+98h] [rbp-148h]
  __int64 v126[2]; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v127[2]; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v128[2]; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v129[2]; // [rsp+D0h] [rbp-110h] BYREF
  _BYTE v130[32]; // [rsp+E0h] [rbp-100h] BYREF
  char v131; // [rsp+100h] [rbp-E0h]
  _BYTE *v132[4]; // [rsp+110h] [rbp-D0h] BYREF
  __int16 v133; // [rsp+130h] [rbp-B0h]
  unsigned int v134[8]; // [rsp+140h] [rbp-A0h] BYREF
  __int16 v135; // [rsp+160h] [rbp-80h]
  _QWORD *v136; // [rsp+170h] [rbp-70h] BYREF
  unsigned int *v137; // [rsp+178h] [rbp-68h] BYREF
  __int64 v138; // [rsp+180h] [rbp-60h] BYREF
  __int64 v139; // [rsp+188h] [rbp-58h]
  __int64 v140; // [rsp+190h] [rbp-50h]
  __int16 v141; // [rsp+198h] [rbp-48h]
  __int64 v142[8]; // [rsp+1A0h] [rbp-40h] BYREF

  v5 = a2;
  v6 = *((_WORD *)a3 + 1) & 0x3F;
  v7 = *((_QWORD *)a3 - 8);
  v8 = *((_QWORD *)a3 - 4);
  v94 = *((_QWORD *)a2 - 4);
  v101 = *((_WORD *)a2 + 1) & 0x3F;
  v109 = *((_QWORD *)a2 - 8);
  v9 = sub_11FAEC0(v101, v6);
  v10 = v101;
  if ( v9 )
  {
    if ( v94 == v7 && v109 == v8 )
    {
      v66 = sub_B52F50(v101);
      v42 = v94;
      v10 = v66;
      goto LABEL_94;
    }
    if ( v109 == v7 && v94 == v8 )
    {
      v109 = v94;
      v42 = v7;
LABEL_94:
      v103 = v42;
      v67 = sub_11FAD40(v10);
      v68 = sub_11FAD40(v6) ^ v67;
      v69 = sub_B532B0(*((_WORD *)a2 + 1) & 0x3F);
      v70 = v103;
      v71 = 1;
      if ( !v69 )
      {
        v74 = sub_B532B0(*((_WORD *)a3 + 1) & 0x3F);
        v70 = v103;
        v71 = v74;
      }
      v113 = v70;
      v72 = (unsigned int **)a1[2].m128i_i64[0];
      result = (_QWORD *)sub_11FAD60(v68, v71, *(_QWORD *)(v70 + 8), v134);
      if ( !result )
      {
        LOWORD(v140) = 257;
        return (_QWORD *)sub_92B530(v72, v134[0], v113, (_BYTE *)v109, (__int64)&v136);
      }
      return result;
    }
  }
  if ( *(_BYTE *)v94 == 17 )
  {
    v11 = v94 + 24;
  }
  else
  {
    v40 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v94 + 8) + 8LL) - 17;
    if ( (unsigned int)v40 > 1 )
      goto LABEL_10;
    if ( *(_BYTE *)v94 > 0x15u )
      goto LABEL_10;
    v41 = sub_AD7630(v94, 0, v40);
    if ( !v41 || *v41 != 17 )
      goto LABEL_10;
    v10 = v101;
    v11 = (__int64)(v41 + 24);
  }
  if ( *(_BYTE *)v8 == 17 )
  {
    v12 = v8 + 24;
  }
  else
  {
    v95 = v11;
    v102 = v10;
    v43 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v43 > 1 )
      goto LABEL_10;
    if ( *(_BYTE *)v8 > 0x15u )
      goto LABEL_10;
    v44 = sub_AD7630(v8, 0, v43);
    if ( !v44 || *v44 != 17 )
      goto LABEL_10;
    v11 = v95;
    v10 = v102;
    v12 = (__int64)(v44 + 24);
  }
  v13 = *(_QWORD *)(v109 + 8);
  if ( *(_QWORD *)(v7 + 8) != v13 )
    goto LABEL_10;
  v51 = *(unsigned __int8 *)(v13 + 8);
  if ( (unsigned int)(v51 - 17) <= 1 )
    LOBYTE(v51) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
  if ( (_BYTE)v51 != 12 )
  {
LABEL_10:
    v14 = sub_101E7C0(29, (__int64 *)a2, (__int64 *)a3, (__m128i *)&a1[6]);
    if ( !v14 )
      return 0;
    v15 = sub_101E7C0(28, (__int64 *)a2, (__int64 *)a3, (__m128i *)&a1[6]);
    if ( !v15 )
      return 0;
    v16 = a2 == v15 && a3 == v14;
    if ( a2 == v14 && a3 == v15 )
    {
      v17 = (__int64)a3;
      if ( v16 )
        v17 = (__int64)a2;
      v18 = *(_QWORD *)(v17 + 16);
      if ( !v18 )
        goto LABEL_17;
    }
    else
    {
      if ( !v16 )
        return 0;
      v17 = (__int64)a2;
      v18 = *((_QWORD *)a2 + 2);
      if ( !v18 )
      {
LABEL_17:
        if ( (unsigned __int8)sub_10C2350(v17, (unsigned __int8 *)a4) )
        {
LABEL_18:
          *(_WORD *)(v17 + 2) = sub_B52870(*(_WORD *)(v17 + 2) & 0x3F) | *(_WORD *)(v17 + 2) & 0xFFC0;
          v19 = *(_QWORD *)(v17 + 16);
          if ( v19 && !*(_QWORD *)(v19 + 8) )
          {
LABEL_39:
            v38 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v140) = 257;
            return (_QWORD *)sub_A82350(v38, v5, a3, (__int64)&v136);
          }
          v20 = a1[2].m128i_i64[0];
          v136 = (_QWORD *)v20;
          v21 = *(_QWORD *)(v20 + 48);
          v137 = 0;
          v138 = 0;
          v139 = v21;
          if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
            sub_BD73F0((__int64)&v137);
          v22 = *(_QWORD *)(v20 + 56);
          v141 = *(_WORD *)(v20 + 64);
          v140 = v22;
          sub_B33910(v142, (__int64 *)v20);
          sub_A88F30(a1[2].m128i_i64[0], *(_QWORD *)(v17 + 40), *(_QWORD *)(v17 + 32), 0);
          v23 = a1[2].m128i_i64[0];
          v24 = sub_BD5D20(v17);
          v25 = *(_QWORD *)(v17 + 8);
          v132[0] = v24;
          v133 = 773;
          v132[1] = v26;
          v132[2] = ".not";
          v27 = sub_AD62B0(v25);
          v110 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v23 + 80) + 16LL))(
                              *(_QWORD *)(v23 + 80),
                              30,
                              v17,
                              v27);
          if ( !v110 )
          {
            v135 = 257;
            v110 = (__int64 *)sub_B504D0(30, v17, v27, (__int64)v134, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64 *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v23 + 88) + 16LL))(
              *(_QWORD *)(v23 + 88),
              v110,
              v132,
              *(_QWORD *)(v23 + 56),
              *(_QWORD *)(v23 + 64));
            v46 = *(_QWORD *)v23;
            v47 = *(_QWORD *)v23 + 16LL * *(unsigned int *)(v23 + 8);
            if ( v46 != v47 )
            {
              v48 = v46;
              do
              {
                v49 = *(_QWORD *)(v48 + 8);
                v50 = *(_DWORD *)v48;
                v48 += 16;
                sub_B99FD0((__int64)v110, v50, v49);
              }
              while ( v47 != v48 );
              v5 = a2;
            }
          }
          v28 = *(_QWORD *)(v17 + 16);
          for ( i = a1[2].m128i_i64[1]; v28; v28 = *(_QWORD *)(v28 + 8) )
            sub_F15FC0(i, *(_QWORD *)(v28 + 24));
          *(_QWORD *)v134 = v110;
          sub_BD79D0(
            (__int64 *)v17,
            v110,
            (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_10B8240,
            (__int64)v134);
          v30 = v136;
          if ( v139 )
          {
            sub_A88F30((__int64)v136, v139, v140, v141);
            v31 = v142[0];
            v30 = v136;
            *(_QWORD *)v134 = v142[0];
            if ( !v142[0] )
              goto LABEL_56;
          }
          else
          {
            v136[6] = 0;
            v30[7] = 0;
            *((_WORD *)v30 + 32) = 0;
            v31 = v142[0];
            *(_QWORD *)v134 = v142[0];
            if ( !v142[0] )
              goto LABEL_56;
          }
          sub_B96E90((__int64)v134, v31, 1);
          v33 = *(_BYTE **)v134;
          if ( *(_QWORD *)v134 )
          {
            v34 = *((unsigned int *)v30 + 2);
            v35 = (_QWORD *)*v30;
            v36 = *((_DWORD *)v30 + 2);
            v37 = (_QWORD *)(*v30 + 16 * v34);
            if ( (_QWORD *)*v30 != v37 )
            {
              while ( *(_DWORD *)v35 )
              {
                v35 += 2;
                if ( v37 == v35 )
                  goto LABEL_62;
              }
              v35[1] = *(_QWORD *)v134;
LABEL_33:
              sub_B91220((__int64)v134, (__int64)v33);
LABEL_34:
              if ( v142[0] )
                sub_B91220((__int64)v142, v142[0]);
              if ( v139 != 0 && v139 != -4096 && v139 != -8192 )
                sub_BD60C0(&v137);
              goto LABEL_39;
            }
LABEL_62:
            v45 = *((unsigned int *)v30 + 3);
            if ( v34 >= v45 )
            {
              v73 = v34 + 1;
              if ( v45 < v73 )
              {
                sub_C8D5F0((__int64)v30, v30 + 2, v73, 0x10u, (__int64)(v30 + 2), v32);
                v37 = (_QWORD *)(*v30 + 16LL * *((unsigned int *)v30 + 2));
              }
              *v37 = 0;
              v37[1] = v33;
              ++*((_DWORD *)v30 + 2);
              v33 = *(_BYTE **)v134;
            }
            else
            {
              if ( v37 )
              {
                *(_DWORD *)v37 = 0;
                v37[1] = v33;
                v36 = *((_DWORD *)v30 + 2);
                v33 = *(_BYTE **)v134;
              }
              *((_DWORD *)v30 + 2) = v36 + 1;
            }
LABEL_57:
            if ( !v33 )
              goto LABEL_34;
            goto LABEL_33;
          }
LABEL_56:
          sub_93FB40((__int64)v30, 0);
          v33 = *(_BYTE **)v134;
          goto LABEL_57;
        }
        return 0;
      }
    }
    if ( !*(_QWORD *)(v18 + 8) )
      goto LABEL_18;
    goto LABEL_17;
  }
  v52 = *((_QWORD *)a2 + 2);
  if ( v52 && !*(_QWORD *)(v52 + 8) || (v53 = *((_QWORD *)a3 + 2)) != 0 && !*(_QWORD *)(v53 + 8) )
  {
    v96 = v11;
    v104 = v10;
    v75 = sub_9893F0(v10, v11, &v119);
    v10 = v104;
    v11 = v96;
    if ( v75 )
    {
      v76 = sub_9893F0(v6, v12, &v120);
      v10 = v104;
      v11 = v96;
      if ( v76 )
      {
        LOWORD(v140) = 257;
        v77 = sub_A825B0((unsigned int **)a1[2].m128i_i64[0], (_BYTE *)v109, (_BYTE *)v7, (__int64)&v136);
        LOWORD(v140) = 257;
        v78 = (__int64 *)a1[2].m128i_i64[0];
        if ( v119 == v120 )
          return sub_10A0880(v78, v77, (__int64)&v136);
        else
          return sub_10BE620(v78, v77, (__int64)&v136);
      }
    }
  }
  if ( v7 != v109 )
  {
LABEL_78:
    if ( v10 - 32 <= 1 && v6 - 32 <= 1 && sub_9867B0(v11) && sub_9867B0(v12) )
    {
      v54 = *((_QWORD *)a2 + 2);
      if ( v54 )
      {
        if ( !*(_QWORD *)(v54 + 8) )
        {
          v55 = *((_QWORD *)a3 + 2);
          if ( v55 )
          {
            if ( !*(_QWORD *)(v55 + 8) )
            {
              v136 = v132;
              v137 = v134;
              if ( *(_BYTE *)v109 == 57 )
              {
                if ( (unsigned __int8)sub_10B8310(&v136, v109) )
                {
                  if ( *(_BYTE *)v7 == 57 )
                  {
                    v57 = *(_BYTE **)(v7 - 64);
                    if ( v57 )
                    {
                      if ( *(_QWORD *)v134 == *(_QWORD *)(v7 - 32) )
                      {
                        v111 = v56;
                        if ( (unsigned __int8)sub_10BE7C0(a1, *(unsigned __int8 **)v134, 1u, 0, a4) )
                        {
                          LOWORD(v140) = 257;
                          v58 = (_BYTE *)sub_A825B0((unsigned int **)a1[2].m128i_i64[0], v132[0], v57, (__int64)&v136);
                          v59 = (unsigned int **)a1[2].m128i_i64[0];
                          v60 = v58;
                          v61 = (__int64)v58;
                          LOWORD(v140) = 257;
                          v62 = sub_A82350(v59, v58, *(_BYTE **)v134, (__int64)&v136);
                          LOWORD(v140) = 257;
                          v63 = (unsigned int **)a1[2].m128i_i64[0];
                          v64 = v62;
                          v65 = (_BYTE *)sub_AD6530(*((_QWORD *)v60 + 1), v61);
                          return (_QWORD *)sub_92B530(v63, (unsigned int)(v111 == v6) + 32, v64, v65, (__int64)&v136);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    goto LABEL_10;
  }
  v97 = v11;
  v105 = v10;
  sub_AB1A50((__int64)v126, v10, v11);
  sub_AB1A50((__int64)v128, v6, v12);
  sub_ABB970((__int64)v130, (__int64)v126, (__int64)v128);
  sub_ABB730((__int64)v132, (__int64)v126, (__int64)v128);
  v79 = v105;
  v80 = v97;
  if ( !v131 )
    goto LABEL_118;
  if ( !(_BYTE)v133 )
  {
LABEL_110:
    if ( v131 )
    {
      v100 = v80;
      v108 = v79;
      sub_9963D0((__int64)v130);
      v80 = v100;
      v79 = v108;
    }
    v98 = v80;
    v106 = v79;
    sub_969240(v129);
    sub_969240(v128);
    sub_969240(v127);
    sub_969240(v126);
    v11 = v98;
    v10 = v106;
    goto LABEL_78;
  }
  sub_ABB300((__int64)&v136, (__int64)v132);
  sub_ABB730((__int64)v134, (__int64)v130, (__int64)&v136);
  sub_969240(&v138);
  sub_969240((__int64 *)&v136);
  v79 = v105;
  v80 = v97;
  if ( !(_BYTE)v135 )
  {
LABEL_118:
    if ( (_BYTE)v133 )
    {
      v99 = v80;
      v107 = v79;
      sub_9963D0((__int64)v132);
      v80 = v99;
      v79 = v107;
    }
    goto LABEL_110;
  }
  if ( !sub_AAF760((__int64)v134) )
  {
    if ( sub_AAF7D0((__int64)v134) )
    {
      v84 = sub_AD6450(*(_QWORD *)(a4 + 8));
      goto LABEL_123;
    }
    v123 = 1;
    v122 = 0;
    v125 = 1;
    v124 = 0;
    sub_AAF830((__int64)v134, (int *)&v121, (__int64)&v122, &v124);
    v81 = sub_9867B0((__int64)&v124);
    v82 = *((_QWORD *)a2 + 2);
    if ( v81 )
    {
      if ( !v82 )
      {
        v83 = *((_QWORD *)a3 + 2);
        if ( v83 && !*(_QWORD *)(v83 + 8) )
          goto LABEL_141;
        goto LABEL_116;
      }
      if ( !*(_QWORD *)(v82 + 8) )
      {
LABEL_141:
        v86 = *(_QWORD *)(v109 + 8);
        v87 = (unsigned int **)a1[2].m128i_i64[0];
LABEL_139:
        LOWORD(v140) = 257;
        v90 = (_BYTE *)sub_AD8D80(v86, (__int64)&v122);
        v118 = sub_92B530(v87, v121, v109, v90, (__int64)&v136);
        sub_969240(&v124);
        sub_969240(&v122);
        v84 = v118;
        goto LABEL_123;
      }
      v91 = *((_QWORD *)a3 + 2);
      if ( v91 )
      {
        if ( !*(_QWORD *)(v91 + 8) )
          goto LABEL_141;
        goto LABEL_134;
      }
    }
    else if ( v82 )
    {
LABEL_134:
      if ( !*(_QWORD *)(v82 + 8) )
      {
        v85 = *((_QWORD *)a3 + 2);
        if ( v85 )
        {
          if ( !*(_QWORD *)(v85 + 8) )
          {
            v86 = *(_QWORD *)(v109 + 8);
            v87 = (unsigned int **)a1[2].m128i_i64[0];
            if ( !v81 )
            {
              v88 = *(_QWORD *)(v109 + 8);
              LOWORD(v140) = 257;
              v89 = (_BYTE *)sub_AD8D80(v88, (__int64)&v124);
              v109 = sub_929C50(v87, (_BYTE *)v109, v89, (__int64)&v136, 0, 0);
              v87 = (unsigned int **)a1[2].m128i_i64[0];
            }
            goto LABEL_139;
          }
        }
      }
    }
LABEL_116:
    sub_969240(&v124);
    sub_969240(&v122);
    v79 = v105;
    v80 = v97;
    if ( (_BYTE)v135 )
    {
      sub_9963D0((__int64)v134);
      v80 = v97;
      v79 = v105;
    }
    goto LABEL_118;
  }
  v84 = sub_AD6400(*(_QWORD *)(a4 + 8));
LABEL_123:
  if ( (_BYTE)v135 )
  {
    v117 = v84;
    sub_9963D0((__int64)v134);
    v84 = v117;
  }
  if ( (_BYTE)v133 )
  {
    v116 = v84;
    sub_9963D0((__int64)v132);
    v84 = v116;
  }
  if ( v131 )
  {
    v115 = v84;
    sub_9963D0((__int64)v130);
    v84 = v115;
  }
  v114 = v84;
  sub_969240(v129);
  sub_969240(v128);
  sub_969240(v127);
  sub_969240(v126);
  return (_QWORD *)v114;
}
