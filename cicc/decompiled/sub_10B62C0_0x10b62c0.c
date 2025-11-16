// Function: sub_10B62C0
// Address: 0x10b62c0
//
unsigned __int8 *__fastcall sub_10B62C0(__m128i *a1, unsigned __int8 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  char v5; // al
  __int64 *v6; // rax
  __int64 v7; // r14
  __int64 v9; // rax
  int v10; // eax
  int v11; // eax
  __int64 *v12; // rbx
  __int64 v13; // r14
  __m128i *v14; // rdi
  __m128i *v15; // rsi
  __int64 i; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v27; // rsi
  unsigned __int8 *v28; // rbx
  bool v29; // zf
  __int64 *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  _BYTE *v36; // rax
  unsigned int **v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int8 *v44; // rdx
  unsigned int **v45; // r15
  __int64 v46; // rdi
  __int64 v47; // rdi
  unsigned __int8 *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r9
  unsigned int **v51; // r14
  unsigned __int8 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r9
  _BYTE *v55; // rsi
  __int64 v56; // rax
  char v57; // al
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  unsigned __int64 v61; // rsi
  unsigned int **v62; // r14
  __int64 v63; // rax
  unsigned int **v64; // r10
  __int64 v65; // rax
  unsigned int **v66; // r15
  __int64 v67; // rax
  __int64 v68; // r9
  __int64 v69; // rax
  _BYTE *v70; // rax
  unsigned int **v71; // r14
  __int64 v72; // rax
  __int64 v73; // r14
  _QWORD *v74; // rbx
  __int64 v75; // rax
  __int64 v76; // r9
  __int64 v77; // rax
  _BYTE *v78; // rax
  unsigned int **v79; // r15
  __int64 v80; // rax
  __int64 v81; // r9
  __int64 v82; // [rsp+8h] [rbp-128h]
  __int64 v83; // [rsp+8h] [rbp-128h]
  _BYTE *v84; // [rsp+10h] [rbp-120h]
  __m128i v85; // [rsp+18h] [rbp-118h]
  __int64 v86; // [rsp+28h] [rbp-108h]
  unsigned __int8 *v87; // [rsp+28h] [rbp-108h]
  unsigned int **v88; // [rsp+28h] [rbp-108h]
  __int64 v89; // [rsp+28h] [rbp-108h]
  _BYTE *v90; // [rsp+30h] [rbp-100h] BYREF
  _BYTE *v91; // [rsp+38h] [rbp-F8h] BYREF
  _BYTE *v92; // [rsp+40h] [rbp-F0h] BYREF
  _BYTE *v93; // [rsp+48h] [rbp-E8h] BYREF
  _QWORD *v94; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v95; // [rsp+58h] [rbp-D8h] BYREF
  _BYTE *v96; // [rsp+60h] [rbp-D0h] BYREF
  _BYTE *v97; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v98; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+78h] [rbp-B8h] BYREF
  _QWORD *v100; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v101; // [rsp+88h] [rbp-A8h]
  _QWORD **v102; // [rsp+90h] [rbp-A0h]
  int v103; // [rsp+98h] [rbp-98h]
  _BYTE **v104; // [rsp+A0h] [rbp-90h]
  __m128i v105; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v106; // [rsp+C0h] [rbp-70h]
  _BYTE **v107; // [rsp+D0h] [rbp-60h]
  unsigned __int8 *v108; // [rsp+D8h] [rbp-58h]
  __m128i v109; // [rsp+E0h] [rbp-50h]
  __int64 v110; // [rsp+F0h] [rbp-40h]

  v3 = (__int64)a2;
  v4 = a1[10].m128i_i64[0];
  v105 = _mm_loadu_si128(a1 + 6);
  v107 = (_BYTE **)_mm_loadu_si128(a1 + 8).m128i_u64[0];
  v110 = v4;
  v108 = a2;
  v106 = _mm_loadu_si128(a1 + 7);
  v109 = _mm_loadu_si128(a1 + 9);
  v5 = sub_B45210((__int64)a2);
  v6 = sub_10091C0(*((__int64 **)a2 - 8), *((__int64 **)a2 - 4), v5, &v105, 0, 1);
  if ( !v6 )
  {
    v9 = (__int64)sub_F0F270((__int64)a1, a2);
    if ( v9 )
      return (unsigned __int8 *)v9;
    v9 = sub_F11DB0(a1->m128i_i64, a2);
    if ( v9 )
      return (unsigned __int8 *)v9;
    if ( (unsigned __int8)sub_920620((__int64)a2) )
    {
      v10 = *a2;
      if ( (unsigned __int8)v10 <= 0x1Cu )
        v11 = *((unsigned __int16 *)a2 + 1);
      else
        v11 = v10 - 29;
      if ( v11 == 12 )
      {
        v27 = *(_QWORD *)sub_986520((__int64)a2);
        if ( !v27 )
          goto LABEL_22;
        goto LABEL_41;
      }
      if ( v11 != 16 )
        goto LABEL_22;
      v29 = (a2[1] & 0x10) == 0;
      v105.m128i_i64[0] = 0;
      if ( !v29 )
      {
        v30 = (__int64 *)sub_986520((__int64)a2);
        if ( !(unsigned __int8)sub_10A62F0((__int64 **)&v105, *v30) )
          goto LABEL_22;
LABEL_45:
        v27 = *(_QWORD *)(sub_986520((__int64)a2) + 32);
        if ( !v27 )
          goto LABEL_22;
LABEL_41:
        LOWORD(v107) = 257;
        v28 = (unsigned __int8 *)sub_B50340(12, v27, (__int64)&v105, 0, 0);
        sub_B45260(v28, v3, 1);
        return v28;
      }
      v31 = (__int64 *)sub_986520((__int64)a2);
      if ( (unsigned __int8)sub_1008640((__int64 **)&v105, *v31) )
        goto LABEL_45;
    }
LABEL_22:
    v9 = (__int64)sub_10A6470((unsigned __int8 *)v3);
    if ( !v9 )
    {
      v9 = (__int64)sub_F18290(a1, (unsigned __int8 *)v3);
      if ( !v9 )
      {
        v12 = *(__int64 **)(v3 - 64);
        v13 = *(_QWORD *)(v3 - 32);
        if ( sub_B451E0(v3) )
          goto LABEL_135;
        v14 = &v105;
        v15 = a1 + 6;
        for ( i = 18; i; --i )
        {
          v14->m128i_i32[0] = v15->m128i_i32[0];
          v15 = (__m128i *)((char *)v15 + 4);
          v14 = (__m128i *)((char *)v14 + 4);
        }
        v108 = (unsigned __int8 *)v3;
        if ( (sub_9B4030(v12, 32, 0, &v105) & 0x20) == 0 )
        {
LABEL_135:
          v17 = *(_QWORD *)(v13 + 16);
          if ( v17 )
          {
            if ( !*(_QWORD *)(v17 + 8) && *(_BYTE *)v13 == 45 )
            {
              if ( *(_QWORD *)(v13 - 64) )
              {
                v90 = *(_BYTE **)(v13 - 64);
                v36 = *(_BYTE **)(v13 - 32);
                if ( v36 )
                {
                  v37 = (unsigned int **)a1[2].m128i_i64[0];
                  v91 = v36;
                  LOWORD(v107) = 257;
                  sub_10A0170((__int64)&v100, v3);
                  v38 = sub_94AB40(v37, v91, v90, (__int64)v100, (__int64)&v105, 0);
LABEL_56:
                  v22 = v38;
LABEL_37:
                  LOWORD(v107) = 257;
                  return sub_109FE60(14, (__int64)v12, v22, v3, (__int64)&v105, v25, 0, 0);
                }
              }
            }
          }
        }
        if ( sub_B451E0(v3) && *(_BYTE *)v12 != 5 )
        {
          v105.m128i_i64[0] = (__int64)&v90;
          v21 = v12[2];
          if ( v21 )
          {
            if ( !*(_QWORD *)(v21 + 8) && (unsigned __int8)sub_995E90(&v105, (unsigned __int64)v12, v18, v19, v20) )
            {
              v45 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v107) = 257;
              sub_10A0170((__int64)&v100, v3);
              v27 = sub_92A220(v45, v90, (_BYTE *)v13, (__int64)v100, (__int64)&v105, 0);
              goto LABEL_41;
            }
          }
        }
        if ( *(_BYTE *)v12 > 0x15u
          || *(_BYTE *)v13 != 86
          || (v9 = (__int64)sub_F26350((__int64)a1, (_BYTE *)v3, v13, 0)) == 0 )
        {
          v105.m128i_i64[0] = (__int64)&v92;
          if ( (unsigned __int8)sub_F11D70(&v105, (_BYTE *)v13) )
          {
            v22 = sub_96E680(12, (__int64)v92);
            if ( v22 )
              goto LABEL_37;
          }
          v85.m128i_i64[1] = (__int64)&v91;
          v105.m128i_i64[0] = (__int64)&v91;
          if ( (unsigned __int8)sub_995E90(&v105, v13, v22, v23, v24) )
          {
            LOWORD(v107) = 257;
            return sub_109FE60(14, (__int64)v12, (__int64)v91, v3, (__int64)&v105, v35, 0, 0);
          }
          v86 = *(_QWORD *)(v3 + 8);
          v105.m128i_i64[0] = (__int64)&v91;
          v39 = *(_QWORD *)(v13 + 16);
          if ( v39
            && !*(_QWORD *)(v39 + 8)
            && *(_BYTE *)v13 == 74
            && (unsigned __int8)sub_995E90(&v105, *(_QWORD *)(v13 - 32), v32, v33, v34) )
          {
            HIDWORD(v98) = 0;
            v47 = a1[2].m128i_i64[0];
            LOWORD(v107) = 257;
            LOWORD(v104) = 257;
            v99 = (unsigned int)v98;
            if ( *(_BYTE *)(v47 + 108) )
              v22 = sub_B358C0(v47, 0x71u, (__int64)v91, v86, (unsigned int)v98, (__int64)&v100, 0, 0, 0);
            else
              v22 = sub_10A0030((__int64 *)v47, 45, (__int64)v91, v86, (__int64)&v100, 0, v98, 0);
            return sub_109FE60(14, (__int64)v12, v22, v3, (__int64)&v105, v25, 0, 0);
          }
          v105.m128i_i64[0] = (__int64)&v91;
          v40 = *(_QWORD *)(v13 + 16);
          if ( v40
            && !*(_QWORD *)(v40 + 8)
            && *(_BYTE *)v13 == 75
            && (unsigned __int8)sub_995E90(&v105, *(_QWORD *)(v13 - 32), v32, v33, v34) )
          {
            HIDWORD(v98) = 0;
            v46 = a1[2].m128i_i64[0];
            LOWORD(v107) = 257;
            LOWORD(v104) = 257;
            v99 = (unsigned int)v98;
            if ( *(_BYTE *)(v46 + 108) )
              v22 = sub_B358C0(v46, 0x6Eu, (__int64)v91, v86, (unsigned int)v98, (__int64)&v100, 0, 0, 0);
            else
              v22 = sub_10A0030((__int64 *)v46, 46, (__int64)v91, v86, (__int64)&v100, 0, v98, 0);
            return sub_109FE60(14, (__int64)v12, v22, v3, (__int64)&v105, v25, 0, 0);
          }
          v105.m128i_i64[0] = (__int64)&v90;
          v85.m128i_i64[0] = (__int64)&v90;
          v105.m128i_i64[1] = (__int64)&v91;
          v41 = *(_QWORD *)(v13 + 16);
          if ( v41 && !*(_QWORD *)(v41 + 8) && *(_BYTE *)v13 == 47 )
          {
            v57 = sub_995E90(&v105, *(_QWORD *)(v13 - 64), v32, v33, v34);
            v61 = *(_QWORD *)(v13 - 32);
            if ( v57 && v61 )
            {
              *(_QWORD *)v105.m128i_i64[1] = v61;
LABEL_109:
              v62 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v107) = 257;
              sub_10A0170((__int64)&v100, v3);
              v38 = sub_A826E0(v62, v90, v91, (__int64)v100, (__int64)&v105, 0);
              goto LABEL_56;
            }
            if ( (unsigned __int8)sub_995E90(&v105, v61, v58, v59, v60) )
            {
              v69 = *(_QWORD *)(v13 - 64);
              if ( v69 )
              {
                *(_QWORD *)v105.m128i_i64[1] = v69;
                goto LABEL_109;
              }
            }
          }
          v100 = &v90;
          v101 = &v91;
          v42 = *(_QWORD *)(v13 + 16);
          if ( v42 )
          {
            if ( !*(_QWORD *)(v42 + 8) && *(_BYTE *)v13 == 50 )
            {
              if ( (unsigned __int8)sub_995E90(&v100, *(_QWORD *)(v13 - 64), v32, v33, v34) )
              {
                v56 = *(_QWORD *)(v13 - 32);
                if ( v56 )
                {
                  *v101 = v56;
LABEL_92:
                  v51 = (unsigned int **)a1[2].m128i_i64[0];
                  LOWORD(v107) = 257;
                  sub_10A0170((__int64)&v100, v3);
                  v38 = sub_A82920(v51, v90, v91, (__int64)v100, (__int64)&v105, 0);
                  goto LABEL_56;
                }
              }
            }
          }
          v105 = v85;
          v43 = *(_QWORD *)(v13 + 16);
          if ( v43 )
          {
            if ( !*(_QWORD *)(v43 + 8) && *(_BYTE *)v13 == 50 )
            {
              if ( *(_QWORD *)(v13 - 64) )
              {
                v90 = *(_BYTE **)(v13 - 64);
                if ( (unsigned __int8)sub_995E90((_QWORD **)&v105.m128i_i64[1], *(_QWORD *)(v13 - 32), v32, v33, v34) )
                  goto LABEL_92;
              }
            }
          }
          v44 = sub_F0D870(a1, (unsigned __int8 *)v3, (__int64)v12, v13);
          if ( v44 )
            return sub_F162A0((__int64)a1, v3, (__int64)v44);
          if ( !sub_B451B0(v3) || !sub_B451E0(v3) )
            return 0;
          if ( *(_BYTE *)v12 == 45 )
          {
            v63 = *(v12 - 8);
            if ( v63 )
            {
              if ( v13 == v63 )
              {
                v55 = (_BYTE *)*(v12 - 4);
                if ( v55 )
                {
                  v90 = (_BYTE *)*(v12 - 4);
                  LOWORD(v107) = 257;
                  goto LABEL_98;
                }
              }
            }
          }
          v105.m128i_i64[0] = (__int64)v12;
          v105.m128i_i64[1] = (__int64)&v90;
          if ( *(_BYTE *)v13 == 43 && (unsigned __int8)sub_109CE30((__int64)&v105, v13) )
          {
            LOWORD(v107) = 257;
            v55 = v90;
LABEL_98:
            v87 = (unsigned __int8 *)sub_B50340(12, (__int64)v55, (__int64)&v105, 0, 0);
            sub_B45260(v87, v3, 1);
            return v87;
          }
          v105.m128i_i64[0] = v13;
          v105.m128i_i64[1] = (__int64)&v92;
          if ( (unsigned __int8)sub_10AA380((__int64)&v105, 18, (unsigned __int8 *)v12)
            && (v82 = a1[5].m128i_i64[1],
                v48 = sub_AD8DD0(v86, 1.0),
                (v49 = sub_96E6C0(0x10u, (__int64)v92, v48, v82)) != 0) )
          {
            LOWORD(v107) = 257;
            return sub_109FE60(18, v13, v49, v3, (__int64)&v105, v50, 0, 0);
          }
          else
          {
            v105.m128i_i64[0] = (__int64)v12;
            v105.m128i_i64[1] = (__int64)&v92;
            if ( (unsigned __int8)sub_10AA380((__int64)&v105, 18, (unsigned __int8 *)v13)
              && (v83 = a1[5].m128i_i64[1],
                  v84 = v92,
                  v52 = sub_AD8DD0(v86, 1.0),
                  (v53 = sub_96E6C0(0x10u, (__int64)v52, v84, v83)) != 0) )
            {
              LOWORD(v107) = 257;
              return sub_109FE60(18, (__int64)v12, v53, v3, (__int64)&v105, v54, 0, 0);
            }
            else
            {
              v105 = v85;
              v106.m128i_i64[0] = (__int64)&v93;
              if ( (unsigned __int8)sub_10A5E90(&v105, (__int64)v12) )
              {
                v64 = (unsigned int **)a1[2].m128i_i64[0];
                LOWORD(v107) = 257;
                v88 = v64;
                sub_10A0170((__int64)&v100, v3);
                v65 = sub_92A220(v88, v90, v93, (__int64)v100, (__int64)&v105, 0);
                v66 = (unsigned int **)a1[2].m128i_i64[0];
                v89 = v65;
                LOWORD(v107) = 257;
                sub_10A0170((__int64)&v100, v3);
                v67 = sub_92A220(v66, v91, (_BYTE *)v13, (__int64)v100, (__int64)&v105, 0);
                LOWORD(v107) = 257;
                return sub_109FE60(16, v89, v67, v3, (__int64)&v105, v68, 0, 0);
              }
              else
              {
                LODWORD(v100) = 389;
                v102 = &v94;
                LODWORD(v101) = 0;
                v103 = 1;
                v104 = &v96;
                if ( (unsigned __int8)sub_10A5F50((__int64)&v100, (__int64)v12)
                  && (v106.m128i_i64[0] = (__int64)&v95,
                      v105.m128i_i32[0] = 389,
                      v105.m128i_i32[2] = 0,
                      v106.m128i_i32[2] = 1,
                      v107 = &v97,
                      (unsigned __int8)sub_10A5F50((__int64)&v105, v13))
                  && *((_QWORD *)v97 + 1) == *((_QWORD *)v96 + 1) )
                {
                  v71 = (unsigned int **)a1[2].m128i_i64[0];
                  LOWORD(v107) = 257;
                  sub_10A0170((__int64)&v100, v3);
                  v72 = sub_94AB40(v71, v96, v97, (__int64)v100, (__int64)&v105, 0);
                  v73 = a1[2].m128i_i64[0];
                  v74 = (_QWORD *)v72;
                  LOWORD(v107) = 257;
                  sub_10A0170((__int64)&v99, v3);
                  v101 = v74;
                  v100 = v94;
                  v98 = v74[1];
                  v75 = sub_B33D10(v73, 0x185u, (__int64)&v98, 1, (int)&v100, 2, v99, (__int64)&v105);
                  LOWORD(v107) = 257;
                  return sub_109FE60(16, v75, v95, v3, (__int64)&v105, v76, 0, 0);
                }
                else
                {
                  v9 = (__int64)sub_10B5C50((_BYTE *)v3, a1[2].m128i_i64[0]);
                  if ( !v9 )
                  {
                    v105 = (__m128i)a1[2].m128i_u64[0];
                    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL) - 17) > 1u
                      && (v70 = sub_109F890(v105.m128i_i64, v3)) != 0 )
                    {
                      return sub_F162A0((__int64)a1, v3, (__int64)v70);
                    }
                    else
                    {
                      v77 = v12[2];
                      if ( !v77 )
                        return 0;
                      if ( *(_QWORD *)(v77 + 8) )
                        return 0;
                      if ( *(_BYTE *)v12 != 45 )
                        return 0;
                      if ( !*(v12 - 8) )
                        return 0;
                      v90 = (_BYTE *)*(v12 - 8);
                      v78 = (_BYTE *)*(v12 - 4);
                      if ( !v78 )
                        return 0;
                      v79 = (unsigned int **)a1[2].m128i_i64[0];
                      LOWORD(v107) = 257;
                      v91 = v78;
                      sub_10A0170((__int64)&v100, v3);
                      v80 = sub_92A220(v79, v91, (_BYTE *)v13, (__int64)v100, (__int64)&v105, 0);
                      LOWORD(v107) = 257;
                      return sub_109FE60(16, (__int64)v90, v80, v3, (__int64)&v105, v81, 0, 0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return (unsigned __int8 *)v9;
  }
  if ( !*((_QWORD *)a2 + 2) )
    return 0;
  v7 = (__int64)v6;
  sub_10A5FE0(a1[2].m128i_i64[1], (__int64)a2);
  if ( a2 == (unsigned __int8 *)v7 )
  {
    v7 = sub_ACADE0(*((__int64 ***)a2 + 1));
    if ( *(_QWORD *)(v7 + 16) )
      goto LABEL_5;
LABEL_9:
    if ( *(_BYTE *)v7 > 0x1Cu && (*(_BYTE *)(v7 + 7) & 0x10) == 0 && (a2[7] & 0x10) != 0 )
      sub_BD6B90((unsigned __int8 *)v7, a2);
    goto LABEL_5;
  }
  if ( !*(_QWORD *)(v7 + 16) )
    goto LABEL_9;
LABEL_5:
  sub_BD84D0((__int64)a2, v7);
  return (unsigned __int8 *)v3;
}
