// Function: sub_1913500
// Address: 0x1913500
//
__int64 __fastcall sub_1913500(
        __int64 *a1,
        __int64 a2,
        __m128 si128,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 i; // r12
  _BYTE **v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // r14d
  int v18; // ebx
  __int64 v19; // rax
  __int64 v20; // rsi
  int v21; // eax
  int v22; // esi
  _BYTE *v23; // rsi
  __int64 v24; // rax
  __m128 *v25; // rdx
  __int64 v26; // rbx
  _BYTE *v27; // rax
  _BYTE *v28; // rdi
  unsigned int v29; // r13d
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r12
  unsigned __int64 *v33; // rax
  unsigned __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // r14
  __int64 v38; // rbx
  char v39; // al
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  int *v45; // r13
  __int64 *v46; // rdi
  __int64 v47; // rdx
  double v48; // xmm4_8
  double v49; // xmm5_8
  _BYTE *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r14
  _QWORD *v53; // rax
  unsigned __int64 v54; // rsi
  __int64 v55; // rcx
  __int64 v56; // rdx
  int *v57; // rax
  __int64 v58; // r13
  __int64 v59; // rbx
  int *v60; // r12
  __int64 v61; // rax
  int *v62; // rax
  int *v63; // r9
  __int64 v64; // rdi
  __int64 v65; // rcx
  int *v66; // r9
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // rax
  int *v70; // rax
  int *v71; // r8
  __int64 v72; // rcx
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r12
  _BYTE *v77; // rax
  _QWORD *j; // r13
  _QWORD *v79; // r14
  unsigned __int64 v80; // r12
  _QWORD *v81; // rdx
  _QWORD *v82; // rbx
  __int64 v83; // rax
  __int64 v84; // rax
  _BYTE *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // r12
  void *v88; // rax
  _BYTE *v89; // rax
  __int64 v90; // rsi
  int *v92; // [rsp+20h] [rbp-1A0h]
  _QWORD *v93; // [rsp+28h] [rbp-198h]
  int v94; // [rsp+34h] [rbp-18Ch]
  __int64 v95; // [rsp+38h] [rbp-188h]
  __int64 v97; // [rsp+60h] [rbp-160h]
  __int64 v98; // [rsp+60h] [rbp-160h]
  __int64 v99; // [rsp+68h] [rbp-158h]
  unsigned __int64 v100; // [rsp+70h] [rbp-150h] BYREF
  unsigned __int64 v101; // [rsp+78h] [rbp-148h] BYREF
  unsigned __int64 v102; // [rsp+80h] [rbp-140h] BYREF
  unsigned __int64 *v103; // [rsp+88h] [rbp-138h] BYREF
  _BYTE *v104; // [rsp+90h] [rbp-130h] BYREF
  _BYTE *v105; // [rsp+98h] [rbp-128h]
  _BYTE *v106; // [rsp+A0h] [rbp-120h]
  _BYTE *v107; // [rsp+B0h] [rbp-110h] BYREF
  _BYTE *v108; // [rsp+B8h] [rbp-108h]
  _BYTE *v109; // [rsp+C0h] [rbp-100h]
  __int64 v110; // [rsp+D0h] [rbp-F0h] BYREF
  int v111; // [rsp+D8h] [rbp-E8h] BYREF
  __int64 v112; // [rsp+E0h] [rbp-E0h]
  int *v113; // [rsp+E8h] [rbp-D8h]
  int *v114; // [rsp+F0h] [rbp-D0h]
  __int64 v115; // [rsp+F8h] [rbp-C8h]
  __int64 v116; // [rsp+100h] [rbp-C0h] BYREF
  int v117; // [rsp+108h] [rbp-B8h] BYREF
  __int64 v118; // [rsp+110h] [rbp-B0h]
  int *v119; // [rsp+118h] [rbp-A8h]
  int *v120; // [rsp+120h] [rbp-A0h]
  __int64 v121; // [rsp+128h] [rbp-98h]
  __int64 v122; // [rsp+130h] [rbp-90h] BYREF
  int v123; // [rsp+138h] [rbp-88h] BYREF
  __int64 v124; // [rsp+140h] [rbp-80h]
  int *v125; // [rsp+148h] [rbp-78h]
  int *v126; // [rsp+150h] [rbp-70h]
  __int64 v127; // [rsp+158h] [rbp-68h]
  __int64 v128; // [rsp+160h] [rbp-60h] BYREF
  int v129; // [rsp+168h] [rbp-58h] BYREF
  _QWORD *v130; // [rsp+170h] [rbp-50h]
  int *v131; // [rsp+178h] [rbp-48h]
  int *v132; // [rsp+180h] [rbp-40h]
  __int64 v133; // [rsp+188h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 80);
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v99 = v10;
  v97 = a2 + 72;
  if ( a2 + 72 == v10 )
    return 0;
  v11 = (__int64)(a1 + 19);
  do
  {
    if ( !v99 )
      BUG();
    v12 = *(_QWORD *)(v99 + 24);
    for ( i = v99 + 16; v12 != i; v12 = *(_QWORD *)(v12 + 8) )
    {
      while ( 1 )
      {
        if ( !v12 )
          BUG();
        v15 = v12 - 24;
        if ( *(_BYTE *)(v12 - 8) != 77 )
          goto LABEL_32;
        v128 = v12 - 24;
        v14 = *(_BYTE ***)(v12 - 24);
        if ( *((_BYTE *)v14 + 8) != 15 )
        {
          if ( dword_4FAECE0 == 1 )
            goto LABEL_7;
          if ( !sub_1642F90((__int64)v14, 32) )
          {
            v14 = *(_BYTE ***)v128;
            if ( !sub_1642F90(*(_QWORD *)v128, 64) )
              goto LABEL_7;
          }
          v15 = v128;
        }
        v16 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
        if ( (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) != 0 )
        {
          v17 = 0;
          v18 = 0;
          if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
          {
LABEL_16:
            v19 = *(_QWORD *)(v15 - 8);
            goto LABEL_17;
          }
          while ( 1 )
          {
            v19 = v15 - 24LL * (unsigned int)v16;
LABEL_17:
            v20 = *(_QWORD *)(v19 + 24LL * v17);
            if ( *(_BYTE *)(v20 + 16) <= 0x17u )
              goto LABEL_7;
            v14 = (_BYTE **)v11;
            v21 = sub_1911FD0(v11, v20);
            v22 = v21;
            if ( v17 )
            {
              if ( v21 != v18 )
                goto LABEL_7;
            }
            v15 = v128;
            ++v17;
            v16 = *(_DWORD *)(v128 + 20) & 0xFFFFFFF;
            if ( (_DWORD)v16 == v17 )
              break;
            v18 = v22;
            if ( (*(_BYTE *)(v128 + 23) & 0x40) != 0 )
              goto LABEL_16;
          }
        }
        v23 = v105;
        if ( v105 == v106 )
        {
          v14 = &v104;
          sub_1769D70((__int64)&v104, v105, &v128);
        }
        else
        {
          if ( v105 )
          {
            *(_QWORD *)v105 = v128;
            v23 = v105;
          }
          v23 += 8;
          v105 = v23;
        }
        if ( dword_4FAEC00 )
          break;
LABEL_7:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v12 == i )
          goto LABEL_32;
      }
      v24 = sub_16BA580((__int64)v14, (__int64)v23, v16);
      v25 = *(__m128 **)(v24 + 24);
      v26 = v24;
      if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 0x12u )
      {
        v26 = sub_16E7EE0(v24, "PHI-removing cand: ", 0x13u);
      }
      else
      {
        si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42BE110);
        v25[1].m128_i8[2] = 32;
        v25[1].m128_i16[0] = 14948;
        *v25 = si128;
        *(_QWORD *)(v24 + 24) += 19LL;
      }
      sub_155C2B0(v128, v26, 0);
      v27 = *(_BYTE **)(v26 + 24);
      if ( *(_BYTE **)(v26 + 16) == v27 )
      {
        sub_16E7EE0(v26, "\n", 1u);
        goto LABEL_7;
      }
      *v27 = 10;
      ++*(_QWORD *)(v26 + 24);
    }
LABEL_32:
    v99 = *(_QWORD *)(v99 + 8);
  }
  while ( v97 != v99 );
  v28 = v104;
  if ( v104 != v105 )
  {
    v29 = 0;
    v30 = 0;
    v94 = 0;
    while ( 1 )
    {
      v31 = *(_QWORD *)&v28[8 * v30];
      v93 = (_QWORD *)v31;
      v32 = *(_QWORD *)(v31 + 40);
      if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
        v33 = *(unsigned __int64 **)(v31 - 8);
      else
        v33 = (unsigned __int64 *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
      v34 = *v33;
      v35 = v32;
      v107 = 0;
      v113 = &v111;
      v114 = &v111;
      v125 = &v123;
      v119 = &v117;
      v120 = &v117;
      v100 = v34;
      v36 = a1[3];
      v108 = 0;
      v109 = 0;
      v111 = 0;
      v112 = 0;
      v115 = 0;
      v117 = 0;
      v118 = 0;
      v121 = 0;
      v123 = 0;
      v124 = 0;
      v126 = &v123;
      v127 = 0;
      sub_190D160(v34, v32, v36, &v110, &v116, (__int64)&v107, &v122);
      v37 = (__int64)v125;
      if ( v125 != &v123 )
      {
        while ( 1 )
        {
          v38 = *(_QWORD *)(v37 + 32);
          v39 = *(_BYTE *)(v38 + 16);
          if ( v39 == 77 )
            break;
          if ( v39 == 78 )
          {
            v35 = 0xFFFFFFFFLL;
            if ( !(unsigned __int8)sub_1560260((_QWORD *)(v38 + 56), -1, 36) )
            {
              if ( *(char *)(v38 + 23) < 0 )
              {
                v40 = sub_1648A40(v38);
                v42 = v41 + v40;
                v43 = 0;
                v98 = v42;
                if ( *(char *)(v38 + 23) < 0 )
                  v43 = sub_1648A40(v38);
                if ( (unsigned int)((v98 - v43) >> 4) )
                  break;
              }
              v44 = *(_QWORD *)(v38 - 24);
              if ( *(_BYTE *)(v44 + 16) )
                break;
              v35 = 0xFFFFFFFFLL;
              v128 = *(_QWORD *)(v44 + 112);
              if ( !(unsigned __int8)sub_1560260(&v128, -1, 36) )
                break;
            }
          }
          else if ( v39 == 54 )
          {
            if ( (*(_BYTE *)(v38 + 18) & 1) != 0 || sub_15F32D0(*(_QWORD *)(v37 + 32)) )
              break;
            v86 = **(_QWORD **)(v38 - 24);
            if ( *(_BYTE *)(v86 + 8) == 16 )
              v86 = **(_QWORD **)(v86 + 16);
            if ( *(_DWORD *)(v86 + 8) >> 8 != 4 )
              break;
          }
          else if ( (unsigned __int8)(v39 - 58) <= 1u )
          {
            break;
          }
          v37 = sub_220EF30(v37);
          if ( (int *)v37 == &v123 )
            goto LABEL_51;
        }
        sub_1909DD0(v124);
        sub_1909FA0(v118);
        sub_1909FA0(v112);
        if ( v107 )
          j_j___libc_free_0(v107, v109 - v107);
        goto LABEL_37;
      }
LABEL_51:
      v45 = &v129;
      v46 = (__int64 *)v32;
      v129 = 0;
      v130 = 0;
      v131 = &v129;
      v132 = &v129;
      v133 = 0;
      v95 = sub_157ED20(v32);
      v50 = v108;
      while ( v108 != v107 )
      {
        v51 = *((_QWORD *)v50 - 1);
        v108 = v50 - 8;
        v101 = v51;
        v52 = sub_15F4880(v51);
        sub_15F2120(v52, v95);
        if ( *a1 )
          sub_14139C0(*a1, v52);
        v53 = v130;
        if ( v130 )
        {
          v54 = (unsigned __int64)v45;
          do
          {
            while ( 1 )
            {
              v55 = v53[2];
              v56 = v53[3];
              if ( v53[4] >= v101 )
                break;
              v53 = (_QWORD *)v53[3];
              if ( !v56 )
                goto LABEL_59;
            }
            v54 = (unsigned __int64)v53;
            v53 = (_QWORD *)v53[2];
          }
          while ( v55 );
LABEL_59:
          if ( (int *)v54 != v45 && *(_QWORD *)(v54 + 32) <= v101 )
            goto LABEL_62;
        }
        else
        {
          v54 = (unsigned __int64)v45;
        }
        v103 = &v101;
        v54 = sub_190F5E0(&v128, (_QWORD *)v54, &v103);
LABEL_62:
        *(_QWORD *)(v54 + 40) = v52;
        if ( (*(_DWORD *)(v52 + 20) & 0xFFFFFFF) != 0 )
        {
          v57 = v45;
          v58 = 0;
          v59 = 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF);
          v60 = v57;
          do
          {
            if ( (*(_BYTE *)(v52 + 23) & 0x40) != 0 )
            {
              v61 = *(_QWORD *)(v52 - 8);
            }
            else
            {
              v56 = 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF);
              v61 = v52 - v56;
            }
            v54 = *(_QWORD *)(v61 + v58);
            if ( *(_BYTE *)(v54 + 16) > 0x17u )
            {
              v56 = (__int64)v130;
              v102 = *(_QWORD *)(v61 + v58);
              if ( v130 )
              {
                v62 = (int *)v130;
                v63 = v60;
                do
                {
                  while ( 1 )
                  {
                    v64 = *((_QWORD *)v62 + 2);
                    v65 = *((_QWORD *)v62 + 3);
                    if ( *((_QWORD *)v62 + 4) >= v54 )
                      break;
                    v62 = (int *)*((_QWORD *)v62 + 3);
                    if ( !v65 )
                      goto LABEL_72;
                  }
                  v63 = v62;
                  v62 = (int *)*((_QWORD *)v62 + 2);
                }
                while ( v64 );
LABEL_72:
                if ( v63 != v60 && *((_QWORD *)v63 + 4) <= v54 )
                {
                  v66 = v60;
                  do
                  {
                    while ( 1 )
                    {
                      v67 = *(_QWORD *)(v56 + 16);
                      v68 = *(_QWORD *)(v56 + 24);
                      if ( *(_QWORD *)(v56 + 32) >= v54 )
                        break;
                      v56 = *(_QWORD *)(v56 + 24);
                      if ( !v68 )
                        goto LABEL_78;
                    }
                    v66 = (int *)v56;
                    v56 = *(_QWORD *)(v56 + 16);
                  }
                  while ( v67 );
LABEL_78:
                  if ( v66 == v60 || *((_QWORD *)v66 + 4) > v54 )
                  {
                    v103 = &v102;
                    v69 = sub_190F5E0(&v128, v66, &v103);
                    v54 = v102;
                    v66 = (int *)v69;
                  }
                  sub_1648780(v52, v54, *((_QWORD *)v66 + 5));
                }
              }
            }
            v58 += 24;
          }
          while ( v58 != v59 );
          v45 = v60;
        }
        if ( dword_4FAEC00 )
        {
          v74 = sub_16BA580((unsigned int)dword_4FAEC00, v54, v56);
          v75 = *(_QWORD *)(v74 + 24);
          v76 = v74;
          if ( (unsigned __int64)(*(_QWORD *)(v74 + 16) - v75) <= 6 )
          {
            v76 = sub_16E7EE0(v74, "clone: ", 7u);
          }
          else
          {
            *(_DWORD *)v75 = 1852796003;
            *(_WORD *)(v75 + 4) = 14949;
            *(_BYTE *)(v75 + 6) = 32;
            *(_QWORD *)(v74 + 24) += 7LL;
          }
          sub_155C2B0(v52, v76, 0);
          v77 = *(_BYTE **)(v76 + 24);
          if ( *(_BYTE **)(v76 + 16) == v77 )
          {
            sub_16E7EE0(v76, "\n", 1u);
          }
          else
          {
            *v77 = 10;
            ++*(_QWORD *)(v76 + 24);
          }
        }
        v35 = (__int64)&v101;
        v46 = &v116;
        if ( sub_190D300((__int64)&v116, &v101) )
        {
          v92 = v45;
          for ( j = sub_190D3E0((__int64)&v116, &v101); ; j = (_QWORD *)sub_220EEE0(j) )
          {
            v35 = (__int64)&v101;
            v46 = &v116;
            sub_190D3E0((__int64)&v116, &v101);
            if ( j == (_QWORD *)v47 )
              break;
            v103 = (unsigned __int64 *)j[5];
            v79 = sub_190D3E0((__int64)&v110, (unsigned __int64 *)&v103);
            sub_190D3E0((__int64)&v110, (unsigned __int64 *)&v103);
            v80 = v101;
            v82 = v81;
            while ( v79 != v82 )
            {
              v83 = sub_220EEE0(v79);
              if ( v79[5] == v80 )
              {
                v84 = sub_220F330(v79, &v111);
                j_j___libc_free_0(v84, 48);
                --v115;
                break;
              }
              v79 = (_QWORD *)v83;
            }
            if ( !sub_190D300((__int64)&v110, (unsigned __int64 *)&v103) )
            {
              v85 = v108;
              if ( v108 == v109 )
              {
                sub_170B610((__int64)&v107, v108, &v103);
              }
              else
              {
                if ( v108 )
                {
                  *(_QWORD *)v108 = v103;
                  v85 = v108;
                }
                v108 = v85 + 8;
              }
            }
          }
          v45 = v92;
        }
        v50 = v108;
      }
      if ( !dword_4FAEC00 )
        goto LABEL_88;
      v87 = sub_16BA580((__int64)v46, v35, v47);
      v88 = *(void **)(v87 + 24);
      if ( *(_QWORD *)(v87 + 16) - (_QWORD)v88 <= 9u )
      {
        v87 = sub_16E7EE0(v87, "Removing: ", 0xAu);
      }
      else
      {
        qmemcpy(v88, "Removing: ", 10);
        *(_QWORD *)(v87 + 24) += 10LL;
      }
      sub_155C2B0((__int64)v93, v87, 0);
      v89 = *(_BYTE **)(v87 + 24);
      if ( *(_BYTE **)(v87 + 16) == v89 )
        break;
      *v89 = 10;
      ++*(_QWORD *)(v87 + 24);
      v70 = (int *)v130;
      if ( v130 )
      {
LABEL_89:
        v71 = v45;
        do
        {
          while ( 1 )
          {
            v72 = *((_QWORD *)v70 + 2);
            v73 = *((_QWORD *)v70 + 3);
            if ( *((_QWORD *)v70 + 4) >= v100 )
              break;
            v70 = (int *)*((_QWORD *)v70 + 3);
            if ( !v73 )
              goto LABEL_93;
          }
          v71 = v70;
          v70 = (int *)*((_QWORD *)v70 + 2);
        }
        while ( v72 );
LABEL_93:
        if ( v71 != v45 && *((_QWORD *)v71 + 4) <= v100 )
          goto LABEL_96;
        goto LABEL_95;
      }
LABEL_137:
      v71 = v45;
LABEL_95:
      v103 = &v100;
      v71 = (int *)sub_190F5E0(&v128, v71, &v103);
LABEL_96:
      sub_164D160((__int64)v93, *((_QWORD *)v71 + 5), si128, a4, a5, a6, v48, v49, a9, a10);
      sub_15F20C0(v93);
      sub_1909FA0((__int64)v130);
      sub_1909DD0(v124);
      sub_1909FA0(v118);
      sub_1909FA0(v112);
      if ( v107 )
        j_j___libc_free_0(v107, v109 - v107);
      v29 = 1;
LABEL_37:
      v28 = v104;
      v30 = (unsigned int)++v94;
      if ( v94 == (v105 - v104) >> 3 )
      {
        v90 = v106 - v104;
        goto LABEL_141;
      }
    }
    sub_16E7EE0(v87, "\n", 1u);
LABEL_88:
    v70 = (int *)v130;
    if ( v130 )
      goto LABEL_89;
    goto LABEL_137;
  }
  v29 = 0;
  v90 = v106 - v104;
LABEL_141:
  if ( v28 )
    j_j___libc_free_0(v28, v90);
  return v29;
}
