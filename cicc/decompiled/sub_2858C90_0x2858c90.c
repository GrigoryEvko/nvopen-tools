// Function: sub_2858C90
// Address: 0x2858c90
//
char __fastcall sub_2858C90(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 *v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 *v7; // r14
  __int16 v8; // ax
  __int64 v9; // rbx
  __int64 *v10; // r15
  unsigned int v11; // r12d
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // r10
  __int64 v19; // r9
  _BYTE *v20; // rsi
  __int64 *v21; // rax
  _QWORD *v22; // rax
  const __m128i *v23; // r13
  char v24; // al
  __int64 v25; // r10
  __int64 v26; // r15
  __int64 *v27; // rax
  __int64 v28; // r14
  __int64 v29; // r14
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __m128i *v33; // rax
  __int64 v34; // r13
  __int64 v35; // r12
  __int64 v36; // rbx
  bool v37; // al
  __int64 v38; // r13
  __int64 v39; // r9
  __int64 *v40; // r15
  __int64 v41; // rbx
  __int64 v42; // r14
  _QWORD *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 *v46; // rsi
  __int64 *v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 *v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rdx
  __int64 *v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 *v60; // rax
  __int64 v61; // rcx
  __int64 *v62; // r15
  __int64 v63; // rsi
  unsigned int v64; // eax
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 v67; // r14
  __int64 v68; // rbx
  __int64 *v69; // r12
  _QWORD *v70; // rax
  __int64 v71; // rdi
  __int64 *v72; // rax
  __int64 v73; // rcx
  unsigned __int64 v74; // rsi
  unsigned __int64 v75; // rdx
  __int64 **v76; // rbx
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rax
  _QWORD *v79; // r14
  __int64 *v80; // rdi
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  unsigned __int64 v84; // r10
  __int64 v85; // r15
  __int64 v86; // r12
  __int64 v87; // r14
  unsigned __int64 v88; // rbx
  unsigned __int64 v89; // r14
  __int64 v90; // r15
  __int64 v91; // rbx
  __int64 v92; // r8
  const void *v93; // rsi
  int v94; // r14d
  unsigned __int64 v95; // r14
  unsigned __int64 v96; // r15
  __int64 v97; // r13
  __int64 v98; // rdi
  char *v99; // rbx
  __int64 *v101; // [rsp+0h] [rbp-E0h]
  __int64 *v102; // [rsp+0h] [rbp-E0h]
  __int64 v103; // [rsp+8h] [rbp-D8h]
  unsigned int v105; // [rsp+18h] [rbp-C8h]
  __int64 v106; // [rsp+18h] [rbp-C8h]
  __int64 v107; // [rsp+18h] [rbp-C8h]
  __int64 v109; // [rsp+28h] [rbp-B8h]
  __int64 *v110; // [rsp+30h] [rbp-B0h]
  __int64 *v111; // [rsp+30h] [rbp-B0h]
  __int64 *v112; // [rsp+30h] [rbp-B0h]
  char v113; // [rsp+30h] [rbp-B0h]
  __int64 v114; // [rsp+30h] [rbp-B0h]
  __int64 v116; // [rsp+40h] [rbp-A0h]
  __int64 v117; // [rsp+40h] [rbp-A0h]
  __int64 v118; // [rsp+40h] [rbp-A0h]
  __int64 v119; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v120; // [rsp+40h] [rbp-A0h]
  __int64 v121; // [rsp+40h] [rbp-A0h]
  __int64 v122; // [rsp+40h] [rbp-A0h]
  __int64 v123; // [rsp+40h] [rbp-A0h]
  __int64 v125; // [rsp+48h] [rbp-98h]
  __int64 v126; // [rsp+48h] [rbp-98h]
  __int64 v127; // [rsp+48h] [rbp-98h]
  __int64 v128; // [rsp+48h] [rbp-98h]
  unsigned int v129; // [rsp+48h] [rbp-98h]
  __int64 v130; // [rsp+48h] [rbp-98h]
  __int64 v131; // [rsp+48h] [rbp-98h]
  __int64 *v132; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v133; // [rsp+58h] [rbp-88h]
  __int64 v134; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v135; // [rsp+68h] [rbp-78h]
  _QWORD v136[14]; // [rsp+70h] [rbp-70h] BYREF

  v116 = a3;
  if ( *(_BYTE *)a3 == 67 )
    v116 = *(_QWORD *)(a3 - 32);
  v4 = sub_DD8400(*(_QWORD *)(a1 + 8), v116);
  v6 = a1;
  v109 = (__int64)v4;
  v7 = v4;
  v8 = *((_WORD *)v4 + 12);
LABEL_4:
  while ( 2 )
  {
    switch ( v8 )
    {
      case 0:
      case 1:
        v7 = 0;
        goto LABEL_7;
      case 2:
      case 3:
      case 4:
        v7 = (__int64 *)v7[4];
        v8 = *((_WORD *)v7 + 12);
        continue;
      case 5:
        v49 = v7[4];
        v50 = v49 + 8 * v7[5];
        if ( v49 == v50 )
          goto LABEL_7;
        break;
      case 8:
        v7 = *(__int64 **)v7[4];
        v8 = *((_WORD *)v7 + 12);
        continue;
      default:
        goto LABEL_7;
    }
    break;
  }
  while ( 1 )
  {
    v8 = *(_WORD *)(*(_QWORD *)(v50 - 8) + 24LL);
    if ( v8 == 5 )
    {
      v7 = *(__int64 **)(v50 - 8);
      goto LABEL_4;
    }
    if ( v8 != 6 )
      break;
    v50 -= 8;
    if ( v49 == v50 )
      goto LABEL_7;
  }
  v7 = *(__int64 **)(v50 - 8);
LABEL_7:
  v105 = *(_DWORD *)(a1 + 36464);
  if ( !v105 )
  {
    v11 = 0;
    LOBYTE(v27) = 0;
    goto LABEL_28;
  }
  v125 = *(unsigned int *)(a1 + 36464);
  v9 = 0;
  v10 = (__int64 *)v6;
  do
  {
    v12 = 48 * v9;
    v13 = 48 * v9 + v10[4557];
    if ( *(__int64 **)(v13 + 40) == v7 )
    {
      v14 = *(_QWORD *)v13 + 24LL * *(unsigned int *)(v13 + 8) - 24;
      v15 = *(_QWORD *)(v14 + 8);
      if ( *(_BYTE *)v15 == 67 )
        v15 = *(_QWORD *)(v15 - 32);
      if ( *(_QWORD *)(v15 + 8) == *(_QWORD *)(v116 + 8) && (*(_BYTE *)a2 != 84 || **(_BYTE **)v14 != 84) )
      {
        v16 = sub_DD8400(v10[1], v15);
        v110 = sub_DCC810((__int64 *)v10[1], v109, (__int64)v16, 0, 0);
        if ( !sub_D96A50((__int64)v110) && sub_DADE90(v10[1], (__int64)v110, v10[7]) )
        {
          v18 = v110;
          v19 = v10[1];
          if ( !*((_WORD *)v110 + 12) )
            goto LABEL_22;
          v20 = *(_BYTE **)(*(_QWORD *)v13 + 8LL);
          if ( *v20 == 67 )
            v20 = (_BYTE *)*((_QWORD *)v20 - 4);
          v101 = v110;
          v111 = (__int64 *)v10[1];
          v21 = sub_DD8400((__int64)v111, (__int64)v20);
          v22 = sub_DCC810(v111, v109, (__int64)v21, 0, 0);
          v19 = (__int64)v111;
          v18 = v101;
          if ( *((_WORD *)v22 + 12) )
          {
LABEL_22:
            BYTE4(v135) = 1;
            v112 = v18;
            v132 = 0;
            v133 = (unsigned __int64)v136;
            v23 = (const __m128i *)&v132;
            v134 = 8;
            LODWORD(v135) = 0;
            v24 = sub_2856C60(v18, (__int64)&v132, v19, v17, v5, v19);
            v25 = (__int64)v112;
            if ( !BYTE4(v135) )
            {
              v102 = v112;
              v113 = v24;
              _libc_free(v133);
              v25 = (__int64)v102;
              v24 = v113;
            }
            if ( !v24 )
            {
              v6 = (__int64)v10;
              v26 = v25;
LABEL_33:
              v28 = *(_QWORD *)(v6 + 36456);
              v134 = v26;
              v132 = a2;
              v29 = v12 + v28;
              v133 = a3;
              v30 = *(unsigned int *)(v29 + 8);
              v31 = v30 + 1;
              if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v29 + 12) )
              {
                v92 = *(_QWORD *)v29;
                v93 = (const void *)(v29 + 16);
                if ( *(_QWORD *)v29 > (unsigned __int64)&v132
                  || (v122 = *(_QWORD *)v29, (unsigned __int64)&v132 >= v92 + 24 * v30) )
                {
                  v131 = v6;
                  sub_C8D5F0(v29, v93, v31, 0x18u, v92, v6);
                  v32 = *(_QWORD *)v29;
                  v30 = *(unsigned int *)(v29 + 8);
                  v6 = v131;
                }
                else
                {
                  v130 = v6;
                  sub_C8D5F0(v29, v93, v31, 0x18u, v92, v6);
                  v32 = *(_QWORD *)v29;
                  v30 = *(unsigned int *)(v29 + 8);
                  v6 = v130;
                  v23 = (const __m128i *)((char *)&v132 + *(_QWORD *)v29 - v122);
                }
              }
              else
              {
                v32 = *(_QWORD *)v29;
              }
              v33 = (__m128i *)(v32 + 24 * v30);
              *v33 = _mm_loadu_si128(v23);
              v33[1].m128i_i64[0] = v23[1].m128i_i64[0];
              ++*(_DWORD *)(v29 + 8);
              v34 = *(_QWORD *)a4;
              goto LABEL_36;
            }
          }
        }
      }
    }
    v11 = ++v9;
  }
  while ( v9 != v125 );
  v6 = (__int64)v10;
  if ( v11 != v105 )
  {
    v9 = v11;
    v26 = 0;
    v23 = (const __m128i *)&v132;
    v12 = 48LL * v11;
    goto LABEL_33;
  }
  LOBYTE(v27) = v105 > 7;
LABEL_28:
  if ( *(_BYTE *)a2 != 84 && !(_BYTE)v27 )
  {
    LOBYTE(v27) = v109;
    if ( *(_WORD *)(v109 + 24) == 8 )
    {
      v134 = (__int64)a2;
      v73 = *(unsigned int *)(v6 + 36464);
      v129 = v11 + 1;
      v74 = v73 + 1;
      v136[1] = v7;
      v75 = v73;
      v76 = &v132;
      v135 = a3;
      v77 = *(unsigned int *)(v6 + 36468);
      v132 = &v134;
      v136[0] = v109;
      v133 = 0x100000001LL;
      v78 = *(_QWORD *)(v6 + 36456);
      if ( v73 + 1 > v77 )
      {
        v123 = v6;
        v98 = v6 + 36456;
        if ( v78 > (unsigned __int64)&v132 || (v75 = v78 + 48 * v73, (unsigned __int64)&v132 >= v75) )
        {
          sub_2850DE0(v98, v74, v75, v73, v5, v6);
          v6 = v123;
          v76 = &v132;
          v73 = *(unsigned int *)(v123 + 36464);
          v78 = *(_QWORD *)(v123 + 36456);
          v75 = v73;
        }
        else
        {
          v99 = (char *)&v132 - v78;
          sub_2850DE0(v98, v74, v75, v73, v5, v6);
          v6 = v123;
          v78 = *(_QWORD *)(v123 + 36456);
          v73 = *(unsigned int *)(v123 + 36464);
          v76 = (__int64 **)&v99[v78];
          v75 = v73;
        }
      }
      v79 = (_QWORD *)(v78 + 48 * v73);
      if ( v79 )
      {
        *v79 = v79 + 2;
        v79[1] = 0x100000000LL;
        if ( *((_DWORD *)v76 + 2) )
        {
          v121 = v6;
          sub_284FFC0((__int64)v79, (char **)v76, v75, v73, v5, v6);
          v6 = v121;
        }
        v79[5] = v76[5];
        LODWORD(v75) = *(_DWORD *)(v6 + 36464);
      }
      v80 = v132;
      *(_DWORD *)(v6 + 36464) = v75 + 1;
      if ( v80 != &v134 )
      {
        v119 = v6;
        _libc_free((unsigned __int64)v80);
        v6 = v119;
      }
      v9 = v11;
      v81 = *(unsigned int *)(a4 + 8);
      v12 = 48LL * v11;
      if ( v129 == v81 )
      {
        v34 = *(_QWORD *)a4;
        v26 = v109;
      }
      else
      {
        v120 = (unsigned __int64)v129 << 7;
        if ( v129 < v81 )
        {
          v34 = *(_QWORD *)a4;
          v95 = *(_QWORD *)a4 + (v81 << 7);
          v96 = *(_QWORD *)a4 + v120;
          if ( v95 != v96 )
          {
            v97 = v6;
            do
            {
              v95 -= 128LL;
              if ( !*(_BYTE *)(v95 + 92) )
                _libc_free(*(_QWORD *)(v95 + 72));
              if ( !*(_BYTE *)(v95 + 28) )
                _libc_free(*(_QWORD *)(v95 + 8));
            }
            while ( v96 != v95 );
            v6 = v97;
            v34 = *(_QWORD *)a4;
          }
        }
        else
        {
          if ( v129 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
          {
            v106 = v6;
            v83 = sub_C8D7D0(a4, a4 + 16, v129, 0x80u, (unsigned __int64 *)&v132, v6);
            v6 = v106;
            v34 = v83;
            v84 = *(_QWORD *)a4 + ((unsigned __int64)*(unsigned int *)(a4 + 8) << 7);
            if ( *(_QWORD *)a4 != v84 )
            {
              v107 = v12;
              v85 = v83;
              v86 = *(_QWORD *)a4;
              v87 = v6;
              v103 = v9;
              v88 = *(_QWORD *)a4 + ((unsigned __int64)*(unsigned int *)(a4 + 8) << 7);
              do
              {
                if ( v85 )
                {
                  sub_C8CF70(v85, (void *)(v85 + 32), 4, v86 + 32, v86);
                  sub_C8CF70(v85 + 64, (void *)(v85 + 96), 4, v86 + 96, v86 + 64);
                }
                v86 += 128;
                v85 += 128;
              }
              while ( v88 != v86 );
              v6 = v87;
              v12 = v107;
              v9 = v103;
              v84 = *(_QWORD *)a4;
              v89 = *(_QWORD *)a4 + ((unsigned __int64)*(unsigned int *)(a4 + 8) << 7);
              if ( *(_QWORD *)a4 != v89 )
              {
                v90 = v6;
                v91 = *(_QWORD *)a4;
                do
                {
                  v89 -= 128LL;
                  if ( !*(_BYTE *)(v89 + 92) )
                    _libc_free(*(_QWORD *)(v89 + 72));
                  if ( !*(_BYTE *)(v89 + 28) )
                    _libc_free(*(_QWORD *)(v89 + 8));
                }
                while ( v89 != v91 );
                v9 = v103;
                v6 = v90;
                v84 = *(_QWORD *)a4;
              }
            }
            v94 = (int)v132;
            if ( a4 + 16 != v84 )
            {
              v114 = v6;
              _libc_free(v84);
              v6 = v114;
            }
            *(_QWORD *)a4 = v34;
            v81 = *(unsigned int *)(a4 + 8);
            *(_DWORD *)(a4 + 12) = v94;
          }
          else
          {
            v34 = *(_QWORD *)a4;
          }
          v82 = v34 + (v81 << 7);
          if ( v82 != v34 + v120 )
          {
            do
            {
              if ( v82 )
              {
                memset((void *)v82, 0, 0x80u);
                *(_BYTE *)(v82 + 28) = 1;
                *(_QWORD *)(v82 + 8) = v82 + 32;
                *(_DWORD *)(v82 + 16) = 4;
                *(_QWORD *)(v82 + 72) = v82 + 96;
                *(_DWORD *)(v82 + 80) = 4;
                *(_BYTE *)(v82 + 92) = 1;
              }
              v82 += 128LL;
            }
            while ( v34 + v120 != v82 );
            v34 = *(_QWORD *)a4;
          }
        }
        v26 = v109;
        *(_DWORD *)(a4 + 8) = v129;
      }
LABEL_36:
      v35 = *(_QWORD *)(v6 + 36456) + v12;
      v126 = v6;
      v36 = v9 << 7;
      v37 = sub_D968A0(v26);
      v38 = v36 + v34;
      v39 = v126;
      if ( v37 )
      {
LABEL_37:
        if ( !*(_QWORD *)(a3 + 16) )
        {
LABEL_46:
          v45 = *(_QWORD *)a4 + v36;
          if ( *(_BYTE *)(v45 + 28) )
          {
            v27 = (__int64 *)*(unsigned int *)(v45 + 20);
            v46 = *(__int64 **)(v45 + 8);
            v47 = &v46[(_QWORD)v27];
            if ( v46 != v47 )
            {
              v27 = *(__int64 **)(v45 + 8);
              while ( a2 != (__int64 *)*v27 )
              {
                if ( v47 == ++v27 )
                  return (char)v27;
              }
              v48 = (unsigned int)(*(_DWORD *)(v45 + 20) - 1);
              *(_DWORD *)(v45 + 20) = v48;
              *v27 = v46[v48];
              ++*(_QWORD *)v45;
            }
          }
          else
          {
            v27 = sub_C8CA60(v45, (__int64)a2);
            if ( v27 )
            {
              *v27 = -2;
              ++*(_DWORD *)(v45 + 24);
              ++*(_QWORD *)v45;
            }
          }
          return (char)v27;
        }
        v117 = v36;
        v40 = (__int64 *)v39;
        v41 = *(_QWORD *)(a3 + 16);
        while ( 1 )
        {
          v42 = *(_QWORD *)(v41 + 24);
          if ( *(_BYTE *)v42 <= 0x1Cu )
            goto LABEL_44;
          v43 = *(_QWORD **)v35;
          v44 = *(_QWORD *)v35 + 24LL * *(unsigned int *)(v35 + 8);
          if ( v44 != *(_QWORD *)v35 )
          {
            while ( *v43 != v42 )
            {
              v43 += 3;
              if ( v43 == (_QWORD *)v44 )
                goto LABEL_59;
            }
            goto LABEL_44;
          }
LABEL_59:
          if ( !sub_D97040(v40[1], *(_QWORD *)(v42 + 8)) || *((_WORD *)sub_DD8400(v40[1], v42) + 12) == 15 )
            goto LABEL_68;
          v55 = *v40;
          if ( *(_BYTE *)(*v40 + 68) )
          {
            v56 = *(__int64 **)(v55 + 48);
            v51 = &v56[*(unsigned int *)(v55 + 60)];
            if ( v56 != v51 )
            {
              while ( v42 != *v56 )
              {
                if ( v51 == ++v56 )
                  goto LABEL_68;
              }
              goto LABEL_44;
            }
            goto LABEL_68;
          }
          if ( sub_C8CA60(v55 + 40, v42) )
          {
LABEL_44:
            v41 = *(_QWORD *)(v41 + 8);
            if ( !v41 )
              goto LABEL_45;
          }
          else
          {
LABEL_68:
            if ( *(_BYTE *)(v38 + 92) )
            {
              v57 = *(__int64 **)(v38 + 72);
              v58 = *(unsigned int *)(v38 + 84);
              v51 = &v57[v58];
              if ( v57 != v51 )
              {
                while ( v42 != *v57 )
                {
                  if ( v51 == ++v57 )
                    goto LABEL_74;
                }
                goto LABEL_44;
              }
LABEL_74:
              if ( (unsigned int)v58 >= *(_DWORD *)(v38 + 80) )
                goto LABEL_75;
              *(_DWORD *)(v38 + 84) = v58 + 1;
              *v51 = v42;
              ++*(_QWORD *)(v38 + 64);
              v41 = *(_QWORD *)(v41 + 8);
              if ( !v41 )
              {
LABEL_45:
                v36 = v117;
                goto LABEL_46;
              }
            }
            else
            {
LABEL_75:
              sub_C8CC70(v38 + 64, v42, (__int64)v51, v52, v53, v54);
              v41 = *(_QWORD *)(v41 + 8);
              if ( !v41 )
                goto LABEL_45;
            }
          }
        }
      }
      v59 = *(unsigned __int8 *)(v38 + 92);
      v60 = *(__int64 **)(v38 + 72);
      v61 = v36 + *(_QWORD *)a4;
      if ( (_BYTE)v59 )
      {
        v62 = &v60[*(unsigned int *)(v38 + 84)];
        if ( v60 != v62 )
          goto LABEL_83;
        ++*(_QWORD *)(v38 + 64);
LABEL_90:
        *(_QWORD *)(v38 + 84) = 0;
        goto LABEL_37;
      }
      v63 = *(unsigned int *)(v38 + 80);
      v62 = &v60[v63];
      if ( v60 == v62 )
      {
        ++*(_QWORD *)(v38 + 64);
      }
      else
      {
LABEL_83:
        while ( 1 )
        {
          v63 = *v60;
          if ( (unsigned __int64)*v60 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v62 == ++v60 )
            goto LABEL_85;
        }
        if ( v60 != v62 )
        {
          v66 = *(unsigned __int8 *)(v61 + 28);
          v128 = v36;
          v67 = v39;
          v68 = v36 + *(_QWORD *)a4;
          v118 = v35;
          v69 = v60;
          if ( !(_BYTE)v66 )
            goto LABEL_103;
LABEL_93:
          v70 = *(_QWORD **)(v68 + 8);
          v71 = *(unsigned int *)(v68 + 20);
          v59 = (__int64)&v70[v71];
          if ( v70 == (_QWORD *)v59 )
          {
LABEL_104:
            if ( (unsigned int)v71 < *(_DWORD *)(v68 + 16) )
            {
              *(_DWORD *)(v68 + 20) = v71 + 1;
              *(_QWORD *)v59 = v63;
              v66 = *(unsigned __int8 *)(v68 + 28);
              ++*(_QWORD *)v68;
              goto LABEL_97;
            }
            goto LABEL_103;
          }
          while ( v63 != *v70 )
          {
            if ( (_QWORD *)v59 == ++v70 )
              goto LABEL_104;
          }
LABEL_97:
          while ( 1 )
          {
            v72 = v69 + 1;
            if ( v69 + 1 == v62 )
              break;
            while ( 1 )
            {
              v63 = *v72;
              v69 = v72;
              if ( (unsigned __int64)*v72 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v62 == ++v72 )
                goto LABEL_100;
            }
            if ( v72 == v62 )
              break;
            if ( (_BYTE)v66 )
              goto LABEL_93;
LABEL_103:
            sub_C8CC70(v68, v63, v59, v61, v66, v39);
            v66 = *(unsigned __int8 *)(v68 + 28);
          }
LABEL_100:
          v36 = v128;
          v35 = v118;
          v39 = v67;
          LOBYTE(v59) = *(_BYTE *)(v38 + 92);
        }
LABEL_85:
        ++*(_QWORD *)(v38 + 64);
        if ( (_BYTE)v59 )
          goto LABEL_90;
      }
      v127 = v39;
      v64 = 4 * (*(_DWORD *)(v38 + 84) - *(_DWORD *)(v38 + 88));
      v65 = *(unsigned int *)(v38 + 80);
      if ( v64 < 0x20 )
        v64 = 32;
      if ( (unsigned int)v65 > v64 )
      {
        sub_C8C990(v38 + 64, v63);
        v39 = v127;
        goto LABEL_37;
      }
      memset(*(void **)(v38 + 72), -1, 8 * v65);
      v39 = v127;
      goto LABEL_90;
    }
  }
  return (char)v27;
}
