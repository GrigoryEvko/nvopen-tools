// Function: sub_77C870
// Address: 0x77c870
//
__int64 __fastcall sub_77C870(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 i; // rax
  char v10; // dl
  __int64 v11; // r9
  __int64 v12; // rsi
  unsigned __int32 v13; // r14d
  _QWORD *v15; // rdx
  FILE *v16; // rsi
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // r15
  int v20; // eax
  __int64 v21; // r9
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r9
  unsigned __int64 v25; // r15
  char j; // al
  int v27; // edx
  unsigned __int64 v28; // r15
  unsigned int v29; // ecx
  __int64 v30; // rax
  __int64 v31; // r11
  unsigned int v32; // ebx
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  int v35; // edx
  unsigned int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // r11
  unsigned __int64 v39; // rsi
  unsigned int v40; // ecx
  __int64 v41; // rax
  __int64 v42; // r11
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // r15
  unsigned int v45; // edx
  __int64 v46; // r9
  unsigned int v47; // ecx
  __int64 v48; // rax
  __int64 v49; // r10
  __int64 v50; // r8
  __int64 v51; // rsi
  __int64 v52; // rcx
  char *v53; // rbx
  __int64 *v54; // r15
  __int64 v55; // rax
  __int64 v56; // r14
  char v57; // al
  __int64 v58; // rsi
  __int64 *v59; // r14
  unsigned int v60; // ebx
  __int64 v61; // r13
  int v62; // r9d
  __int64 v63; // rdi
  _QWORD *v64; // rax
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 *v67; // r14
  __int64 v68; // rbx
  __int64 v69; // r13
  __int64 v70; // rdx
  __m128i *v71; // rax
  int v72; // eax
  __int64 v73; // rax
  char v74; // dl
  __int64 v75; // r13
  size_t v76; // rax
  __int64 v77; // r13
  __m128i *v78; // rax
  char *v79; // rdx
  char *v80; // r8
  __m128i *v81; // rcx
  const __m128i *v82; // rdi
  __m128i *v83; // r9
  __int64 v84; // [rsp+0h] [rbp-170h]
  __int64 v85; // [rsp+8h] [rbp-168h]
  __int64 v86; // [rsp+10h] [rbp-160h]
  unsigned int v87; // [rsp+20h] [rbp-150h]
  __int64 v88; // [rsp+20h] [rbp-150h]
  __int64 v89; // [rsp+28h] [rbp-148h]
  __int64 v90; // [rsp+28h] [rbp-148h]
  __int64 v91; // [rsp+28h] [rbp-148h]
  unsigned int v92; // [rsp+28h] [rbp-148h]
  __int64 v93; // [rsp+30h] [rbp-140h]
  __int64 v94; // [rsp+30h] [rbp-140h]
  __int64 v95; // [rsp+38h] [rbp-138h]
  unsigned __int64 v96; // [rsp+38h] [rbp-138h]
  int v97; // [rsp+40h] [rbp-130h]
  unsigned int v98; // [rsp+40h] [rbp-130h]
  __int64 v99; // [rsp+48h] [rbp-128h]
  unsigned int v100; // [rsp+48h] [rbp-128h]
  __int64 v101; // [rsp+48h] [rbp-128h]
  int v102; // [rsp+50h] [rbp-120h]
  _QWORD *v103; // [rsp+50h] [rbp-120h]
  __int64 v104; // [rsp+50h] [rbp-120h]
  __int64 v105; // [rsp+50h] [rbp-120h]
  int v106; // [rsp+5Ch] [rbp-114h]
  __int64 v107; // [rsp+60h] [rbp-110h]
  int v108; // [rsp+60h] [rbp-110h]
  __int64 v109; // [rsp+60h] [rbp-110h]
  __int64 v110; // [rsp+68h] [rbp-108h]
  char *s; // [rsp+70h] [rbp-100h]
  char *sa; // [rsp+70h] [rbp-100h]
  char *sb; // [rsp+70h] [rbp-100h]
  __int64 v114; // [rsp+78h] [rbp-F8h]
  __int64 v115; // [rsp+78h] [rbp-F8h]
  int v116; // [rsp+78h] [rbp-F8h]
  unsigned int v117; // [rsp+78h] [rbp-F8h]
  __int64 v118; // [rsp+78h] [rbp-F8h]
  __int64 v119; // [rsp+78h] [rbp-F8h]
  unsigned int v120; // [rsp+88h] [rbp-E8h] BYREF
  int v121; // [rsp+8Ch] [rbp-E4h] BYREF
  char *v122; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v123; // [rsp+98h] [rbp-D8h]
  __int64 v124; // [rsp+A0h] [rbp-D0h]
  __m128i v125; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v126; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v127[20]; // [rsp+D0h] [rbp-A0h] BYREF

  v7 = *a4;
  v120 = 1;
  v124 = 0;
  v123 = 0;
  v122 = (char *)sub_823970(0);
  v8 = (__int64)v122;
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v10 = *(_BYTE *)(a1 + 132);
  if ( (v10 & 1) == 0 )
  {
    v120 = 0;
    v12 = 0;
    v13 = 0;
    if ( (v10 & 0x20) != 0 )
      goto LABEL_8;
    v15 = (_QWORD *)(a1 + 96);
    v16 = (FILE *)(a3 + 28);
    v17 = 2721;
LABEL_10:
    sub_6855B0(v17, v16, v15);
    sub_770D30(a1);
    goto LABEL_11;
  }
  if ( *(_BYTE *)v7 == 6 )
  {
    v11 = *(_QWORD *)(v7 + 8);
    if ( (*(_BYTE *)(v11 + 141) & 0x20) != 0 && (unsigned __int8)(*(_BYTE *)(v11 + 140) - 9) <= 2u )
    {
      v114 = *(_QWORD *)(v7 + 8);
      v18 = **(__int64 ***)(i + 168);
      v19 = *v18;
      v20 = sub_8D27E0(*(_QWORD *)(*v18 + 8));
      sub_620E00((_WORD *)a4[1], v20, v127, v125.m128i_i32);
      v13 = v125.m128i_i32[0];
      v21 = v114;
      if ( v125.m128i_i32[0] )
      {
        v120 = 0;
        v13 = 0;
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          v15 = (_QWORD *)(a1 + 96);
          v16 = (FILE *)(a3 + 28);
          v17 = 61;
          goto LABEL_10;
        }
      }
      else
      {
        v102 = v127[0];
        v106 = v127[0];
        if ( SLODWORD(v127[0]) > v123 )
        {
          v101 = v114;
          v119 = SLODWORD(v127[0]);
          v109 = sub_823970(32LL * SLODWORD(v127[0]));
          sub_823A00(v122, 32 * v123);
          v21 = v101;
          v122 = (char *)v109;
          v123 = v119;
        }
        v22 = (__int64 *)a4[2];
        v115 = v21;
        v23 = sub_8D46C0(*(_QWORD *)(*(_QWORD *)v19 + 8LL));
        v24 = v115;
        v25 = v23;
        v107 = *v22;
        v110 = v22[3];
        for ( j = *(_BYTE *)(v23 + 140); j == 12; j = *(_BYTE *)(v25 + 140) )
          v25 = *(_QWORD *)(v25 + 160);
        v27 = 16;
        if ( (unsigned __int8)(j - 2) > 1u )
        {
          v72 = sub_7764B0(a1, v25, &v120);
          v24 = v115;
          v27 = v72;
        }
        if ( !v120 )
          goto LABEL_92;
        if ( (unsigned __int8)(*(_BYTE *)(v25 + 140) - 9) > 2u )
          goto LABEL_92;
        v28 = *(_QWORD *)(v25 + 160);
        if ( !v28 )
          goto LABEL_92;
        v99 = v24;
        v116 = v27;
        if ( !(unsigned int)sub_8D2630(*(_QWORD *)(v28 + 120), v120) )
          goto LABEL_92;
        v29 = qword_4F08388 & (v28 >> 3);
        v30 = qword_4F08380 + 16LL * v29;
        v31 = *(_QWORD *)v30;
        if ( *(_QWORD *)v30 == v28 )
        {
LABEL_95:
          v32 = *(_DWORD *)(v30 + 8);
        }
        else
        {
          while ( v31 )
          {
            v29 = qword_4F08388 & (v29 + 1);
            v30 = qword_4F08380 + 16LL * v29;
            v31 = *(_QWORD *)v30;
            if ( v28 == *(_QWORD *)v30 )
              goto LABEL_95;
          }
          v32 = 0;
        }
        v33 = *(_QWORD *)(v28 + 112);
        if ( !v33 )
          goto LABEL_92;
        if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(v33 + 120)) )
          goto LABEL_92;
        v34 = sub_8D46C0(*(_QWORD *)(v33 + 120));
        if ( !(unsigned int)sub_8D29E0(v34) )
          goto LABEL_92;
        v35 = v116;
        v36 = qword_4F08388 & (v33 >> 3);
        v37 = qword_4F08380 + 16LL * v36;
        v38 = *(_QWORD *)v37;
        if ( v33 == *(_QWORD *)v37 )
        {
LABEL_96:
          v117 = *(_DWORD *)(v37 + 8);
        }
        else
        {
          while ( v38 )
          {
            v36 = qword_4F08388 & (v36 + 1);
            v37 = qword_4F08380 + 16LL * v36;
            v38 = *(_QWORD *)v37;
            if ( v33 == *(_QWORD *)v37 )
              goto LABEL_96;
          }
          v117 = 0;
        }
        v39 = *(_QWORD *)(v33 + 112);
        v95 = v99;
        v97 = v35;
        if ( !v39 || !(unsigned int)sub_8D3A00(*(_QWORD *)(v39 + 120)) )
          goto LABEL_92;
        v40 = qword_4F08388 & (v39 >> 3);
        v41 = qword_4F08380 + 16LL * v40;
        v42 = *(_QWORD *)v41;
        if ( *(_QWORD *)v41 == v39 )
        {
LABEL_124:
          v100 = *(_DWORD *)(v41 + 8);
        }
        else
        {
          while ( v42 )
          {
            v40 = qword_4F08388 & (v40 + 1);
            v41 = qword_4F08380 + 16LL * v40;
            v42 = *(_QWORD *)v41;
            if ( v39 == *(_QWORD *)v41 )
              goto LABEL_124;
          }
          v100 = 0;
        }
        v43 = *(_QWORD *)(v39 + 112);
        v93 = v95;
        v96 = v43;
        v44 = v43;
        if ( v43 && (unsigned int)sub_8D3A00(*(_QWORD *)(v43 + 120)) )
        {
          v45 = v97;
          v46 = v93;
          v47 = qword_4F08388 & (v44 >> 3);
          v48 = qword_4F08380 + 16LL * v47;
          v49 = *(_QWORD *)v48;
          if ( v44 == *(_QWORD *)v48 )
          {
LABEL_123:
            v98 = *(_DWORD *)(v48 + 8);
          }
          else
          {
            while ( v49 )
            {
              v47 = qword_4F08388 & (v47 + 1);
              v48 = qword_4F08380 + 16LL * v47;
              v49 = *(_QWORD *)v48;
              if ( v96 == *(_QWORD *)v48 )
                goto LABEL_123;
            }
            v98 = 0;
          }
          if ( v102 > 0 )
          {
            v50 = v32;
            v125 = 0;
            v51 = v107 + v32;
            v126 = 0;
            if ( ((unsigned __int8)(1 << ((v107 + v50 - v110) & 7))
                & *(_BYTE *)(v110 + -(((unsigned int)(v107 + v50 - v110) >> 3) + 10))) == 0 )
            {
LABEL_91:
              v120 = 0;
              sub_770DD0(0xABFu, (FILE *)(a3 + 28), a1);
              goto LABEL_11;
            }
            v52 = v117;
            v84 = v93;
            v118 = a3;
            v94 = v52;
            v86 = v45;
            v53 = (char *)v127;
            v54 = (__int64 *)(v52 + v107);
            v55 = v50 + v45 + v107 - v51;
            v108 = 0;
            v56 = v51;
            v85 = v55;
            while ( 1 )
            {
              if ( *(_BYTE *)v56 == 48 )
              {
                v73 = *(_QWORD *)(v56 + 8);
                v74 = *(_BYTE *)(v73 + 8);
                if ( v74 == 1 )
                {
                  *(_BYTE *)v56 = 2;
                  v75 = v118;
                  *(_QWORD *)(v56 + 8) = *(_QWORD *)(v73 + 32);
                  goto LABEL_100;
                }
                if ( v74 == 2 )
                {
                  *(_BYTE *)v56 = 59;
                  v75 = v118;
                  *(_QWORD *)(v56 + 8) = *(_QWORD *)(v73 + 32);
LABEL_100:
                  v120 = 0;
                  sub_770DD0(0xD2Fu, (FILE *)(v75 + 28), a1);
                  goto LABEL_11;
                }
                if ( v74 )
                  sub_721090();
                *(_BYTE *)v56 = 6;
                *(_QWORD *)(v56 + 8) = *(_QWORD *)(v73 + 32);
              }
              else if ( *(_BYTE *)v56 != 6 )
              {
                v75 = v118;
                goto LABEL_100;
              }
              v125.m128i_i64[0] = *(_QWORD *)(v56 + 8);
              if ( ((unsigned __int8)(1 << (((_BYTE)v54 - v110) & 7))
                  & *(_BYTE *)(v110 + -(((unsigned int)((_DWORD)v54 - v110) >> 3) + 10))) == 0 )
                goto LABEL_90;
              v57 = *((_BYTE *)v54 + 8);
              if ( (v57 & 1) != 0 )
              {
                if ( !(unsigned int)sub_710600(v54[2]) )
                {
LABEL_128:
                  v120 = 0;
                  sub_770DD0(0xA8Du, (FILE *)(v118 + 28), a1);
                  goto LABEL_11;
                }
                v57 = *((_BYTE *)v54 + 8);
              }
              if ( (v57 & 0x20) != 0 )
                goto LABEL_128;
              v58 = v54[3];
              if ( v58 )
              {
                v121 = 0;
                s = (char *)(v118 + 28);
                if ( (v57 & 0x21) != 0 )
                {
                  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
                  {
                    sub_6855B0(0xA8Du, (FILE *)(v118 + 28), (_QWORD *)(a1 + 96));
                    sub_770D30(a1);
                  }
                  goto LABEL_119;
                }
                if ( ((unsigned __int8)(1 << ((*v54 - v58) & 7))
                    & *(_BYTE *)(v58 + -((((unsigned int)*v54 - (unsigned int)v58) >> 3) + 10))) == 0 )
                {
LABEL_120:
                  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
                  {
                    sub_6855B0(0xABFu, (FILE *)s, (_QWORD *)(a1 + 96));
                    sub_770D30(a1);
                  }
LABEL_119:
                  v120 = 0;
                  goto LABEL_11;
                }
                v89 = v56;
                v59 = (__int64 *)v53;
                v60 = 0;
                v61 = *v54;
                while ( 1 )
                {
                  sub_620E00((_WORD *)(v61 + 16LL * v60), 0, v59, &v121);
                  v62 = v60++;
                  if ( !v127[0] )
                    break;
                  if ( ((unsigned __int8)(1 << ((v61 - v54[3]) & 7))
                      & *(_BYTE *)(v54[3] + -((((unsigned int)v61 - (unsigned int)v54[3]) >> 3) + 10))) == 0 )
                    goto LABEL_120;
                }
                v63 = v60;
                v87 = v62;
                v53 = (char *)v59;
                v56 = v89;
                v64 = sub_724830(v63);
                v65 = v87;
                v121 = 0;
                v125.m128i_i64[1] = (__int64)v64;
                v103 = v64;
                if ( (v54[1] & 0x21) != 0 || (v66 = v54[3]) == 0 )
                {
                  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
                  {
                    sub_6855B0(0xA8Du, (FILE *)s, (_QWORD *)(a1 + 96));
                    sub_770D30(a1);
                  }
                }
                else
                {
                  if ( v87 )
                  {
                    v88 = v89;
                    v67 = (__int64 *)v53;
                    v68 = 0;
                    v90 = (unsigned int)v65;
                    v69 = *v54;
                    while ( 1 )
                    {
                      v65 = (unsigned int)v68;
                      if ( ((unsigned __int8)(1 << ((v69 - v66) & 7))
                          & *(_BYTE *)(v66 + -(((unsigned int)(v69 - v66) >> 3) + 10))) == 0 )
                        break;
                      sub_620E00((_WORD *)(v69 + 16 * v68), 0, v67, &v121);
                      v65 = (unsigned int)(v68 + 1);
                      *((_BYTE *)v103 + v68++) = v127[0];
                      if ( v90 == v68 )
                      {
                        v53 = (char *)v67;
                        v56 = v88;
                        goto LABEL_76;
                      }
                      v66 = v54[3];
                    }
                    v53 = (char *)v67;
                    v56 = v88;
                    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
                    {
                      v92 = v65;
                      sub_6855B0(0xABFu, (FILE *)s, (_QWORD *)(a1 + 96));
                      sub_770D30(a1);
                      v65 = v92;
                    }
                  }
LABEL_76:
                  *((_BYTE *)v103 + v65) = 0;
                }
              }
              else
              {
                sprintf(v53, "__field_%u", v108);
                v76 = strlen(v53);
                v125.m128i_i64[1] = (__int64)sub_724830(v76 + 1);
                strcpy((char *)v125.m128i_i64[1], v53);
              }
              if ( ((unsigned __int8)(1 << (((_BYTE)v54 + v100 - v94 - v110) & 7))
                  & *(_BYTE *)(v110 + -((((unsigned int)v54 + v100 - (_DWORD)v94 - (unsigned int)v110) >> 3) + 10))) == 0 )
                goto LABEL_90;
              v121 = 0;
              sub_620E00((__int64 *)((char *)v54 + v100 - v94), 0, (__int64 *)v53, &v121);
              if ( v121 )
              {
LABEL_129:
                v120 = 0;
                sub_770DD0(0x3Du, (FILE *)(v118 + 28), a1);
                goto LABEL_11;
              }
              if ( v127[0] && !(unsigned int)sub_7A7520(v127[0], &v126) )
              {
                v120 = 0;
                sub_770DD0(0x294u, (FILE *)(v118 + 28), a1);
              }
              if ( ((unsigned __int8)(1 << (((_BYTE)v54 + v98 - v94 - v110) & 7))
                  & *(_BYTE *)(v110 + -((((unsigned int)v54 + v98 - (_DWORD)v94 - (unsigned int)v110) >> 3) + 10))) == 0 )
              {
LABEL_90:
                a3 = v118;
                goto LABEL_91;
              }
              v121 = 0;
              sub_620E00((__int64 *)((char *)v54 + v98 - v94), 0, (__int64 *)v53, &v121);
              if ( v121 )
                goto LABEL_129;
              v126.m128i_i64[1] = v127[0];
              if ( (v127[0] || !*(_BYTE *)v125.m128i_i64[1]) && !(unsigned int)sub_8D2930(*(_QWORD *)(v96 + 120)) )
              {
                v120 = 0;
                sub_770DD0(0x6Au, (FILE *)(v118 + 28), a1);
                goto LABEL_11;
              }
              v70 = v124;
              if ( v124 == v123 )
              {
                v77 = 2;
                if ( v124 > 1 )
                  v77 = v124 + (v124 >> 1) + 1;
                v91 = v123;
                v104 = v124;
                sa = v122;
                v78 = (__m128i *)sub_823970(32 * v77);
                v79 = (char *)v104;
                v80 = sa;
                v81 = v78;
                if ( v104 > 0 )
                {
                  v82 = (const __m128i *)sa;
                  v83 = &v78[2 * v104];
                  do
                  {
                    if ( v78 )
                    {
                      *v78 = _mm_loadu_si128(v82);
                      v78[1] = _mm_loadu_si128(v82 + 1);
                    }
                    v78 += 2;
                    v82 += 2;
                  }
                  while ( v83 != v78 );
                }
                v105 = (__int64)v81;
                sb = v79;
                sub_823A00(v80, 32 * v91);
                v123 = v77;
                v70 = (__int64)sb;
                v122 = (char *)v105;
              }
              v71 = (__m128i *)&v122[32 * v70];
              if ( v71 )
              {
                *v71 = _mm_loadu_si128(&v125);
                v71[1] = _mm_loadu_si128(&v126);
              }
              ++v108;
              v124 = v70 + 1;
              if ( v106 == v108 )
              {
                v46 = v84;
                a3 = v118;
                break;
              }
              v56 += v85;
              v125 = 0;
              v54 = (__int64 *)((char *)v54 + v86);
              v126 = 0;
              if ( ((unsigned __int8)(1 << ((v56 - v110) & 7))
                  & *(_BYTE *)(v110 + -(((unsigned int)(v56 - v110) >> 3) + 10))) == 0 )
                goto LABEL_90;
            }
          }
          sub_603440(v46, (__int64 *)&v122, a3 + 28);
LABEL_11:
          v13 = v120;
        }
        else
        {
LABEL_92:
          v120 = 0;
        }
      }
      v8 = (__int64)v122;
      v12 = 32 * v123;
      goto LABEL_8;
    }
  }
  v120 = 0;
  v12 = 0;
  v13 = 0;
  if ( (v10 & 0x20) == 0 )
  {
    v15 = (_QWORD *)(a1 + 96);
    v16 = (FILE *)(a3 + 28);
    v17 = 3375;
    goto LABEL_10;
  }
LABEL_8:
  sub_823A00(v8, v12);
  return v13;
}
