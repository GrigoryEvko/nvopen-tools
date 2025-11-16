// Function: sub_1950FB0
// Address: 0x1950fb0
//
__int64 __fastcall sub_1950FB0(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        __m128 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v11; // r14
  __int64 *v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  _BYTE *v24; // rsi
  __int64 *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rcx
  char v31; // si
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  char v37; // cl
  __int64 v38; // r12
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // r8
  unsigned __int8 v42; // al
  __int64 v43; // rdx
  __int64 v44; // r12
  _QWORD *v45; // rcx
  _QWORD *v46; // rax
  _QWORD *v47; // r13
  __int64 v48; // rax
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 j; // r13
  _QWORD *v52; // rsi
  _QWORD *v53; // rax
  _QWORD *v54; // rdi
  unsigned int v55; // r8d
  _QWORD *v56; // rcx
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rdi
  int v59; // eax
  unsigned int v60; // esi
  __int64 v61; // rdi
  __int64 v62; // r14
  __int64 *v63; // rax
  char v64; // dl
  __int64 v65; // rbx
  __int64 v66; // rax
  char v67; // si
  char v68; // cl
  void *v69; // rdi
  unsigned int v70; // eax
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 *v74; // rcx
  __int64 *i; // rsi
  _QWORD *v76; // rdx
  __int64 v80; // [rsp+70h] [rbp-460h]
  __int64 v82; // [rsp+90h] [rbp-440h]
  __int64 v83; // [rsp+98h] [rbp-438h]
  unsigned __int8 v84; // [rsp+A7h] [rbp-429h]
  __int64 v85; // [rsp+A8h] [rbp-428h]
  __int64 v86; // [rsp+A8h] [rbp-428h]
  __int64 v87; // [rsp+B0h] [rbp-420h]
  __int64 v88; // [rsp+B8h] [rbp-418h]
  __int64 v89; // [rsp+C0h] [rbp-410h] BYREF
  _BYTE *v90; // [rsp+C8h] [rbp-408h]
  _BYTE *v91; // [rsp+D0h] [rbp-400h]
  __int64 v92; // [rsp+D8h] [rbp-3F8h]
  int v93; // [rsp+E0h] [rbp-3F0h]
  _BYTE v94[72]; // [rsp+E8h] [rbp-3E8h] BYREF
  __int64 v95; // [rsp+130h] [rbp-3A0h] BYREF
  _BYTE *v96; // [rsp+138h] [rbp-398h]
  _BYTE *v97; // [rsp+140h] [rbp-390h]
  __int64 v98; // [rsp+148h] [rbp-388h]
  int v99; // [rsp+150h] [rbp-380h]
  _BYTE v100[72]; // [rsp+158h] [rbp-378h] BYREF
  _QWORD v101[16]; // [rsp+1A0h] [rbp-330h] BYREF
  __int64 v102; // [rsp+220h] [rbp-2B0h] BYREF
  _QWORD *v103; // [rsp+228h] [rbp-2A8h]
  _QWORD *v104; // [rsp+230h] [rbp-2A0h]
  __int64 v105; // [rsp+238h] [rbp-298h]
  int v106; // [rsp+240h] [rbp-290h]
  _QWORD v107[8]; // [rsp+248h] [rbp-288h] BYREF
  __int64 v108; // [rsp+288h] [rbp-248h] BYREF
  __int64 v109; // [rsp+290h] [rbp-240h]
  unsigned __int64 v110; // [rsp+298h] [rbp-238h]
  __int64 v111; // [rsp+2A0h] [rbp-230h] BYREF
  __int64 *v112; // [rsp+2A8h] [rbp-228h]
  __int64 *v113; // [rsp+2B0h] [rbp-220h]
  unsigned int v114; // [rsp+2B8h] [rbp-218h]
  unsigned int v115; // [rsp+2BCh] [rbp-214h]
  int v116; // [rsp+2C0h] [rbp-210h]
  _BYTE v117[64]; // [rsp+2C8h] [rbp-208h] BYREF
  __int64 v118; // [rsp+308h] [rbp-1C8h] BYREF
  __int64 v119; // [rsp+310h] [rbp-1C0h]
  unsigned __int64 v120; // [rsp+318h] [rbp-1B8h]
  __int64 v121; // [rsp+320h] [rbp-1B0h] BYREF
  __int64 v122; // [rsp+328h] [rbp-1A8h]
  unsigned __int64 v123; // [rsp+330h] [rbp-1A0h]
  _BYTE v124[64]; // [rsp+348h] [rbp-188h] BYREF
  __int64 v125; // [rsp+388h] [rbp-148h]
  __int64 v126; // [rsp+390h] [rbp-140h]
  unsigned __int64 v127; // [rsp+398h] [rbp-138h]
  _QWORD v128[2]; // [rsp+3A0h] [rbp-130h] BYREF
  unsigned __int64 v129; // [rsp+3B0h] [rbp-120h]
  char v130; // [rsp+3B8h] [rbp-118h]
  char v131[64]; // [rsp+3C8h] [rbp-108h] BYREF
  __int64 v132; // [rsp+408h] [rbp-C8h]
  __int64 v133; // [rsp+410h] [rbp-C0h]
  unsigned __int64 v134; // [rsp+418h] [rbp-B8h]
  _QWORD v135[2]; // [rsp+420h] [rbp-B0h] BYREF
  unsigned __int64 v136; // [rsp+430h] [rbp-A0h]
  char v137[64]; // [rsp+448h] [rbp-88h] BYREF
  __int64 v138; // [rsp+488h] [rbp-48h]
  __int64 v139; // [rsp+490h] [rbp-40h]
  unsigned __int64 v140; // [rsp+498h] [rbp-38h]

  v11 = &v95;
  v12 = &v89;
  v90 = v94;
  v91 = v94;
  v96 = v100;
  v97 = v100;
  v89 = 0;
  v92 = 8;
  v93 = 0;
  v95 = 0;
  v98 = 8;
  v99 = 0;
  v84 = 0;
  while ( 1 )
  {
    v13 = *(_QWORD *)(a1 + 80);
    v108 = 0;
    v109 = 0;
    v110 = 0;
    if ( v13 )
      v13 -= 24;
    v130 = 0;
    memset(v101, 0, sizeof(v101));
    LODWORD(v101[3]) = 8;
    v107[0] = v13;
    v101[1] = &v101[5];
    v101[2] = &v101[5];
    v128[0] = v13;
    v103 = v107;
    v104 = v107;
    v105 = 0x100000008LL;
    v106 = 0;
    v102 = 1;
    sub_144A690(&v108, (__int64)v128);
    sub_16CCEE0(&v121, (__int64)v124, 8, (__int64)v101);
    v14 = v101[13];
    memset(&v101[13], 0, 24);
    v125 = v14;
    v126 = v101[14];
    v127 = v101[15];
    sub_16CCEE0(&v111, (__int64)v117, 8, (__int64)&v102);
    v15 = v108;
    v108 = 0;
    v118 = v15;
    v16 = v109;
    v109 = 0;
    v119 = v16;
    v17 = v110;
    v110 = 0;
    v120 = v17;
    sub_16CCEE0(v128, (__int64)v131, 8, (__int64)&v111);
    v18 = v118;
    v118 = 0;
    v132 = v18;
    v19 = v119;
    v119 = 0;
    v133 = v19;
    v20 = v120;
    v120 = 0;
    v134 = v20;
    sub_16CCEE0(v135, (__int64)v137, 8, (__int64)&v121);
    v21 = v125;
    v125 = 0;
    v138 = v21;
    v22 = v126;
    v126 = 0;
    v139 = v22;
    v23 = v127;
    v127 = 0;
    v140 = v23;
    if ( v118 )
      j_j___libc_free_0(v118, v120 - v118);
    if ( v113 != v112 )
      _libc_free((unsigned __int64)v113);
    if ( v125 )
      j_j___libc_free_0(v125, v127 - v125);
    if ( v123 != v122 )
      _libc_free(v123);
    if ( v108 )
      j_j___libc_free_0(v108, v110 - v108);
    if ( v104 != v103 )
      _libc_free((unsigned __int64)v104);
    if ( v101[13] )
      j_j___libc_free_0(v101[13], v101[15] - v101[13]);
    if ( v101[2] != v101[1] )
      _libc_free(v101[2]);
    v24 = v117;
    v25 = &v111;
    sub_16CCCB0(&v111, (__int64)v117, (__int64)v128);
    v26 = v133;
    v27 = v132;
    v118 = 0;
    v119 = 0;
    v120 = 0;
    v28 = v133 - v132;
    if ( v133 == v132 )
    {
      v28 = 0;
      v29 = 0;
    }
    else
    {
      if ( v28 > 0x7FFFFFFFFFFFFFE0LL )
        goto LABEL_152;
      v29 = sub_22077B0(v133 - v132);
      v26 = v133;
      v27 = v132;
    }
    v118 = v29;
    v119 = v29;
    v120 = v29 + v28;
    if ( v27 == v26 )
    {
      v30 = v29;
    }
    else
    {
      v30 = v29 + v26 - v27;
      do
      {
        if ( v29 )
        {
          *(_QWORD *)v29 = *(_QWORD *)v27;
          v31 = *(_BYTE *)(v27 + 24);
          *(_BYTE *)(v29 + 24) = v31;
          if ( v31 )
          {
            a4 = (__m128)_mm_loadu_si128((const __m128i *)(v27 + 8));
            *(__m128 *)(v29 + 8) = a4;
          }
        }
        v29 += 32;
        v27 += 32;
      }
      while ( v29 != v30 );
    }
    v24 = v124;
    v119 = v30;
    v25 = &v121;
    sub_16CCCB0(&v121, (__int64)v124, (__int64)v135);
    v32 = v139;
    v27 = v138;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v33 = v139 - v138;
    if ( v139 == v138 )
    {
      v35 = 0;
    }
    else
    {
      if ( v33 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_152:
        sub_4261EA(v25, v24, v27);
      v34 = sub_22077B0(v139 - v138);
      v27 = v138;
      v35 = v34;
      v32 = v139;
    }
    v125 = v35;
    v126 = v35;
    v127 = v35 + v33;
    if ( v27 == v32 )
    {
      v36 = v35;
    }
    else
    {
      v36 = v35 + v32 - v27;
      do
      {
        if ( v35 )
        {
          *(_QWORD *)v35 = *(_QWORD *)v27;
          v37 = *(_BYTE *)(v27 + 24);
          *(_BYTE *)(v35 + 24) = v37;
          if ( v37 )
          {
            a5 = (__m128)_mm_loadu_si128((const __m128i *)(v27 + 8));
            *(__m128 *)(v35 + 8) = a5;
          }
        }
        v35 += 32;
        v27 += 32;
      }
      while ( v35 != v36 );
      v35 = v125;
    }
    v38 = v119;
    v39 = v118;
    v126 = v36;
    v80 = (__int64)v11;
    if ( v119 - v118 != v36 - v35 )
    {
LABEL_40:
      while ( 1 )
      {
        v40 = *(_QWORD *)(v38 - 32);
        v41 = *(_QWORD *)(v40 + 48);
        v87 = v40 + 40;
        if ( v41 != v40 + 40 )
          break;
        while ( 1 )
        {
LABEL_78:
          if ( !*(_BYTE *)(v38 - 8) )
          {
            v57 = sub_157EBA0(v40);
            *(_BYTE *)(v38 - 8) = 1;
            *(_QWORD *)(v38 - 24) = v57;
            *(_DWORD *)(v38 - 16) = 0;
          }
          while ( 1 )
          {
LABEL_80:
            v58 = sub_157EBA0(v40);
            v59 = 0;
            if ( v58 )
              v59 = sub_15F4D60(v58);
            v60 = *(_DWORD *)(v38 - 16);
            if ( v60 == v59 )
              break;
            v61 = *(_QWORD *)(v38 - 24);
            *(_DWORD *)(v38 - 16) = v60 + 1;
            v62 = sub_15F4DF0(v61, v60);
            v63 = v112;
            if ( v113 == v112 )
            {
              v74 = &v112[v115];
              if ( v112 != v74 )
              {
                for ( i = 0; ; i = v63++ )
                {
                  while ( 1 )
                  {
                    if ( v62 == *v63 )
                      goto LABEL_80;
                    if ( *v63 == -2 )
                      break;
                    if ( v74 == ++v63 )
                    {
                      if ( !i )
                        goto LABEL_132;
                      v65 = v62;
                      goto LABEL_129;
                    }
                  }
                  if ( v74 == v63 + 1 )
                  {
                    v65 = v62;
                    i = v63;
LABEL_129:
                    *i = v65;
                    --v116;
                    ++v111;
                    goto LABEL_86;
                  }
                }
              }
LABEL_132:
              if ( v115 < v114 )
              {
                v65 = v62;
                ++v115;
                *v74 = v62;
                ++v111;
                goto LABEL_86;
              }
            }
            sub_16CCBA0((__int64)&v111, v62);
            if ( v64 )
            {
              v65 = v62;
LABEL_86:
              v102 = v65;
              LOBYTE(v105) = 0;
              sub_144A690(&v118, (__int64)&v102);
              v39 = v118;
              v38 = v119;
              goto LABEL_87;
            }
          }
          v119 -= 32;
          v39 = v118;
          v38 = v119;
          if ( v119 == v118 )
            break;
          v40 = *(_QWORD *)(v119 - 32);
        }
LABEL_87:
        v35 = v125;
        if ( v38 - v39 == v126 - v125 )
          goto LABEL_88;
      }
      v82 = *(_QWORD *)(v38 - 32);
      while ( 1 )
      {
        v43 = *((unsigned int *)v12 + 7);
        v44 = v41 - 24;
        v88 = *(_QWORD *)(v41 + 8);
        if ( (_DWORD)v43 == *((_DWORD *)v12 + 8) )
          goto LABEL_51;
        v45 = (_QWORD *)v12[2];
        v46 = (_QWORD *)v12[1];
        if ( v45 == v46 )
        {
          v47 = &v46[v43];
          if ( v46 == v47 )
          {
            v76 = (_QWORD *)v12[1];
          }
          else
          {
            do
            {
              if ( v44 == *v46 )
                break;
              ++v46;
            }
            while ( v47 != v46 );
            v76 = v47;
          }
LABEL_74:
          while ( v76 != v46 )
          {
            if ( *v46 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_50;
            ++v46;
          }
          if ( v46 == v47 )
            goto LABEL_76;
LABEL_51:
          v86 = v41;
          if ( !*(_QWORD *)(v41 - 16) )
            goto LABEL_42;
          v83 = sub_13E3350(v44, a2, a3, 1, v41);
          if ( !v83 )
            goto LABEL_42;
          for ( j = *(_QWORD *)(v86 - 16); j; j = *(_QWORD *)(j + 8) )
          {
LABEL_57:
            v52 = sub_1648700(j);
            v53 = *(_QWORD **)(v80 + 8);
            if ( *(_QWORD **)(v80 + 16) != v53 )
              goto LABEL_55;
            v54 = &v53[*(unsigned int *)(v80 + 28)];
            v55 = *(_DWORD *)(v80 + 28);
            if ( v53 != v54 )
            {
              v56 = 0;
              while ( v52 != (_QWORD *)*v53 )
              {
                if ( *v53 == -2 )
                  v56 = v53;
                if ( v54 == ++v53 )
                {
                  if ( !v56 )
                    goto LABEL_137;
                  *v56 = v52;
                  --*(_DWORD *)(v80 + 32);
                  ++*(_QWORD *)v80;
                  j = *(_QWORD *)(j + 8);
                  if ( j )
                    goto LABEL_57;
                  goto LABEL_66;
                }
              }
              continue;
            }
LABEL_137:
            if ( v55 < *(_DWORD *)(v80 + 24) )
            {
              *(_DWORD *)(v80 + 28) = v55 + 1;
              *v54 = v52;
              ++*(_QWORD *)v80;
            }
            else
            {
LABEL_55:
              sub_16CCBA0(v80, (__int64)v52);
            }
          }
LABEL_66:
          sub_164D160(v44, v83, a4, *(double *)a5.m128_u64, a6, a7, v49, v50, a10, a11);
          v84 = 1;
LABEL_42:
          v42 = sub_1AEB370(v44, a2->m128i_i64[1]);
          v41 = v88;
          if ( v42 )
          {
            v84 = v42;
            v41 = *(_QWORD *)(v82 + 48);
          }
          if ( v87 == v41 )
            goto LABEL_77;
        }
        else
        {
          v85 = v41;
          v47 = &v45[*((unsigned int *)v12 + 6)];
          v46 = sub_16CC9F0((__int64)v12, v41 - 24);
          v41 = v85;
          if ( v44 == *v46 )
          {
            v72 = v12[2];
            if ( v72 == v12[1] )
              v73 = *((unsigned int *)v12 + 7);
            else
              v73 = *((unsigned int *)v12 + 6);
            v76 = (_QWORD *)(v72 + 8 * v73);
            goto LABEL_74;
          }
          v48 = v12[2];
          if ( v48 == v12[1] )
          {
            v46 = (_QWORD *)(v48 + 8LL * *((unsigned int *)v12 + 7));
            v76 = v46;
            goto LABEL_74;
          }
          v46 = (_QWORD *)(v48 + 8LL * *((unsigned int *)v12 + 6));
LABEL_50:
          if ( v46 != v47 )
            goto LABEL_51;
LABEL_76:
          v41 = v88;
          if ( v87 == v88 )
          {
LABEL_77:
            v38 = v119;
            v40 = *(_QWORD *)(v119 - 32);
            goto LABEL_78;
          }
        }
      }
    }
LABEL_88:
    if ( v38 != v39 )
    {
      v66 = v35;
      while ( *(_QWORD *)v39 == *(_QWORD *)v66 )
      {
        v67 = *(_BYTE *)(v39 + 24);
        v68 = *(_BYTE *)(v66 + 24);
        if ( v67 && v68 )
        {
          if ( *(_DWORD *)(v39 + 16) != *(_DWORD *)(v66 + 16) )
            goto LABEL_40;
        }
        else if ( v67 != v68 )
        {
          goto LABEL_40;
        }
        v39 += 32;
        v66 += 32;
        if ( v38 == v39 )
          goto LABEL_95;
      }
      goto LABEL_40;
    }
LABEL_95:
    if ( v35 )
      j_j___libc_free_0(v35, v127 - v35);
    if ( v123 != v122 )
      _libc_free(v123);
    if ( v118 )
      j_j___libc_free_0(v118, v120 - v118);
    if ( v113 != v112 )
      _libc_free((unsigned __int64)v113);
    if ( v138 )
      j_j___libc_free_0(v138, v140 - v138);
    if ( v136 != v135[1] )
      _libc_free(v136);
    if ( v132 )
      j_j___libc_free_0(v132, v134 - v132);
    if ( v129 != v128[1] )
      _libc_free(v129);
    ++*v12;
    v69 = (void *)v12[2];
    if ( v69 == (void *)v12[1] )
      goto LABEL_116;
    v70 = 4 * (*((_DWORD *)v12 + 7) - *((_DWORD *)v12 + 8));
    v71 = *((unsigned int *)v12 + 6);
    if ( v70 < 0x20 )
      v70 = 32;
    if ( (unsigned int)v71 <= v70 )
    {
      memset(v69, -1, 8 * v71);
LABEL_116:
      *(__int64 *)((char *)v12 + 28) = 0;
      goto LABEL_117;
    }
    sub_16CC920((__int64)v12);
LABEL_117:
    v11 = v12;
    if ( *(_DWORD *)(v80 + 28) == *(_DWORD *)(v80 + 32) )
      break;
    v12 = (__int64 *)v80;
  }
  if ( v97 != v96 )
    _libc_free((unsigned __int64)v97);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  return v84;
}
