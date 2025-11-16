// Function: sub_2EE36B0
// Address: 0x2ee36b0
//
__int64 __fastcall sub_2EE36B0(__int64 a1, char *a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // r8
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r11d
  unsigned int v23; // r8d
  unsigned int j; // eax
  __int64 v25; // r9
  unsigned int v26; // eax
  __int64 v27; // rdi
  __int64 v28; // rbx
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 v31; // rax
  int v32; // r14d
  unsigned int k; // eax
  __int64 v34; // r9
  unsigned int v35; // eax
  int v36; // r14d
  unsigned int m; // eax
  __int64 v38; // r9
  unsigned int v39; // eax
  int v40; // r11d
  unsigned int n; // eax
  __int64 v42; // rsi
  unsigned int v43; // eax
  char v44; // dl
  __m128i v45; // xmm4
  __m128i v46; // xmm3
  __m128i v47; // xmm2
  __m128i v48; // xmm1
  __m128i v49; // xmm0
  char *v50; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // [rsp+0h] [rbp-5C0h]
  char v62; // [rsp+8h] [rbp-5B8h]
  __int64 v63; // [rsp+10h] [rbp-5B0h]
  __int64 v64; // [rsp+18h] [rbp-5A8h]
  __int64 v65; // [rsp+20h] [rbp-5A0h]
  __int64 v66; // [rsp+28h] [rbp-598h]
  __int64 v67; // [rsp+30h] [rbp-590h]
  __int64 v68; // [rsp+38h] [rbp-588h]
  char v69[8]; // [rsp+40h] [rbp-580h] BYREF
  unsigned __int64 v70; // [rsp+48h] [rbp-578h]
  char v71; // [rsp+5Ch] [rbp-564h]
  char v72[16]; // [rsp+60h] [rbp-560h] BYREF
  char v73[8]; // [rsp+70h] [rbp-550h] BYREF
  unsigned __int64 v74; // [rsp+78h] [rbp-548h]
  char v75; // [rsp+8Ch] [rbp-534h]
  char v76[16]; // [rsp+90h] [rbp-530h] BYREF
  __int64 v77; // [rsp+A0h] [rbp-520h] BYREF
  __int64 v78; // [rsp+A8h] [rbp-518h]
  __int64 v79; // [rsp+B0h] [rbp-510h] BYREF
  __int64 v80; // [rsp+B8h] [rbp-508h]
  __int64 v81; // [rsp+C0h] [rbp-500h]
  __int64 v82; // [rsp+C8h] [rbp-4F8h]
  __int64 v83; // [rsp+D0h] [rbp-4F0h]
  __int64 v84; // [rsp+D8h] [rbp-4E8h]
  __int64 v85; // [rsp+E0h] [rbp-4E0h]
  __int64 v86; // [rsp+E8h] [rbp-4D8h]
  __int64 v87; // [rsp+F0h] [rbp-4D0h]
  char v88[56]; // [rsp+F8h] [rbp-4C8h] BYREF
  char v89; // [rsp+130h] [rbp-490h] BYREF
  __m128i v90; // [rsp+238h] [rbp-388h]
  __m128i v91; // [rsp+248h] [rbp-378h]
  __m128i v92; // [rsp+258h] [rbp-368h]
  __m128i v93; // [rsp+268h] [rbp-358h]
  __m128i v94; // [rsp+278h] [rbp-348h]
  __m128i v95; // [rsp+288h] [rbp-338h]
  __m128i v96; // [rsp+298h] [rbp-328h]
  __m128i v97; // [rsp+2A8h] [rbp-318h]
  __m128i v98; // [rsp+2B8h] [rbp-308h]
  __m128i v99; // [rsp+2C8h] [rbp-2F8h]
  __int64 v100; // [rsp+2D8h] [rbp-2E8h]
  __int64 v101; // [rsp+2E0h] [rbp-2E0h]
  __int64 v102; // [rsp+2E8h] [rbp-2D8h]
  __int64 v103; // [rsp+2F0h] [rbp-2D0h]
  __int64 v104; // [rsp+2F8h] [rbp-2C8h]
  __int64 v105; // [rsp+300h] [rbp-2C0h]
  char *v106; // [rsp+308h] [rbp-2B8h]
  __int64 v107; // [rsp+310h] [rbp-2B0h]
  char v108; // [rsp+318h] [rbp-2A8h] BYREF
  __int64 v109; // [rsp+358h] [rbp-268h]
  __int64 v110; // [rsp+360h] [rbp-260h]
  __int64 v111; // [rsp+368h] [rbp-258h]
  __int64 v112; // [rsp+370h] [rbp-250h]
  __int64 v113; // [rsp+378h] [rbp-248h]
  char *v114; // [rsp+380h] [rbp-240h]
  __int64 v115; // [rsp+388h] [rbp-238h]
  char v116; // [rsp+390h] [rbp-230h] BYREF
  int v117; // [rsp+418h] [rbp-1A8h] BYREF
  __int64 v118; // [rsp+420h] [rbp-1A0h]
  int *v119; // [rsp+428h] [rbp-198h]
  int *v120; // [rsp+430h] [rbp-190h]
  __int64 v121; // [rsp+438h] [rbp-188h]
  __int64 v122; // [rsp+440h] [rbp-180h]
  __int64 v123; // [rsp+448h] [rbp-178h]
  __int64 v124; // [rsp+450h] [rbp-170h]
  int v125; // [rsp+458h] [rbp-168h]
  __int64 v126; // [rsp+460h] [rbp-160h]
  __int64 v127; // [rsp+468h] [rbp-158h]
  __int64 v128; // [rsp+470h] [rbp-150h]
  __int64 v129; // [rsp+478h] [rbp-148h]
  _QWORD *v130; // [rsp+480h] [rbp-140h]
  __int64 v131; // [rsp+488h] [rbp-138h]
  _QWORD v132[6]; // [rsp+490h] [rbp-130h] BYREF
  char v133; // [rsp+4C0h] [rbp-100h] BYREF
  _QWORD v134[7]; // [rsp+500h] [rbp-C0h] BYREF
  int v135; // [rsp+538h] [rbp-88h]
  __int64 v136; // [rsp+540h] [rbp-80h]
  __int64 v137; // [rsp+548h] [rbp-78h]
  __int64 v138; // [rsp+550h] [rbp-70h]
  int v139; // [rsp+558h] [rbp-68h]
  __int64 v140; // [rsp+560h] [rbp-60h]
  __int64 v141; // [rsp+568h] [rbp-58h]
  __int64 v142; // [rsp+570h] [rbp-50h]
  int v143; // [rsp+578h] [rbp-48h]
  char v144; // [rsp+580h] [rbp-40h]

  v64 = sub_2EB2140(a4, qword_501FE48, (__int64)a3) + 8;
  v65 = sub_2EB2140(a4, qword_50209E0, (__int64)a3) + 8;
  v66 = sub_2EB2140(a4, &qword_501FE30, (__int64)a3) + 8;
  v7 = *(_QWORD *)(sub_2EB2140(a4, &qword_50209B8, (__int64)a3) + 8);
  v8 = *(_QWORD *)(*a3 + 40);
  v9 = *(unsigned int *)(v7 + 88);
  v10 = *(_QWORD *)(v7 + 72);
  if ( !(_DWORD)v9 )
    goto LABEL_66;
  v11 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; i = (v9 - 1) & v14 )
  {
    v13 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F87C68 && v8 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_66;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v10 + 24 * v9 )
  {
LABEL_66:
    v68 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    v68 = v15;
    if ( v15 )
    {
      v78 = 1;
      v68 = v15 + 8;
      v16 = &v79;
      do
      {
        *v16 = -4096;
        v16 += 2;
      }
      while ( v16 != (__int64 *)&v89 );
      if ( (v78 & 1) == 0 )
        sub_C7D6A0(v79, 16LL * (unsigned int)v80, 8);
    }
  }
  v67 = 0;
  if ( (_BYTE)qword_5022228 )
    v67 = sub_2EB2140(a4, (__int64 *)&unk_501EC10, (__int64)a3) + 8;
  v63 = sub_2EB2140(a4, &qword_501F1C0, (__int64)a3) + 8;
  v17 = sub_2EB2140(a4, &qword_50209D0, (__int64)a3);
  v18 = sub_BC1CD0(*(_QWORD *)(v17 + 8), &unk_4F86540, *a3);
  v19 = *(_DWORD *)(a4 + 88);
  v20 = *(_QWORD *)(a4 + 72);
  v21 = v18 + 8;
  if ( v19 )
  {
    v22 = 1;
    v23 = v19 - 1;
    for ( j = (v19 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_501EAD0 >> 9) ^ ((unsigned int)&unk_501EAD0 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v23 & v26 )
    {
      v25 = v20 + 24LL * j;
      if ( *(_UNKNOWN **)v25 == &unk_501EAD0 && a3 == *(__int64 **)(v25 + 8) )
        break;
      if ( *(_QWORD *)v25 == -4096 && *(_QWORD *)(v25 + 8) == -4096 )
        goto LABEL_21;
      v26 = v22 + j;
      ++v22;
    }
    v27 = v20 + 24LL * v19;
    if ( v27 != v25 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)(v25 + 16) + 24LL);
      if ( v28 )
        v28 += 8;
      goto LABEL_28;
    }
  }
  else
  {
LABEL_21:
    v27 = v20 + 24LL * v19;
    if ( !v19 )
    {
      v28 = 0;
      v29 = 0;
      v30 = 0;
LABEL_23:
      v31 = 0;
      goto LABEL_52;
    }
    v23 = v19 - 1;
  }
  v28 = 0;
LABEL_28:
  v32 = 1;
  for ( k = v23
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_5025C20 >> 9) ^ ((unsigned int)&unk_5025C20 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; k = v23 & v35 )
  {
    v34 = v20 + 24LL * k;
    if ( *(_UNKNOWN **)v34 == &unk_5025C20 && a3 == *(__int64 **)(v34 + 8) )
      break;
    if ( *(_QWORD *)v34 == -4096 && *(_QWORD *)(v34 + 8) == -4096 )
      goto LABEL_64;
    v35 = v32 + k;
    ++v32;
  }
  if ( v34 == v27 )
  {
LABEL_64:
    v29 = 0;
    goto LABEL_36;
  }
  v29 = *(_QWORD *)(*(_QWORD *)(v34 + 16) + 24LL);
  if ( v29 )
    v29 += 8;
LABEL_36:
  v36 = 1;
  for ( m = v23
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_501EB18 >> 9) ^ ((unsigned int)&unk_501EB18 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; m = v23 & v39 )
  {
    v38 = v20 + 24LL * m;
    if ( *(_UNKNOWN **)v38 == &unk_501EB18 && a3 == *(__int64 **)(v38 + 8) )
      break;
    if ( *(_QWORD *)v38 == -4096 && *(_QWORD *)(v38 + 8) == -4096 )
      goto LABEL_62;
    v39 = v36 + m;
    ++v36;
  }
  if ( v38 == v27 )
  {
LABEL_62:
    v30 = 0;
    goto LABEL_44;
  }
  v30 = *(_QWORD *)(*(_QWORD *)(v38 + 16) + 24LL);
  if ( v30 )
    v30 += 8;
LABEL_44:
  v40 = 1;
  for ( n = v23
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&qword_50208B0 >> 9) ^ ((unsigned int)&qword_50208B0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; n = v23 & v43 )
  {
    v42 = v20 + 24LL * n;
    if ( *(__int64 **)v42 == &qword_50208B0 && a3 == *(__int64 **)(v42 + 8) )
      break;
    if ( *(_QWORD *)v42 == -4096 && *(_QWORD *)(v42 + 8) == -4096 )
      goto LABEL_23;
    v43 = v40 + n;
    ++v40;
  }
  if ( v42 == v27 )
    goto LABEL_23;
  v31 = *(_QWORD *)(*(_QWORD *)(v42 + 16) + 24LL);
  if ( v31 )
    v31 += 8;
LABEL_52:
  v60 = v31;
  v87 = v21;
  v77 = 0;
  v44 = *a2;
  v78 = 0;
  v62 = v44;
  v81 = v64;
  v79 = 0;
  v82 = v65;
  v80 = 0;
  v83 = v66;
  v84 = v68;
  v85 = v67;
  v86 = v63;
  sub_2F5FEE0(v88);
  v112 = v30;
  v107 = 0x1000000000LL;
  v45 = _mm_loadu_si128(xmmword_3F8F0C0);
  v46 = _mm_loadu_si128(&xmmword_3F8F0C0[1]);
  v100 = 0;
  v47 = _mm_loadu_si128(&xmmword_3F8F0C0[2]);
  v48 = _mm_loadu_si128(&xmmword_3F8F0C0[3]);
  v113 = v60;
  v49 = _mm_loadu_si128(&xmmword_3F8F0C0[4]);
  v114 = &v116;
  v106 = &v108;
  v115 = 0x800000000LL;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v109 = 0;
  v110 = v28;
  v111 = v29;
  v90 = v45;
  v91 = v46;
  v92 = v47;
  v93 = v48;
  v94 = v49;
  v95 = v45;
  v96 = v46;
  v97 = v47;
  v98 = v48;
  v99 = v49;
  v117 = 0;
  v118 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v131 = 0;
  memset(v132, 0, 40);
  v132[5] = 1;
  v119 = &v117;
  v120 = &v117;
  v130 = v132;
  v50 = &v133;
  do
  {
    *(_DWORD *)v50 = -1;
    v50 += 16;
  }
  while ( v50 != (char *)v134 );
  memset(v134, 0, sizeof(v134));
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = v62;
  if ( (unsigned __int8)sub_2EE0CF0(&v77, (__int64)a3) )
  {
    sub_2EAFFB0((__int64)v69);
    sub_2ED2320((__int64)v69, (__int64)&qword_501FE30, v52, v53, v54, v55);
    sub_2ED2320((__int64)v69, (__int64)&qword_50208B0, v56, v57, v58, v59);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v72, (__int64)v69);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v76, (__int64)v73);
    if ( !v75 )
      _libc_free(v74);
    if ( !v71 )
      _libc_free(v70);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  sub_2ED5940((__int64)&v77);
  return a1;
}
