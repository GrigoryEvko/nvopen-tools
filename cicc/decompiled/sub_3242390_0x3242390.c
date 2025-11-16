// Function: sub_3242390
// Address: 0x3242390
//
char __fastcall sub_3242390(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4)
{
  char result; // al
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // r9
  __int16 *v9; // rax
  __int16 *v10; // r15
  unsigned int v11; // ebx
  unsigned int i; // r14d
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rax
  __int16 *v16; // rax
  __int16 *v17; // rcx
  int v18; // edx
  __int16 *v19; // r15
  unsigned int j; // r14d
  int v21; // r13d
  unsigned int v22; // ebx
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // r8
  unsigned int v25; // r13d
  _QWORD *v26; // rax
  unsigned __int64 *v27; // r14
  unsigned __int64 v28; // rsi
  char v29; // r11
  char v30; // di
  unsigned int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rax
  unsigned __int64 v35; // r8
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // rsi
  const __m128i *v39; // r14
  __m128i *v40; // rax
  int v41; // eax
  unsigned __int64 v42; // r12
  const __m128i *v43; // rdx
  __int64 v44; // rax
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rcx
  unsigned __int64 v47; // r8
  __m128i *v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // rsi
  unsigned __int64 v51; // rcx
  const __m128i *v52; // rbx
  unsigned __int64 v53; // r8
  __m128i *v54; // rdx
  __int64 v55; // rdx
  __int64 *v56; // rdi
  __int64 v57; // r8
  unsigned int v58; // ecx
  unsigned int m; // eax
  unsigned __int64 *v60; // rsi
  __int64 v61; // rdi
  unsigned __int64 v62; // r8
  unsigned int v63; // ecx
  unsigned int k; // edx
  __int64 v65; // r13
  int v66; // r14d
  __int16 v67; // bx
  __int64 v68; // r8
  __int64 v69; // r9
  __int16 v70; // r12
  __int64 v71; // rax
  unsigned __int64 v72; // rdx
  __int64 v73; // rcx
  const __m128i *v74; // rdx
  __m128i *v75; // rax
  const __m128i *v76; // rcx
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rsi
  __int64 v79; // rdx
  unsigned __int64 v80; // r8
  __m128i *v81; // rdx
  unsigned __int64 v82; // rax
  unsigned int v83; // ecx
  __int64 v84; // rax
  unsigned __int64 v85; // rax
  unsigned int v86; // eax
  unsigned int v87; // ecx
  __int64 v88; // rdx
  unsigned __int64 v89; // rdx
  const __m128i *v90; // r14
  unsigned __int64 v91; // rsi
  unsigned __int64 v92; // rcx
  __m128i *v93; // rax
  const __m128i *v94; // rdx
  unsigned __int64 v95; // rsi
  unsigned __int64 v96; // rcx
  __int64 v97; // rax
  unsigned __int64 v98; // r8
  __m128i *v99; // rax
  __int64 v100; // rdi
  const void *v101; // rsi
  char *v102; // rbx
  __int64 v103; // rdi
  const void *v104; // rsi
  __int64 v105; // r9
  char *v106; // r14
  unsigned __int64 v107; // r13
  __int64 v108; // rdi
  const void *v109; // rsi
  __int64 v110; // rdi
  const void *v111; // r9
  char *v112; // rbx
  char *v113; // rbx
  const void *v114; // rsi
  char *v115; // r14
  __int64 v116; // rdi
  const void *v117; // rsi
  char *v118; // rbx
  unsigned int v119; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v120; // [rsp+8h] [rbp-A8h]
  unsigned int v121; // [rsp+28h] [rbp-88h]
  __int64 v122; // [rsp+30h] [rbp-80h]
  int v123; // [rsp+38h] [rbp-78h]
  unsigned int v125; // [rsp+40h] [rbp-70h]
  char v127; // [rsp+44h] [rbp-6Ch]
  char v128; // [rsp+44h] [rbp-6Ch]
  char v129; // [rsp+44h] [rbp-6Ch]
  char v131; // [rsp+48h] [rbp-68h]
  char v132; // [rsp+48h] [rbp-68h]
  unsigned __int64 v133; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v134; // [rsp+58h] [rbp-58h] BYREF
  __int64 v135; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v136; // [rsp+68h] [rbp-48h]
  const char *v137; // [rsp+70h] [rbp-40h]

  if ( a3 - 1 > 0x3FFFFFFE )
  {
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 80LL))(a1);
    if ( result )
    {
      v135 = -1;
      v136 = 0;
      v49 = *(unsigned int *)(a1 + 32);
      v50 = *(unsigned int *)(a1 + 36);
      v137 = 0;
      v51 = *(_QWORD *)(a1 + 24);
      v52 = (const __m128i *)&v135;
      v53 = v49 + 1;
      if ( v49 + 1 > v50 )
      {
        v103 = a1 + 24;
        v104 = (const void *)(a1 + 40);
        if ( v51 > (unsigned __int64)&v135 )
        {
          v129 = result;
          sub_C8D5F0(v103, v104, v49 + 1, 0x18u, v53, v6);
          v51 = *(_QWORD *)(a1 + 24);
          v49 = *(unsigned int *)(a1 + 32);
          result = v129;
        }
        else
        {
          v127 = result;
          if ( (unsigned __int64)&v135 < v51 + 24 * v49 )
          {
            v113 = (char *)&v135 - v51;
            sub_C8D5F0(v103, v104, v53, 0x18u, v53, v6);
            v51 = *(_QWORD *)(a1 + 24);
            v49 = *(unsigned int *)(a1 + 32);
            result = v127;
            v52 = (const __m128i *)&v113[v51];
          }
          else
          {
            sub_C8D5F0(v103, v104, v53, 0x18u, v53, v6);
            v51 = *(_QWORD *)(a1 + 24);
            v49 = *(unsigned int *)(a1 + 32);
            result = v127;
          }
        }
      }
      v54 = (__m128i *)(v51 + 24 * v49);
      *v54 = _mm_loadu_si128(v52);
      v54[1].m128i_i64[0] = v52[1].m128i_i64[0];
      ++*(_DWORD *)(a1 + 32);
    }
    else
    {
      result = sub_32420F0(a1);
      if ( result )
      {
        v76 = (const __m128i *)&v135;
        v136 = 0;
        v137 = 0;
        v77 = *(unsigned int *)(a1 + 36);
        v78 = *(_QWORD *)(a1 + 24);
        v135 = (int)a3;
        v79 = *(unsigned int *)(a1 + 32);
        v80 = v79 + 1;
        if ( v79 + 1 > v77 )
        {
          v128 = result;
          v110 = a1 + 24;
          v111 = (const void *)(a1 + 40);
          if ( v78 > (unsigned __int64)&v135 || (unsigned __int64)&v135 >= v78 + 24 * v79 )
          {
            sub_C8D5F0(v110, (const void *)(a1 + 40), v80, 0x18u, v80, (__int64)v111);
            v76 = (const __m128i *)&v135;
            v78 = *(_QWORD *)(a1 + 24);
            v79 = *(unsigned int *)(a1 + 32);
            result = v128;
          }
          else
          {
            v112 = (char *)&v135 - v78;
            sub_C8D5F0(v110, v111, v80, 0x18u, v80, (__int64)v111);
            v78 = *(_QWORD *)(a1 + 24);
            v79 = *(unsigned int *)(a1 + 32);
            result = v128;
            v76 = (const __m128i *)&v112[v78];
          }
        }
        v81 = (__m128i *)(v78 + 24 * v79);
        *v81 = _mm_loadu_si128(v76);
        v81[1].m128i_i64[0] = v76[1].m128i_i64[0];
        ++*(_DWORD *)(a1 + 32);
      }
    }
    return result;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a2 + 16LL))(a2, a3, 0);
  if ( v7 >= 0 )
  {
    v135 = v7;
    v43 = (const __m128i *)&v135;
    v136 = 0;
    v44 = *(unsigned int *)(a1 + 32);
    v45 = *(unsigned int *)(a1 + 36);
    v137 = 0;
    v46 = *(_QWORD *)(a1 + 24);
    v47 = v44 + 1;
    if ( v44 + 1 > v45 )
    {
      v100 = a1 + 24;
      v101 = (const void *)(a1 + 40);
      if ( v46 > (unsigned __int64)&v135 || (unsigned __int64)&v135 >= v46 + 24 * v44 )
      {
        sub_C8D5F0(v100, v101, v47, 0x18u, v47, v8);
        v43 = (const __m128i *)&v135;
        v46 = *(_QWORD *)(a1 + 24);
        v44 = *(unsigned int *)(a1 + 32);
      }
      else
      {
        v102 = (char *)&v135 - v46;
        sub_C8D5F0(v100, v101, v47, 0x18u, v47, v8);
        v46 = *(_QWORD *)(a1 + 24);
        v44 = *(unsigned int *)(a1 + 32);
        v43 = (const __m128i *)&v102[v46];
      }
    }
    v48 = (__m128i *)(v46 + 24 * v44);
    *v48 = _mm_loadu_si128(v43);
    v48[1].m128i_i64[0] = v43[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 32);
    return 1;
  }
  v9 = (__int16 *)(a2[7] + 2LL * *(unsigned int *)(a2[1] + 24LL * a3 + 8));
  v10 = v9 + 1;
  LODWORD(v9) = *v9;
  v11 = a3 + (_DWORD)v9;
  if ( (_WORD)v9 )
  {
    for ( i = (unsigned __int16)v11; ; i = (unsigned __int16)v11 )
    {
      v13 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a2 + 16LL))(a2, i, 0);
      if ( v13 >= 0 )
      {
        v65 = v13;
        v66 = sub_E91E30(a2, i, a3);
        v67 = sub_2FF7530((__int64)a2, v66);
        v135 = v65;
        v70 = sub_2FF7550((__int64)a2, v66);
        v136 = 0;
        v137 = "super-register";
        v71 = *(unsigned int *)(a1 + 32);
        v72 = v71 + 1;
        if ( v71 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
        {
          v107 = *(_QWORD *)(a1 + 24);
          v108 = a1 + 24;
          v109 = (const void *)(a1 + 40);
          if ( v107 > (unsigned __int64)&v135 || (unsigned __int64)&v135 >= v107 + 24 * v71 )
          {
            sub_C8D5F0(v108, v109, v72, 0x18u, v68, v69);
            v74 = (const __m128i *)&v135;
            v73 = *(_QWORD *)(a1 + 24);
            v71 = *(unsigned int *)(a1 + 32);
          }
          else
          {
            sub_C8D5F0(v108, v109, v72, 0x18u, v68, v69);
            v73 = *(_QWORD *)(a1 + 24);
            v71 = *(unsigned int *)(a1 + 32);
            v74 = (const __m128i *)((char *)&v135 + v73 - v107);
          }
        }
        else
        {
          v73 = *(_QWORD *)(a1 + 24);
          v74 = (const __m128i *)&v135;
        }
        v75 = (__m128i *)(v73 + 24 * v71);
        *v75 = _mm_loadu_si128(v74);
        v75[1].m128i_i64[0] = v74[1].m128i_i64[0];
        ++*(_DWORD *)(a1 + 32);
        *(_WORD *)(a1 + 96) = v67;
        *(_WORD *)(a1 + 98) = v70;
        return 1;
      }
      v14 = *v10++;
      if ( !(_WORD)v14 )
        break;
      v11 += v14;
    }
  }
  v15 = *(unsigned int *)(a2[39]
                        + 16LL
                        * (*(unsigned __int16 *)(*sub_2FF6500((__int64)a2, a3, 1) + 24)
                         + *((_DWORD *)a2 + 82) * (unsigned int)((__int64)(a2[36] - a2[35]) >> 3)));
  LOBYTE(v136) = 0;
  v135 = v15;
  v119 = sub_CA1930(&v135);
  sub_B48880((__int64 *)&v133, v119, 0);
  v16 = (__int16 *)(a2[7] + 2LL * *(unsigned int *)(a2[1] + 24LL * a3 + 4));
  v17 = v16 + 1;
  v18 = *v16;
  v123 = a3 + v18;
  result = 0;
  if ( !(_WORD)v18 )
    goto LABEL_44;
  v121 = 0;
  v19 = v17;
  for ( j = (unsigned __int16)(a3 + v18); ; j = (unsigned __int16)v123 )
  {
    v21 = sub_E91E30(a2, a3, j);
    v125 = sub_2FF7530((__int64)a2, v21);
    v22 = sub_2FF7550((__int64)a2, v21);
    v122 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a2 + 16LL))(a2, j, 0);
    if ( v122 < 0 )
      goto LABEL_41;
    sub_B48880((__int64 *)&v134, v119, 0);
    v25 = v125 + v22;
    if ( v125 + v22 == v22 )
    {
      if ( v25 >= a4 )
        goto LABEL_35;
    }
    else
    {
      v26 = (_QWORD *)v134;
      if ( (v134 & 1) != 0 )
      {
        v134 = 2
             * ((v134 >> 58 << 57)
              | ~(-1LL << (v134 >> 58)) & (((1LL << v25) - (1LL << v22)) | ~(-1LL << (v134 >> 58)) & (v134 >> 1)))
             + 1;
      }
      else
      {
        v23 = v22 & 0x3F;
        v60 = (unsigned __int64 *)(*(_QWORD *)v134 + 8LL * (v22 >> 6));
        v61 = 1LL << v25;
        v62 = *v60;
        if ( v22 >> 6 == v25 >> 6 )
        {
          v24 = (v61 - (1LL << v23)) | v62;
          *v60 = v24;
        }
        else
        {
          v24 = (-1LL << v23) | v62;
          *v60 = v24;
          v63 = ((v22 != 0) + (unsigned int)((v22 - (unsigned __int64)(v22 != 0)) >> 6)) << 6;
          for ( k = v63 + 64; v25 >= k; k += 64 )
          {
            *(_QWORD *)(*v26 + 8LL * ((k - 64) >> 6)) = -1;
            v63 = k;
          }
          if ( v25 > v63 )
            *(_QWORD *)(*v26 + 8LL * (v63 >> 6)) |= v61 - 1;
        }
      }
      if ( v22 >= a4 )
        goto LABEL_33;
    }
    v27 = (unsigned __int64 *)v134;
    v28 = v133;
    v29 = v133 & 1;
    v30 = v134 & 1;
    if ( (v134 & 1) != 0 )
    {
      if ( v29 )
      {
        if ( ((v134 >> 1) & ((-1LL << (v133 >> 58)) | ~(v133 >> 1)) & ~(-1LL << (v134 >> 58))) == 0 )
        {
          if ( v25 == v22 )
            goto LABEL_40;
LABEL_34:
          v133 = 2
               * ((v28 >> 58 << 57)
                | ~(-1LL << (v28 >> 58)) & (~(-1LL << (v28 >> 58)) & (v28 >> 1) | ((1LL << v25) - (1LL << v22))))
               + 1;
          goto LABEL_35;
        }
        goto LABEL_25;
      }
      v82 = *(unsigned int *)(v133 + 64);
      v120 = v134 >> 58;
LABEL_78:
      if ( v120 <= v82 )
        v82 = v120;
      v24 = (unsigned int)v82;
      if ( v82 )
      {
        v23 = (v134 >> 1) & ~(-1LL << (v134 >> 58));
        v83 = 0;
        while ( 1 )
        {
          v84 = v30 ? (v23 >> v83) & 1 : (*(_QWORD *)(*(_QWORD *)v134 + 8LL * (v83 >> 6)) >> v83) & 1LL;
          if ( (_BYTE)v84 )
          {
            v85 = v29
                ? (((v133 >> 1) & ~(-1LL << (v133 >> 58))) >> v83) & 1
                : (*(_QWORD *)(*(_QWORD *)v133 + 8LL * (v83 >> 6)) >> v83) & 1LL;
            if ( !(_BYTE)v85 )
              break;
          }
          if ( (_DWORD)v24 == ++v83 )
          {
            v86 = v83;
            goto LABEL_92;
          }
        }
      }
      else
      {
        v86 = 0;
LABEL_92:
        if ( (_DWORD)v120 == v86 )
        {
LABEL_32:
          if ( v25 == v22 )
            goto LABEL_35;
LABEL_33:
          v28 = v133;
          if ( (v133 & 1) != 0 )
            goto LABEL_34;
          goto LABEL_56;
        }
        v87 = v86;
        v24 = (v134 >> 1) & ~(-1LL << (v134 >> 58));
        while ( 1 )
        {
          v88 = v30 ? (v24 >> v87) & 1 : (*(_QWORD *)(*(_QWORD *)v134 + 8LL * (v87 >> 6)) >> v87) & 1LL;
          if ( (_BYTE)v88 )
            break;
          if ( (_DWORD)v120 == ++v87 )
            goto LABEL_32;
        }
      }
LABEL_25:
      v33 = a1 + 24;
      v34 = *(unsigned int *)(a1 + 32);
      if ( v22 > v121 )
      {
        v89 = v34 + 1;
        v136 = v22 - v121;
        v90 = (const __m128i *)&v135;
        v137 = "no DWARF register encoding";
        v91 = *(unsigned int *)(a1 + 36);
        v135 = -1;
        v92 = *(_QWORD *)(a1 + 24);
        if ( v34 + 1 > v91 )
        {
          v114 = (const void *)(a1 + 40);
          if ( v92 > (unsigned __int64)&v135 || (unsigned __int64)&v135 >= v92 + 24 * v34 )
          {
            sub_C8D5F0(v33, v114, v89, 0x18u, v24, v23);
            v90 = (const __m128i *)&v135;
            v33 = a1 + 24;
            v92 = *(_QWORD *)(a1 + 24);
            v34 = *(unsigned int *)(a1 + 32);
          }
          else
          {
            v115 = (char *)&v135 - v92;
            sub_C8D5F0(v33, v114, v89, 0x18u, v24, v23);
            v33 = a1 + 24;
            v92 = *(_QWORD *)(a1 + 24);
            v34 = *(unsigned int *)(a1 + 32);
            v90 = (const __m128i *)&v115[v92];
          }
        }
        v93 = (__m128i *)(v92 + 24 * v34);
        *v93 = _mm_loadu_si128(v90);
        v93[1].m128i_i64[0] = v90[1].m128i_i64[0];
        v34 = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
        *(_DWORD *)(a1 + 32) = v34;
      }
      v35 = v34 + 1;
      v36 = *(_QWORD *)(a1 + 24);
      v37 = *(unsigned int *)(a1 + 36);
      v38 = v36 + 24 * v34;
      if ( v125 < a4 || v22 )
      {
        v39 = (const __m128i *)&v135;
        v23 = a4 - v22;
        v135 = v122;
        if ( (unsigned int)v23 > v125 )
          v23 = v125;
        v137 = "sub-register";
        v136 = v23;
        if ( v37 >= v35 )
          goto LABEL_31;
        v105 = a1 + 40;
        if ( v36 <= (unsigned __int64)&v135 && v38 > (unsigned __int64)&v135 )
        {
LABEL_122:
          v106 = (char *)&v135 - v36;
          sub_C8D5F0(v33, (const void *)(a1 + 40), v35, 0x18u, v35, v105);
          v36 = *(_QWORD *)(a1 + 24);
          v34 = *(unsigned int *)(a1 + 32);
          v39 = (const __m128i *)&v106[v36];
LABEL_31:
          v40 = (__m128i *)(v36 + 24 * v34);
          *v40 = _mm_loadu_si128(v39);
          v40[1].m128i_i64[0] = v39[1].m128i_i64[0];
          ++*(_DWORD *)(a1 + 32);
          goto LABEL_32;
        }
      }
      else
      {
        v136 = 0;
        v39 = (const __m128i *)&v135;
        v135 = v122;
        v137 = "sub-register";
        if ( v37 >= v35 )
          goto LABEL_31;
        v105 = a1 + 40;
        if ( v36 <= (unsigned __int64)&v135 && v38 > (unsigned __int64)&v135 )
          goto LABEL_122;
      }
      sub_C8D5F0(v33, (const void *)(a1 + 40), v35, 0x18u, v35, v105);
      v39 = (const __m128i *)&v135;
      v36 = *(_QWORD *)(a1 + 24);
      v34 = *(unsigned int *)(a1 + 32);
      goto LABEL_31;
    }
    if ( v29 )
    {
      v82 = v133 >> 58;
      v120 = *(unsigned int *)(v134 + 64);
      goto LABEL_78;
    }
    v23 = *(unsigned int *)(v134 + 8);
    v31 = *(_DWORD *)(v134 + 8);
    if ( *(_DWORD *)(v133 + 8) <= (unsigned int)v23 )
      v31 = *(_DWORD *)(v133 + 8);
    if ( v31 )
    {
      v24 = v31;
      v32 = 0;
      while ( (*(_QWORD *)(*(_QWORD *)v134 + 8 * v32) & ~*(_QWORD *)(*(_QWORD *)v133 + 8 * v32)) == 0 )
      {
        v31 = ++v32;
        if ( v24 == v32 )
          goto LABEL_107;
      }
      goto LABEL_25;
    }
LABEL_107:
    if ( v31 != (_DWORD)v23 )
    {
      while ( !*(_QWORD *)(*(_QWORD *)v134 + 8LL * v31) )
      {
        if ( (_DWORD)v23 == ++v31 )
          goto LABEL_55;
      }
      goto LABEL_25;
    }
LABEL_55:
    if ( v25 == v22 )
    {
LABEL_36:
      if ( v27 )
      {
        if ( (unsigned __int64 *)*v27 != v27 + 2 )
          _libc_free(*v27);
        j_j___libc_free_0((unsigned __int64)v27);
      }
      goto LABEL_40;
    }
LABEL_56:
    v55 = 1LL << v25;
    v23 = v22 & 0x3F;
    v56 = (__int64 *)(*(_QWORD *)v28 + 8LL * (v22 >> 6));
    v57 = *v56;
    if ( v22 >> 6 == v25 >> 6 )
    {
      *v56 = v57 | (v55 - (1LL << v23));
    }
    else
    {
      *v56 = v57 | (-1LL << v23);
      v58 = ((unsigned int)((v22 - (unsigned __int64)(v22 != 0)) >> 6) + (v22 != 0)) << 6;
      for ( m = v58 + 64; v25 >= m; m += 64 )
      {
        *(_QWORD *)(*(_QWORD *)v28 + 8LL * ((m - 64) >> 6)) = -1;
        v58 = m;
      }
      if ( v25 > v58 )
        *(_QWORD *)(*(_QWORD *)v28 + 8LL * (v58 >> 6)) |= v55 - 1;
    }
LABEL_35:
    v27 = (unsigned __int64 *)v134;
    if ( (v134 & 1) == 0 )
      goto LABEL_36;
LABEL_40:
    v121 = v125 + v22;
LABEL_41:
    v41 = *v19++;
    if ( !(_WORD)v41 )
      break;
    v123 += v41;
  }
  result = 0;
  if ( v121 )
  {
    result = 1;
    if ( v121 < v119 )
    {
      v135 = -1;
      v94 = (const __m128i *)&v135;
      v136 = v119 - v121;
      v95 = *(unsigned int *)(a1 + 36);
      v96 = *(_QWORD *)(a1 + 24);
      v137 = "no DWARF register encoding";
      v97 = *(unsigned int *)(a1 + 32);
      v98 = v97 + 1;
      if ( v97 + 1 > v95 )
      {
        v116 = a1 + 24;
        v117 = (const void *)(a1 + 40);
        if ( v96 > (unsigned __int64)&v135 || (unsigned __int64)&v135 >= v96 + 24 * v97 )
        {
          sub_C8D5F0(v116, v117, v98, 0x18u, v98, v23);
          v94 = (const __m128i *)&v135;
          v96 = *(_QWORD *)(a1 + 24);
          v97 = *(unsigned int *)(a1 + 32);
        }
        else
        {
          v118 = (char *)&v135 - v96;
          sub_C8D5F0(v116, v117, v98, 0x18u, v98, v23);
          v96 = *(_QWORD *)(a1 + 24);
          v97 = *(unsigned int *)(a1 + 32);
          v94 = (const __m128i *)&v118[v96];
        }
      }
      v99 = (__m128i *)(v96 + 24 * v97);
      *v99 = _mm_loadu_si128(v94);
      v99[1].m128i_i64[0] = v94[1].m128i_i64[0];
      ++*(_DWORD *)(a1 + 32);
      result = 1;
    }
  }
LABEL_44:
  v42 = v133;
  if ( (v133 & 1) == 0 && v133 )
  {
    if ( *(_QWORD *)v133 != v133 + 16 )
    {
      v131 = result;
      _libc_free(*(_QWORD *)v133);
      result = v131;
    }
    v132 = result;
    j_j___libc_free_0(v42);
    return v132;
  }
  return result;
}
