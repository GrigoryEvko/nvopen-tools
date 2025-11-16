// Function: sub_FD88D0
// Address: 0xfd88d0
//
__int64 __fastcall sub_FD88D0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // r8
  int v12; // eax
  __int64 v13; // r14
  _QWORD *v14; // rbx
  int v15; // edx
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rcx
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rdi
  int v29; // esi
  __int64 v30; // rdi
  int v31; // ecx
  int v32; // esi
  __int64 v33; // rdi
  int v34; // edx
  int v35; // esi
  __int64 v36; // rdi
  int v37; // eax
  int v38; // ecx
  __int64 v39; // rdi
  int v40; // edx
  int v41; // ecx
  __int64 v42; // rdi
  int v43; // eax
  int v44; // edx
  char *v45; // rax
  __int64 v46; // rdx
  char *v47; // rsi
  signed __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rdx
  char *v51; // rcx
  char v52; // cl
  __int64 v53; // r12
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // rdi
  int v62; // esi
  int v63; // r9d
  __int64 v64; // rdi
  int v65; // ecx
  int v66; // esi
  __int64 v67; // rdi
  int v68; // edx
  int v69; // esi
  __int64 v70; // rdi
  int v71; // eax
  int v72; // ecx
  __int64 v73; // rdi
  int v74; // edx
  int v75; // ecx
  __int64 v76; // rdi
  int v77; // eax
  int v78; // edx
  __int64 v80; // rdx
  int v81; // edx
  __int64 v82; // rsi
  _QWORD *v83; // rdi
  unsigned int v84; // eax
  __int64 v85; // rcx
  int v86; // r10d
  int v87; // eax
  int v88; // eax
  int v89; // edx
  __int64 v90; // rsi
  unsigned int v91; // eax
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // [rsp+0h] [rbp-70h]
  __int64 v98; // [rsp+0h] [rbp-70h]
  __int64 v99; // [rsp+8h] [rbp-68h]
  __int64 v100; // [rsp+8h] [rbp-68h]
  __int64 v101; // [rsp+8h] [rbp-68h]
  __int64 v102; // [rsp+8h] [rbp-68h]
  __int64 v103; // [rsp+8h] [rbp-68h]
  __int64 v104; // [rsp+10h] [rbp-60h]
  __int64 v105; // [rsp+10h] [rbp-60h]
  __int64 v106; // [rsp+10h] [rbp-60h]
  __int64 v107; // [rsp+10h] [rbp-60h]
  __int64 v108; // [rsp+10h] [rbp-60h]
  __int64 v109; // [rsp+10h] [rbp-60h]
  __int64 v110; // [rsp+10h] [rbp-60h]
  __int64 v111; // [rsp+18h] [rbp-58h]
  __int64 v112; // [rsp+18h] [rbp-58h]
  __int64 v113; // [rsp+18h] [rbp-58h]
  __int64 v114; // [rsp+18h] [rbp-58h]
  __int64 v115; // [rsp+18h] [rbp-58h]
  __int64 v116; // [rsp+18h] [rbp-58h]
  __int64 v117; // [rsp+18h] [rbp-58h]
  __int64 v118; // [rsp+18h] [rbp-58h]
  __int64 v119; // [rsp+18h] [rbp-58h]
  __int64 v120; // [rsp+18h] [rbp-58h]
  __int64 v121; // [rsp+18h] [rbp-58h]
  _QWORD v122[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v123; // [rsp+30h] [rbp-40h]

  v6 = a1 + 24;
  v9 = a2->m128i_i64[0];
  v122[0] = 0;
  v122[1] = 0;
  v123 = v9;
  if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
    sub_BD73F0((__int64)v122);
  v10 = *(_DWORD *)(a1 + 48);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 24);
LABEL_6:
    sub_FD8550(v6, 2 * v10);
    v12 = *(_DWORD *)(a1 + 48);
    if ( !v12 )
    {
LABEL_7:
      v13 = v123;
      v14 = 0;
LABEL_8:
      v15 = *(_DWORD *)(a1 + 40) + 1;
      goto LABEL_9;
    }
    v13 = v123;
    v81 = v12 - 1;
    v82 = *(_QWORD *)(a1 + 32);
    v83 = 0;
    v11 = 1;
    v84 = (v12 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
    v14 = (_QWORD *)(v82 + 32LL * v84);
    v85 = v14[2];
    if ( v123 == v85 )
      goto LABEL_8;
    while ( v85 != -4096 )
    {
      if ( v85 == -8192 && !v83 )
        v83 = v14;
      a6 = (unsigned int)(v11 + 1);
      v84 = v81 & (v11 + v84);
      v14 = (_QWORD *)(v82 + 32LL * v84);
      v85 = v14[2];
      if ( v123 == v85 )
        goto LABEL_8;
      v11 = (unsigned int)a6;
    }
LABEL_125:
    if ( v83 )
      v14 = v83;
    goto LABEL_8;
  }
  v13 = v123;
  v18 = *(_QWORD *)(a1 + 32);
  v19 = (v10 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
  v20 = (_QWORD *)(v18 + 32LL * v19);
  v11 = v20[2];
  if ( v123 == v11 )
  {
LABEL_20:
    v17 = v20 + 3;
    goto LABEL_21;
  }
  v86 = 1;
  v14 = 0;
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v14 )
      v14 = v20;
    a6 = (unsigned int)(v86 + 1);
    v19 = (v10 - 1) & (v86 + v19);
    v20 = (_QWORD *)(v18 + 32LL * v19);
    v11 = v20[2];
    if ( v123 == v11 )
      goto LABEL_20;
    ++v86;
  }
  if ( !v14 )
    v14 = v20;
  v87 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v15 = v87 + 1;
  if ( 4 * (v87 + 1) >= 3 * v10 )
    goto LABEL_6;
  if ( v10 - *(_DWORD *)(a1 + 44) - v15 <= v10 >> 3 )
  {
    sub_FD8550(v6, v10);
    v88 = *(_DWORD *)(a1 + 48);
    if ( !v88 )
      goto LABEL_7;
    v13 = v123;
    v89 = v88 - 1;
    v90 = *(_QWORD *)(a1 + 32);
    v83 = 0;
    v11 = 1;
    v91 = (v88 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
    v14 = (_QWORD *)(v90 + 32LL * v91);
    v92 = v14[2];
    if ( v123 == v92 )
      goto LABEL_8;
    while ( v92 != -4096 )
    {
      if ( v92 == -8192 && !v83 )
        v83 = v14;
      a6 = (unsigned int)(v11 + 1);
      v91 = v89 & (v11 + v91);
      v14 = (_QWORD *)(v90 + 32LL * v91);
      v92 = v14[2];
      if ( v123 == v92 )
        goto LABEL_8;
      v11 = (unsigned int)a6;
    }
    goto LABEL_125;
  }
LABEL_9:
  *(_DWORD *)(a1 + 40) = v15;
  if ( v14[2] == -4096 )
  {
    if ( v13 != -4096 )
    {
LABEL_14:
      v14[2] = v13;
      if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
        sub_BD73F0((__int64)v14);
      v13 = v123;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 44);
    v16 = v14[2];
    if ( v16 != v13 )
    {
      if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
        sub_BD60C0(v14);
      goto LABEL_14;
    }
  }
  v14[3] = 0;
  v17 = v14 + 3;
LABEL_21:
  if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
    sub_BD60C0(v122);
  v21 = *v17;
  if ( !*v17 )
    goto LABEL_63;
  v22 = *(_QWORD *)(v21 + 16);
  if ( v22 )
  {
    v23 = *(_QWORD *)(v22 + 16);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 16);
      if ( v24 )
      {
        v25 = *(_QWORD *)(v24 + 16);
        if ( v25 )
        {
          v11 = *(_QWORD *)(v25 + 16);
          if ( v11 )
          {
            v111 = *(_QWORD *)(v24 + 16);
            if ( *(_QWORD *)(v11 + 16) )
            {
              v99 = *(_QWORD *)(v23 + 16);
              v104 = *(_QWORD *)(v22 + 16);
              sub_FD5AF0(a1, (__int64 *)(v11 + 16));
              v26 = v111;
              v23 = v104;
              v24 = v99;
              v27 = *(_QWORD *)(*(_QWORD *)(v111 + 16) + 16LL);
              *(_DWORD *)(v27 + 64) = (*(_DWORD *)(v27 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v27 + 64) & 0xF8000000;
              v28 = *(_QWORD *)(v111 + 16);
              v29 = *(_DWORD *)(v28 + 64);
              a6 = (v29 + 0x7FFFFFF) & 0x7FFFFFF;
              *(_DWORD *)(v28 + 64) = a6 | v29 & 0xF8000000;
              if ( !(_DWORD)a6 )
              {
                v97 = v27;
                sub_FD59A0(v28, a1);
                v27 = v97;
                v24 = v99;
                v23 = v104;
                v26 = v111;
              }
              *(_QWORD *)(v26 + 16) = v27;
              v11 = *(_QWORD *)(*(_QWORD *)(v24 + 16) + 16LL);
            }
            *(_DWORD *)(v11 + 64) = (*(_DWORD *)(v11 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v11 + 64) & 0xF8000000;
            v30 = *(_QWORD *)(v24 + 16);
            v31 = *(_DWORD *)(v30 + 64);
            v32 = (v31 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v30 + 64) = v32 | v31 & 0xF8000000;
            if ( !v32 )
            {
              v101 = v24;
              v108 = v23;
              v119 = v11;
              sub_FD59A0(v30, a1);
              v24 = v101;
              v23 = v108;
              v11 = v119;
            }
            *(_QWORD *)(v24 + 16) = v11;
            v25 = *(_QWORD *)(*(_QWORD *)(v23 + 16) + 16LL);
          }
          *(_DWORD *)(v25 + 64) = (*(_DWORD *)(v25 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v25 + 64) & 0xF8000000;
          v33 = *(_QWORD *)(v23 + 16);
          v34 = *(_DWORD *)(v33 + 64);
          v35 = (v34 + 0x7FFFFFF) & 0x7FFFFFF;
          *(_DWORD *)(v33 + 64) = v35 | v34 & 0xF8000000;
          if ( !v35 )
          {
            v106 = v23;
            v117 = v25;
            sub_FD59A0(v33, a1);
            v23 = v106;
            v25 = v117;
          }
          *(_QWORD *)(v23 + 16) = v25;
          v24 = *(_QWORD *)(*(_QWORD *)(v22 + 16) + 16LL);
        }
        *(_DWORD *)(v24 + 64) = (*(_DWORD *)(v24 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v24 + 64) & 0xF8000000;
        v36 = *(_QWORD *)(v22 + 16);
        v37 = *(_DWORD *)(v36 + 64);
        v38 = (v37 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v36 + 64) = v38 | v37 & 0xF8000000;
        if ( !v38 )
        {
          v116 = v24;
          sub_FD59A0(v36, a1);
          v24 = v116;
        }
        *(_QWORD *)(v22 + 16) = v24;
        v23 = *(_QWORD *)(*(_QWORD *)(v21 + 16) + 16LL);
      }
      *(_DWORD *)(v23 + 64) = (*(_DWORD *)(v23 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v23 + 64) & 0xF8000000;
      v39 = *(_QWORD *)(v21 + 16);
      v40 = *(_DWORD *)(v39 + 64);
      v41 = (v40 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(v39 + 64) = v41 | v40 & 0xF8000000;
      if ( !v41 )
      {
        v113 = v23;
        sub_FD59A0(v39, a1);
        v23 = v113;
      }
      *(_QWORD *)(v21 + 16) = v23;
      v21 = *(_QWORD *)(*v17 + 16);
    }
    else
    {
      v21 = *(_QWORD *)(v21 + 16);
    }
    *(_DWORD *)(v21 + 64) = (*(_DWORD *)(v21 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v21 + 64) & 0xF8000000;
    v42 = *v17;
    v43 = *(_DWORD *)(*v17 + 64);
    v44 = (v43 + 0x7FFFFFF) & 0x7FFFFFF;
    *(_DWORD *)(*v17 + 64) = v44 | v43 & 0xF8000000;
    if ( !v44 )
      sub_FD59A0(v42, a1);
    *v17 = v21;
  }
  v45 = *(char **)(v21 + 24);
  v46 = 48LL * *(unsigned int *)(v21 + 32);
  v47 = &v45[v46];
  v48 = 0xAAAAAAAAAAAAAAABLL * (v46 >> 4);
  v49 = v48 >> 2;
  if ( v48 >> 2 )
  {
    v50 = a2->m128i_i64[0];
    v51 = &v45[192 * v49];
    while ( *(_QWORD *)v45 != v50
         || *((_QWORD *)v45 + 1) != a2->m128i_i64[1]
         || *((_QWORD *)v45 + 2) != a2[1].m128i_i64[0]
         || *((_QWORD *)v45 + 3) != a2[1].m128i_i64[1]
         || *((_QWORD *)v45 + 4) != a2[2].m128i_i64[0]
         || *((_QWORD *)v45 + 5) != a2[2].m128i_i64[1] )
    {
      if ( v50 == *((_QWORD *)v45 + 6)
        && *((_QWORD *)v45 + 7) == a2->m128i_i64[1]
        && *((_QWORD *)v45 + 8) == a2[1].m128i_i64[0]
        && *((_QWORD *)v45 + 9) == a2[1].m128i_i64[1]
        && *((_QWORD *)v45 + 10) == a2[2].m128i_i64[0]
        && *((_QWORD *)v45 + 11) == a2[2].m128i_i64[1] )
      {
        v45 += 48;
        break;
      }
      if ( v50 == *((_QWORD *)v45 + 12)
        && *((_QWORD *)v45 + 13) == a2->m128i_i64[1]
        && *((_QWORD *)v45 + 14) == a2[1].m128i_i64[0]
        && *((_QWORD *)v45 + 15) == a2[1].m128i_i64[1]
        && *((_QWORD *)v45 + 16) == a2[2].m128i_i64[0]
        && *((_QWORD *)v45 + 17) == a2[2].m128i_i64[1] )
      {
        v45 += 96;
        break;
      }
      if ( v50 == *((_QWORD *)v45 + 18)
        && *((_QWORD *)v45 + 19) == a2->m128i_i64[1]
        && *((_QWORD *)v45 + 20) == a2[1].m128i_i64[0]
        && *((_QWORD *)v45 + 21) == a2[1].m128i_i64[1]
        && *((_QWORD *)v45 + 22) == a2[2].m128i_i64[0]
        && *((_QWORD *)v45 + 23) == a2[2].m128i_i64[1] )
      {
        v45 += 144;
        break;
      }
      v45 += 192;
      if ( v51 == v45 )
      {
        v48 = 0xAAAAAAAAAAAAAAABLL * ((v47 - v45) >> 4);
        goto LABEL_112;
      }
    }
LABEL_62:
    if ( v47 != v45 )
      return v21;
    goto LABEL_63;
  }
LABEL_112:
  if ( v48 == 2 )
  {
    v80 = a2->m128i_i64[0];
LABEL_146:
    if ( *(_QWORD *)v45 == v80
      && *((_QWORD *)v45 + 1) == a2->m128i_i64[1]
      && *((_QWORD *)v45 + 2) == a2[1].m128i_i64[0]
      && *((_QWORD *)v45 + 3) == a2[1].m128i_i64[1]
      && *((_QWORD *)v45 + 4) == a2[2].m128i_i64[0]
      && *((_QWORD *)v45 + 5) == a2[2].m128i_i64[1] )
    {
      goto LABEL_62;
    }
    v45 += 48;
    goto LABEL_116;
  }
  if ( v48 == 3 )
  {
    v80 = a2->m128i_i64[0];
    if ( *(_QWORD *)v45 == a2->m128i_i64[0]
      && *((_QWORD *)v45 + 1) == a2->m128i_i64[1]
      && *((_QWORD *)v45 + 2) == a2[1].m128i_i64[0]
      && *((_QWORD *)v45 + 3) == a2[1].m128i_i64[1]
      && *((_QWORD *)v45 + 4) == a2[2].m128i_i64[0]
      && *((_QWORD *)v45 + 5) == a2[2].m128i_i64[1] )
    {
      goto LABEL_62;
    }
    v45 += 48;
    goto LABEL_146;
  }
  if ( v48 != 1 )
    goto LABEL_63;
  v80 = a2->m128i_i64[0];
LABEL_116:
  if ( v80 == *(_QWORD *)v45
    && *((_QWORD *)v45 + 1) == a2->m128i_i64[1]
    && *((_QWORD *)v45 + 2) == a2[1].m128i_i64[0]
    && *((_QWORD *)v45 + 3) == a2[1].m128i_i64[1]
    && *((_QWORD *)v45 + 4) == a2[2].m128i_i64[0]
    && *((_QWORD *)v45 + 5) == a2[2].m128i_i64[1] )
  {
    goto LABEL_62;
  }
LABEL_63:
  v21 = *(_QWORD *)(a1 + 64);
  LOBYTE(v122[0]) = 0;
  v52 = 0;
  if ( !v21 )
  {
    v21 = sub_FD7AA0(a1, a2, *v17, v122, v11, a6);
    if ( v21 )
    {
      v52 = v122[0];
    }
    else
    {
      v93 = sub_22077B0(72);
      v21 = v93;
      if ( v93 )
      {
        *(_QWORD *)(v93 + 16) = 0;
        v94 = 0;
        *(_DWORD *)(v93 + 64) &= 0x80000000;
        *(_QWORD *)(v93 + 24) = v93 + 40;
        *(_QWORD *)(v93 + 32) = 0;
        *(_QWORD *)(v93 + 40) = 0;
        *(_QWORD *)(v93 + 48) = 0;
        *(_QWORD *)(v93 + 56) = 0;
      }
      else
      {
        v94 = MEMORY[0] & 7;
      }
      v95 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(v21 + 8) = a1 + 8;
      v52 = 1;
      v95 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v21 = v95 | v94;
      *(_QWORD *)(v95 + 8) = v21;
      v96 = *(_QWORD *)(a1 + 8);
      LOBYTE(v122[0]) = 1;
      *(_QWORD *)(a1 + 8) = v21 | v96 & 7;
    }
  }
  sub_FD5CC0(v21, a1, a2, v52, v11, a6);
  v53 = *v17;
  if ( *v17 )
  {
    v54 = *(_QWORD *)(v53 + 16);
    if ( v54 )
    {
      v55 = *(_QWORD *)(v54 + 16);
      if ( v55 )
      {
        v56 = *(_QWORD *)(v55 + 16);
        if ( v56 )
        {
          v57 = *(_QWORD *)(v56 + 16);
          if ( v57 )
          {
            v58 = *(_QWORD *)(v57 + 16);
            if ( v58 )
            {
              v112 = *(_QWORD *)(v56 + 16);
              if ( *(_QWORD *)(v58 + 16) )
              {
                v100 = *(_QWORD *)(v55 + 16);
                v105 = *(_QWORD *)(v54 + 16);
                sub_FD5AF0(a1, (__int64 *)(v58 + 16));
                v59 = v112;
                v55 = v105;
                v56 = v100;
                v60 = *(_QWORD *)(*(_QWORD *)(v112 + 16) + 16LL);
                *(_DWORD *)(v60 + 64) = (*(_DWORD *)(v60 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v60 + 64) & 0xF8000000;
                v61 = *(_QWORD *)(v112 + 16);
                v62 = *(_DWORD *)(v61 + 64);
                v63 = (v62 + 0x7FFFFFF) & 0x7FFFFFF;
                *(_DWORD *)(v61 + 64) = v63 | v62 & 0xF8000000;
                if ( !v63 )
                {
                  v98 = v60;
                  v103 = v112;
                  v110 = v56;
                  v121 = v55;
                  sub_FD59A0(v61, a1);
                  v60 = v98;
                  v59 = v103;
                  v56 = v110;
                  v55 = v121;
                }
                *(_QWORD *)(v59 + 16) = v60;
                v58 = *(_QWORD *)(*(_QWORD *)(v56 + 16) + 16LL);
              }
              *(_DWORD *)(v58 + 64) = (*(_DWORD *)(v58 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v58 + 64) & 0xF8000000;
              v64 = *(_QWORD *)(v56 + 16);
              v65 = *(_DWORD *)(v64 + 64);
              v66 = (v65 + 0x7FFFFFF) & 0x7FFFFFF;
              *(_DWORD *)(v64 + 64) = v66 | v65 & 0xF8000000;
              if ( !v66 )
              {
                v102 = v58;
                v109 = v56;
                v120 = v55;
                sub_FD59A0(v64, a1);
                v58 = v102;
                v56 = v109;
                v55 = v120;
              }
              *(_QWORD *)(v56 + 16) = v58;
              v57 = *(_QWORD *)(*(_QWORD *)(v55 + 16) + 16LL);
            }
            *(_DWORD *)(v57 + 64) = (*(_DWORD *)(v57 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v57 + 64) & 0xF8000000;
            v67 = *(_QWORD *)(v55 + 16);
            v68 = *(_DWORD *)(v67 + 64);
            v69 = (v68 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v67 + 64) = v69 | v68 & 0xF8000000;
            if ( !v69 )
            {
              v107 = v57;
              v118 = v55;
              sub_FD59A0(v67, a1);
              v57 = v107;
              v55 = v118;
            }
            *(_QWORD *)(v55 + 16) = v57;
            v56 = *(_QWORD *)(*(_QWORD *)(v54 + 16) + 16LL);
          }
          *(_DWORD *)(v56 + 64) = (*(_DWORD *)(v56 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v56 + 64) & 0xF8000000;
          v70 = *(_QWORD *)(v54 + 16);
          v71 = *(_DWORD *)(v70 + 64);
          v72 = (v71 + 0x7FFFFFF) & 0x7FFFFFF;
          *(_DWORD *)(v70 + 64) = v72 | v71 & 0xF8000000;
          if ( !v72 )
          {
            v114 = v56;
            sub_FD59A0(v70, a1);
            v56 = v114;
          }
          *(_QWORD *)(v54 + 16) = v56;
          v55 = *(_QWORD *)(*(_QWORD *)(v53 + 16) + 16LL);
        }
        *(_DWORD *)(v55 + 64) = (*(_DWORD *)(v55 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v55 + 64) & 0xF8000000;
        v73 = *(_QWORD *)(v53 + 16);
        v74 = *(_DWORD *)(v73 + 64);
        v75 = (v74 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v73 + 64) = v75 | v74 & 0xF8000000;
        if ( !v75 )
        {
          v115 = v55;
          sub_FD59A0(v73, a1);
          v55 = v115;
        }
        *(_QWORD *)(v53 + 16) = v55;
        v54 = *(_QWORD *)(*v17 + 16);
      }
      *(_DWORD *)(v54 + 64) = (*(_DWORD *)(v54 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v54 + 64) & 0xF8000000;
      v76 = *v17;
      v77 = *(_DWORD *)(*v17 + 64);
      v78 = (v77 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(*v17 + 64) = v78 | v77 & 0xF8000000;
      if ( !v78 )
        sub_FD59A0(v76, a1);
      *v17 = v54;
    }
  }
  else
  {
    *(_DWORD *)(v21 + 64) = (*(_DWORD *)(v21 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v21 + 64) & 0xF8000000;
    *v17 = v21;
  }
  return v21;
}
