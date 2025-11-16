// Function: sub_20F1CA0
// Address: 0x20f1ca0
//
__int64 *__fastcall sub_20F1CA0(__int64 a1, __int64 a2, __int32 a3, int a4, int a5)
{
  int v5; // r10d
  unsigned int v6; // eax
  __int64 v7; // r9
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // edi
  int *v17; // rcx
  int v18; // r11d
  __int64 v19; // rdi
  unsigned __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // rdi
  unsigned int v23; // r9d
  __int64 *v24; // rax
  __int64 v25; // r14
  __int64 v26; // rbx
  unsigned int v27; // r9d
  int *v28; // rax
  int v29; // edi
  __int64 v30; // r14
  unsigned __int64 v31; // rbx
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 *result; // rax
  unsigned int v36; // eax
  __int64 v37; // rcx
  int v38; // eax
  int v39; // eax
  int v40; // esi
  __int64 v41; // r10
  unsigned int v42; // edx
  int v43; // eax
  __int32 *v44; // rcx
  int v45; // edi
  int v46; // r11d
  int *v47; // r9
  __int64 v48; // r15
  __int64 v49; // rdi
  _QWORD *v50; // rsi
  _QWORD *v51; // rdx
  __int64 *v52; // rsi
  unsigned int v53; // r8d
  __int64 *v54; // rcx
  int v55; // ecx
  _QWORD *v56; // rax
  int v57; // r8d
  __int64 v58; // rbx
  __int64 *v59; // r9
  __int64 *v60; // r14
  int v61; // r15d
  __int64 *v62; // r12
  _QWORD *v63; // r13
  __int64 v64; // rbx
  __int64 v65; // rax
  int v66; // r9d
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // r14
  __int64 v70; // r9
  __int64 v71; // rax
  __int64 v72; // r13
  __int64 v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rcx
  __m128i *v76; // rax
  unsigned int v77; // esi
  __int64 v78; // r9
  __int64 v79; // rdi
  int *v80; // rax
  int v81; // ecx
  unsigned __int64 *v82; // r15
  unsigned __int64 v83; // r14
  __int64 v84; // rbx
  __int64 v85; // rdi
  unsigned __int64 v86; // rdi
  int v87; // r11d
  int v88; // eax
  int v89; // r8d
  int v90; // r8d
  int v91; // eax
  int v92; // edx
  __int64 v93; // r9
  int v94; // edi
  int *v95; // r10
  unsigned int v96; // r14d
  int v97; // esi
  int v98; // r11d
  __int32 *v99; // rdx
  int v100; // eax
  int v101; // ecx
  int v102; // r9d
  int v103; // r9d
  __int64 v104; // r10
  __int64 v105; // r14
  int v106; // esi
  int v107; // eax
  int *v108; // rdi
  int v109; // r9d
  int v110; // r9d
  __int64 v111; // r10
  unsigned int v112; // r14d
  int v113; // eax
  int v114; // esi
  int *v115; // r15
  __int64 v116; // [rsp+8h] [rbp-98h]
  const void *v117; // [rsp+10h] [rbp-90h]
  __int64 v118; // [rsp+18h] [rbp-88h]
  __int32 v119; // [rsp+24h] [rbp-7Ch]
  __int64 v120; // [rsp+28h] [rbp-78h]
  __int64 v121; // [rsp+30h] [rbp-70h]
  __int64 v122; // [rsp+30h] [rbp-70h]
  __int64 v123; // [rsp+38h] [rbp-68h]
  __int64 v125; // [rsp+40h] [rbp-60h]
  __int64 v126; // [rsp+40h] [rbp-60h]
  __int64 *v127; // [rsp+40h] [rbp-60h]
  __int32 v128; // [rsp+40h] [rbp-60h]
  unsigned int v129; // [rsp+40h] [rbp-60h]
  __m128i v131; // [rsp+50h] [rbp-50h] BYREF
  __int64 v132; // [rsp+60h] [rbp-40h]

  v5 = a4;
  v6 = a4 & 0x7FFFFFFF;
  v7 = a4 & 0x7FFFFFFF;
  v9 = a1;
  v10 = 8 * v7;
  v11 = *(_QWORD *)(a1 + 16);
  v12 = *(unsigned int *)(v11 + 408);
  if ( (a4 & 0x7FFFFFFFu) >= (unsigned int)v12 || (v13 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8LL * v6)) == 0 )
  {
    v36 = v6 + 1;
    if ( (unsigned int)v12 < v36 )
    {
      v48 = v36;
      if ( v36 < v12 )
      {
        *(_DWORD *)(v11 + 408) = v36;
      }
      else if ( v36 > v12 )
      {
        if ( v36 > (unsigned __int64)*(unsigned int *)(v11 + 412) )
        {
          v129 = v36;
          v122 = a4 & 0x7FFFFFFF;
          sub_16CD150(v11 + 400, (const void *)(v11 + 416), v36, 8, a5, v7);
          v12 = *(unsigned int *)(v11 + 408);
          v7 = v122;
          v5 = a4;
          v36 = v129;
        }
        v37 = *(_QWORD *)(v11 + 400);
        v49 = *(_QWORD *)(v11 + 416);
        v50 = (_QWORD *)(v37 + 8 * v48);
        v51 = (_QWORD *)(v37 + 8 * v12);
        if ( v50 != v51 )
        {
          do
            *v51++ = v49;
          while ( v50 != v51 );
          v37 = *(_QWORD *)(v11 + 400);
        }
        *(_DWORD *)(v11 + 408) = v36;
        goto LABEL_22;
      }
    }
    v37 = *(_QWORD *)(v11 + 400);
LABEL_22:
    v125 = v7;
    *(_QWORD *)(v37 + v10) = sub_1DBA290(v5);
    v13 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8 * v125);
    sub_1DBB110((_QWORD *)v11, v13);
  }
  v14 = *(_QWORD *)(v9 + 256);
  v123 = v9 + 248;
  v15 = *(unsigned int *)(v9 + 272);
  if ( (_DWORD)v15 )
  {
    v16 = (v15 - 1) & (37 * a3);
    v17 = (int *)(v14 + 16LL * v16);
    v18 = *v17;
    if ( *v17 == a3 )
    {
LABEL_5:
      if ( (int *)(v14 + 16 * v15) != v17 )
        goto LABEL_6;
    }
    else
    {
      v55 = 1;
      while ( v18 != 0x7FFFFFFF )
      {
        v89 = v55 + 1;
        v16 = (v15 - 1) & (v55 + v16);
        v17 = (int *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( *v17 == a3 )
          goto LABEL_5;
        v55 = v89;
      }
    }
  }
  v126 = *(_QWORD *)(v13 + 112);
  v56 = (_QWORD *)sub_22077B0(120);
  v58 = (__int64)v56;
  if ( v56 )
  {
    v56[12] = 0;
    *v56 = v56 + 2;
    v56[1] = 0x200000000LL;
    v56[8] = v56 + 10;
    v56[9] = 0x200000000LL;
    v56[13] = 0;
    v56[14] = v126;
  }
  if ( v56 != (_QWORD *)v13 )
  {
    v59 = *(__int64 **)(v13 + 64);
    v127 = &v59[*(unsigned int *)(v13 + 72)];
    if ( v59 != v127 )
    {
      v121 = v13;
      v118 = (__int64)(v56 + 8);
      v60 = (__int64 *)(v11 + 296);
      v117 = v56 + 10;
      v61 = *((_DWORD *)v56 + 18);
      v120 = v9;
      v62 = v59;
      v119 = a3;
      v63 = v56;
      do
      {
        v64 = *v62;
        v65 = sub_145CBF0(v60, 16, 16);
        v67 = *(_QWORD *)(v64 + 8);
        *(_DWORD *)v65 = v61;
        *(_QWORD *)(v65 + 8) = v67;
        v68 = *((unsigned int *)v63 + 18);
        if ( (unsigned int)v68 >= *((_DWORD *)v63 + 19) )
        {
          v116 = v65;
          sub_16CD150(v118, v117, 0, 8, v57, v66);
          v68 = *((unsigned int *)v63 + 18);
          v65 = v116;
        }
        ++v62;
        *(_QWORD *)(v63[8] + 8 * v68) = v65;
        v61 = *((_DWORD *)v63 + 18) + 1;
        *((_DWORD *)v63 + 18) = v61;
      }
      while ( v127 != v62 );
      v58 = (__int64)v63;
      v13 = v121;
      v9 = v120;
      a3 = v119;
    }
    v69 = *(_QWORD *)v13;
    v70 = *(_QWORD *)v13 + 24LL * *(unsigned int *)(v13 + 8);
    if ( *(_QWORD *)v13 != v70 )
    {
      v71 = *(unsigned int *)(v58 + 8);
      v128 = a3;
      v72 = v9;
      v73 = *(_QWORD *)v13 + 24LL * *(unsigned int *)(v13 + 8);
      do
      {
        v74 = *(_QWORD *)(*(_QWORD *)(v58 + 64) + 8LL * **(unsigned int **)(v69 + 16));
        v75 = *(_QWORD *)(v69 + 8);
        v131.m128i_i64[0] = *(_QWORD *)v69;
        v131.m128i_i64[1] = v75;
        v132 = v74;
        if ( *(_DWORD *)(v58 + 12) <= (unsigned int)v71 )
        {
          sub_16CD150(v58, (const void *)(v58 + 16), 0, 24, v57, v70);
          v71 = *(unsigned int *)(v58 + 8);
        }
        v69 += 24;
        v76 = (__m128i *)(*(_QWORD *)v58 + 24 * v71);
        *v76 = _mm_loadu_si128(&v131);
        v76[1].m128i_i64[0] = v132;
        v71 = (unsigned int)(*(_DWORD *)(v58 + 8) + 1);
        *(_DWORD *)(v58 + 8) = v71;
      }
      while ( v73 != v69 );
      v9 = v72;
      a3 = v128;
    }
  }
  v77 = *(_DWORD *)(v9 + 272);
  if ( !v77 )
  {
    ++*(_QWORD *)(v9 + 248);
    goto LABEL_112;
  }
  v78 = *(_QWORD *)(v9 + 256);
  LODWORD(v79) = (v77 - 1) & (37 * a3);
  v80 = (int *)(v78 + 16LL * (unsigned int)v79);
  v81 = *v80;
  if ( *v80 != a3 )
  {
    v98 = 1;
    v99 = 0;
    while ( v81 != 0x7FFFFFFF )
    {
      if ( !v99 && v81 == 0x80000000 )
        v99 = v80;
      v79 = (v77 - 1) & ((_DWORD)v79 + v98);
      v80 = (int *)(v78 + 16 * v79);
      v81 = *v80;
      if ( *v80 == a3 )
        goto LABEL_72;
      ++v98;
    }
    if ( !v99 )
      v99 = v80;
    v100 = *(_DWORD *)(v9 + 264);
    ++*(_QWORD *)(v9 + 248);
    v101 = v100 + 1;
    if ( 4 * (v100 + 1) < 3 * v77 )
    {
      if ( v77 - *(_DWORD *)(v9 + 268) - v101 > v77 >> 3 )
      {
LABEL_108:
        *(_DWORD *)(v9 + 264) = v101;
        if ( *v99 != 0x7FFFFFFF )
          --*(_DWORD *)(v9 + 268);
        *v99 = a3;
        *((_QWORD *)v99 + 1) = v58;
        goto LABEL_82;
      }
      sub_20EBB90(v123, v77);
      v109 = *(_DWORD *)(v9 + 272);
      if ( v109 )
      {
        v110 = v109 - 1;
        v111 = *(_QWORD *)(v9 + 256);
        v108 = 0;
        v112 = v110 & (37 * a3);
        v101 = *(_DWORD *)(v9 + 264) + 1;
        v113 = 1;
        v99 = (__int32 *)(v111 + 16LL * v112);
        v114 = *v99;
        if ( *v99 == a3 )
          goto LABEL_108;
        while ( v114 != 0x7FFFFFFF )
        {
          if ( !v108 && v114 == 0x80000000 )
            v108 = v99;
          v112 = v110 & (v113 + v112);
          v99 = (__int32 *)(v111 + 16LL * v112);
          v114 = *v99;
          if ( *v99 == a3 )
            goto LABEL_108;
          ++v113;
        }
        goto LABEL_116;
      }
      goto LABEL_154;
    }
LABEL_112:
    sub_20EBB90(v123, 2 * v77);
    v102 = *(_DWORD *)(v9 + 272);
    if ( v102 )
    {
      v103 = v102 - 1;
      v104 = *(_QWORD *)(v9 + 256);
      v105 = v103 & (unsigned int)(37 * a3);
      v99 = (__int32 *)(v104 + 16 * v105);
      v101 = *(_DWORD *)(v9 + 264) + 1;
      v106 = *v99;
      if ( *v99 == a3 )
        goto LABEL_108;
      v107 = 1;
      v108 = 0;
      while ( v106 != 0x7FFFFFFF )
      {
        if ( v106 == 0x80000000 && !v108 )
          v108 = v99;
        LODWORD(v105) = v103 & (v107 + v105);
        v99 = (__int32 *)(v104 + 16LL * (unsigned int)v105);
        v106 = *v99;
        if ( *v99 == a3 )
          goto LABEL_108;
        ++v107;
      }
LABEL_116:
      if ( v108 )
        v99 = v108;
      goto LABEL_108;
    }
LABEL_154:
    ++*(_DWORD *)(v9 + 264);
    BUG();
  }
LABEL_72:
  v82 = (unsigned __int64 *)*((_QWORD *)v80 + 1);
  *((_QWORD *)v80 + 1) = v58;
  if ( v82 )
  {
    sub_1DB4CE0((__int64)v82);
    v83 = v82[12];
    if ( v83 )
    {
      v84 = *(_QWORD *)(v83 + 16);
      while ( v84 )
      {
        sub_20EA3F0(*(_QWORD *)(v84 + 24));
        v85 = v84;
        v84 = *(_QWORD *)(v84 + 16);
        j_j___libc_free_0(v85, 56);
      }
      j_j___libc_free_0(v83, 48);
    }
    v86 = v82[8];
    if ( (unsigned __int64 *)v86 != v82 + 10 )
      _libc_free(v86);
    if ( (unsigned __int64 *)*v82 != v82 + 2 )
      _libc_free(*v82);
    j_j___libc_free_0(v82, 120);
  }
LABEL_82:
  v14 = *(_QWORD *)(v9 + 256);
  LODWORD(v15) = *(_DWORD *)(v9 + 272);
LABEL_6:
  v19 = *(_QWORD *)(*(_QWORD *)(v9 + 16) + 272LL);
  v20 = a2;
  if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
  {
    do
      v20 = *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v20 + 46) & 4) != 0 );
  }
  v21 = *(_QWORD *)(v19 + 368);
  v22 = *(unsigned int *)(v19 + 384);
  if ( (_DWORD)v22 )
  {
    v23 = (v22 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v24 = (__int64 *)(v21 + 16LL * v23);
    v25 = *v24;
    if ( *v24 == v20 )
    {
LABEL_10:
      v26 = v24[1];
      if ( (_DWORD)v15 )
        goto LABEL_11;
LABEL_26:
      ++*(_QWORD *)(v9 + 248);
      goto LABEL_27;
    }
    v38 = 1;
    while ( v25 != -8 )
    {
      v90 = v38 + 1;
      v23 = (v22 - 1) & (v38 + v23);
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( *v24 == v20 )
        goto LABEL_10;
      v38 = v90;
    }
  }
  v26 = *(_QWORD *)(v21 + 16 * v22 + 8);
  if ( !(_DWORD)v15 )
    goto LABEL_26;
LABEL_11:
  v27 = (v15 - 1) & (37 * a3);
  v28 = (int *)(v14 + 16LL * v27);
  v29 = *v28;
  if ( *v28 == a3 )
  {
    v30 = *((_QWORD *)v28 + 1);
    goto LABEL_13;
  }
  v87 = 1;
  v44 = 0;
  while ( v29 != 0x7FFFFFFF )
  {
    if ( v44 || v29 != 0x80000000 )
      v28 = v44;
    v27 = (v15 - 1) & (v87 + v27);
    v115 = (int *)(v14 + 16LL * v27);
    v29 = *v115;
    if ( *v115 == a3 )
    {
      v30 = *((_QWORD *)v115 + 1);
      goto LABEL_13;
    }
    ++v87;
    v44 = v28;
    v28 = (int *)(v14 + 16LL * v27);
  }
  if ( !v44 )
    v44 = v28;
  v88 = *(_DWORD *)(v9 + 264);
  ++*(_QWORD *)(v9 + 248);
  v43 = v88 + 1;
  if ( 4 * v43 >= (unsigned int)(3 * v15) )
  {
LABEL_27:
    sub_20EBB90(v123, 2 * v15);
    v39 = *(_DWORD *)(v9 + 272);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(v9 + 256);
      v42 = (v39 - 1) & (37 * a3);
      v43 = *(_DWORD *)(v9 + 264) + 1;
      v44 = (__int32 *)(v41 + 16LL * v42);
      v45 = *v44;
      if ( *v44 != a3 )
      {
        v46 = 1;
        v47 = 0;
        while ( v45 != 0x7FFFFFFF )
        {
          if ( v45 == 0x80000000 && !v47 )
            v47 = v44;
          v42 = v40 & (v46 + v42);
          v44 = (__int32 *)(v41 + 16LL * v42);
          v45 = *v44;
          if ( *v44 == a3 )
            goto LABEL_89;
          ++v46;
        }
        if ( v47 )
          v44 = v47;
      }
      goto LABEL_89;
    }
    goto LABEL_153;
  }
  if ( (int)v15 - (v43 + *(_DWORD *)(v9 + 268)) <= (unsigned int)v15 >> 3 )
  {
    sub_20EBB90(v123, v15);
    v91 = *(_DWORD *)(v9 + 272);
    if ( v91 )
    {
      v92 = v91 - 1;
      v93 = *(_QWORD *)(v9 + 256);
      v94 = 1;
      v95 = 0;
      v96 = (v91 - 1) & (37 * a3);
      v43 = *(_DWORD *)(v9 + 264) + 1;
      v44 = (__int32 *)(v93 + 16LL * v96);
      v97 = *v44;
      if ( *v44 != a3 )
      {
        while ( v97 != 0x7FFFFFFF )
        {
          if ( !v95 && v97 == 0x80000000 )
            v95 = v44;
          v96 = v92 & (v94 + v96);
          v44 = (__int32 *)(v93 + 16LL * v96);
          v97 = *v44;
          if ( *v44 == a3 )
            goto LABEL_89;
          ++v94;
        }
        if ( v95 )
          v44 = v95;
      }
      goto LABEL_89;
    }
LABEL_153:
    ++*(_DWORD *)(v9 + 264);
    BUG();
  }
LABEL_89:
  *(_DWORD *)(v9 + 264) = v43;
  if ( *v44 != 0x7FFFFFFF )
    --*(_DWORD *)(v9 + 268);
  *v44 = a3;
  v30 = 0;
  *((_QWORD *)v44 + 1) = 0;
LABEL_13:
  v31 = v26 & 0xFFFFFFFFFFFFFFF8LL;
  v32 = (__int64 *)sub_1DB3C70((__int64 *)v30, v31 | 4);
  if ( v32 == (__int64 *)(*(_QWORD *)v30 + 24LL * *(unsigned int *)(v30 + 8))
    || (*(_DWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v32 >> 1) & 3) > (*(_DWORD *)(v31 + 24) | 2u) )
  {
    v33 = 0;
  }
  else
  {
    v33 = v32[2];
  }
  v131.m128i_i32[0] = a3;
  v131.m128i_i64[1] = v33;
  v34 = sub_20F1860(v9 + 280, &v131);
  result = *(__int64 **)(v34 + 8);
  if ( *(__int64 **)(v34 + 16) != result )
    return sub_16CCBA0(v34, a2);
  v52 = &result[*(unsigned int *)(v34 + 28)];
  v53 = *(_DWORD *)(v34 + 28);
  if ( result == v52 )
  {
LABEL_51:
    if ( v53 < *(_DWORD *)(v34 + 24) )
    {
      *(_DWORD *)(v34 + 28) = v53 + 1;
      *v52 = a2;
      ++*(_QWORD *)v34;
      return (__int64 *)a2;
    }
    return sub_16CCBA0(v34, a2);
  }
  v54 = 0;
  while ( a2 != *result )
  {
    if ( *result == -2 )
      v54 = result;
    if ( v52 == ++result )
    {
      if ( !v54 )
        goto LABEL_51;
      *v54 = a2;
      --*(_DWORD *)(v34 + 32);
      ++*(_QWORD *)v34;
      return (__int64 *)a2;
    }
  }
  return result;
}
