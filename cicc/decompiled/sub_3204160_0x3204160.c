// Function: sub_3204160
// Address: 0x3204160
//
__int64 __fastcall sub_3204160(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 *v5; // rbx
  unsigned int v6; // eax
  __int16 v7; // r14
  unsigned int v8; // ecx
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  unsigned int v11; // eax
  int v12; // r12d
  __int16 v13; // ax
  int v14; // r12d
  __int64 v15; // r12
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int32 v18; // eax
  int v19; // r12d
  __int32 v20; // r14d
  __int16 v21; // ax
  int v22; // r12d
  __int64 *v23; // r12
  __int64 v24; // rdx
  unsigned __int8 v25; // al
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // ebx
  __int16 v30; // ax
  int v31; // ebx
  int v32; // eax
  unsigned __int64 v33; // r15
  __int64 v34; // r14
  unsigned __int8 v35; // al
  __int64 v36; // r15
  unsigned __int8 v37; // dl
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  __int8 v43; // dl
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rbx
  __int64 v49; // rcx
  __int64 v50; // rax
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  unsigned __int64 v55; // rax
  int v56; // esi
  int v57; // r14d
  __int16 v58; // ax
  __int16 v59; // r14
  __m128i *v60; // rsi
  _DWORD *v61; // r12
  __int32 v62; // eax
  int v63; // r14d
  int v64; // edx
  int v65; // r8d
  __int16 v66; // cx
  int v67; // eax
  int v68; // r14d
  _DWORD *v69; // rax
  __m128i *v70; // r14
  const __m128i *v71; // r12
  __int8 *v72; // rbx
  __m128i *v73; // rax
  __m128i *v74; // rdx
  __int64 *v75; // rbx
  __int64 v76; // r15
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // r12
  __int32 v80; // eax
  unsigned __int8 v81; // al
  int v82; // eax
  bool v83; // zf
  int v84; // edx
  __int64 *v85; // rbx
  __int64 *v86; // r13
  __int64 v87; // rax
  unsigned __int64 *v88; // rax
  unsigned __int64 v89; // r12
  unsigned __int8 v91; // al
  __int64 v92; // r15
  __int32 v93; // eax
  __int64 *v95; // [rsp+10h] [rbp-270h]
  __int64 *v96; // [rsp+28h] [rbp-258h]
  __int64 v97; // [rsp+30h] [rbp-250h]
  __int64 v98; // [rsp+30h] [rbp-250h]
  unsigned int v99; // [rsp+38h] [rbp-248h]
  __int64 *v100; // [rsp+38h] [rbp-248h]
  __int64 v101; // [rsp+38h] [rbp-248h]
  __int64 v102; // [rsp+40h] [rbp-240h]
  unsigned int v103; // [rsp+40h] [rbp-240h]
  __int64 v104; // [rsp+40h] [rbp-240h]
  __int64 v105; // [rsp+40h] [rbp-240h]
  int i; // [rsp+4Ch] [rbp-234h]
  __int64 v107; // [rsp+50h] [rbp-230h]
  __int64 v109; // [rsp+58h] [rbp-228h]
  __int64 v110; // [rsp+58h] [rbp-228h]
  __int64 v111; // [rsp+58h] [rbp-228h]
  __int32 v112; // [rsp+58h] [rbp-228h]
  __int64 *v113; // [rsp+60h] [rbp-220h]
  __int32 v114; // [rsp+60h] [rbp-220h]
  __int16 v115; // [rsp+60h] [rbp-220h]
  int v116; // [rsp+60h] [rbp-220h]
  __int64 *j; // [rsp+60h] [rbp-220h]
  __int32 v118; // [rsp+68h] [rbp-218h]
  unsigned __int64 v119; // [rsp+70h] [rbp-210h] BYREF
  __m128i *v120; // [rsp+78h] [rbp-208h]
  const __m128i *v121; // [rsp+80h] [rbp-200h]
  _WORD v122[2]; // [rsp+90h] [rbp-1F0h] BYREF
  int v123; // [rsp+94h] [rbp-1ECh]
  __int64 v124; // [rsp+98h] [rbp-1E8h]
  __int64 v125; // [rsp+A0h] [rbp-1E0h]
  __m128i v126; // [rsp+B0h] [rbp-1D0h] BYREF
  __m128i v127; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 *v128; // [rsp+D0h] [rbp-1B0h] BYREF
  __int64 *v129; // [rsp+D8h] [rbp-1A8h]
  __int64 *v130; // [rsp+E8h] [rbp-198h]
  __int64 *v131; // [rsp+F0h] [rbp-190h]
  __int64 v132; // [rsp+108h] [rbp-178h]
  unsigned int v133; // [rsp+118h] [rbp-168h]
  __int64 *v134; // [rsp+120h] [rbp-160h]
  unsigned int v135; // [rsp+128h] [rbp-158h]
  int v136; // [rsp+130h] [rbp-150h] BYREF
  __int64 *v137; // [rsp+138h] [rbp-148h]
  __int64 *v138; // [rsp+140h] [rbp-140h]
  _BYTE v139[304]; // [rsp+150h] [rbp-130h] BYREF

  v3 = a3;
  v4 = a2;
  sub_3203600((__int64)&v128, a2, a3);
  sub_3702BF0(v139);
  sub_3702E30(v139, 0);
  v5 = v128;
  v113 = v129;
  if ( v129 == v128 )
  {
    i = 0;
  }
  else
  {
    for ( i = 1; ; ++i )
    {
      v15 = *v5;
      if ( (*(_BYTE *)(*v5 + 20) & 0x20) != 0 )
      {
        v102 = *v5 - 16;
        v6 = sub_AF2D10(*v5);
        v107 = (unsigned int)(*(_QWORD *)(v15 + 32) >> 2);
        v7 = ((*(_DWORD *)(v15 + 20) & 0x24) == 36) + 5121;
        v109 = v6;
        v8 = sub_31F8540(a2);
        v9 = *(_BYTE *)(v15 - 16);
        if ( (v9 & 2) != 0 )
          v10 = *(_QWORD *)(v15 - 32);
        else
          v10 = v102 - 8LL * ((v9 >> 2) & 0xF);
        v99 = v8;
        v11 = sub_3206530(a2, *(_QWORD *)(v10 + 24), 0);
        v12 = *(_DWORD *)(v15 + 20);
        v103 = v11;
        v13 = sub_AF18C0(v3);
        v14 = v12 & 3;
        if ( (unsigned int)(v14 - 1) > 2 )
          LOWORD(v14) = 2 * (v13 != 2) + 1;
        v126.m128i_i16[0] = v7;
        v126.m128i_i16[1] = v14;
        v127.m128i_i64[0] = v109;
        *(__int64 *)((char *)v126.m128i_i64 + 4) = __PAIR64__(v99, v103);
        v127.m128i_i64[1] = v107;
        sub_3703480(v139, &v126);
      }
      else
      {
        v110 = *(_QWORD *)(v15 + 32) >> 3;
        v16 = *(_BYTE *)(v15 - 16);
        if ( (v16 & 2) != 0 )
          v17 = *(_QWORD *)(v15 - 32);
        else
          v17 = *v5 - 16 - 8LL * ((v16 >> 2) & 0xF);
        v18 = sub_3206530(a2, *(_QWORD *)(v17 + 24), 0);
        v19 = *(_DWORD *)(v15 + 20);
        v20 = v18;
        v21 = sub_AF18C0(v3);
        v22 = v19 & 3;
        if ( (unsigned int)(v22 - 1) > 2 )
          LOWORD(v22) = 2 * (v21 != 2) + 1;
        v126.m128i_i16[1] = v22;
        v126.m128i_i16[0] = 5120;
        v126.m128i_i32[1] = v20;
        v126.m128i_i64[1] = v110;
        sub_3703350(v139, &v126);
      }
      if ( v113 == ++v5 )
        break;
    }
  }
  v100 = v131;
  if ( v130 != v131 )
  {
    v23 = v130;
    v97 = a2 + 632;
    v104 = v3;
    while ( 1 )
    {
      v34 = *v23;
      v35 = *(_BYTE *)(*v23 - 16);
      v36 = *v23 - 16;
      if ( (v35 & 2) != 0 )
        v24 = *(_QWORD *)(v34 - 32);
      else
        v24 = v36 - 8LL * ((v35 >> 2) & 0xF);
      v114 = sub_3206530(a2, *(_QWORD *)(v24 + 24), 0);
      v25 = *(_BYTE *)(v34 - 16);
      if ( (v25 & 2) != 0 )
      {
        v26 = *(_QWORD *)(*(_QWORD *)(v34 - 32) + 16LL);
        if ( !v26 )
          goto LABEL_35;
      }
      else
      {
        v26 = *(_QWORD *)(v36 - 8LL * ((v25 >> 2) & 0xF) + 16);
        if ( !v26 )
        {
LABEL_35:
          v111 = 0;
          goto LABEL_25;
        }
      }
      v27 = sub_B91420(v26);
      v111 = v28;
      v26 = v27;
LABEL_25:
      v29 = *(_DWORD *)(v34 + 20);
      v30 = sub_AF18C0(v104);
      v31 = v29 & 3;
      if ( (unsigned int)(v31 - 1) > 2 )
        LOBYTE(v31) = 2 * (v30 != 2) + 1;
      v32 = *(_DWORD *)(v34 + 20);
      ++i;
      if ( (v32 & 0x1000) != 0 )
      {
        v23 += 2;
        v126.m128i_i16[1] = (unsigned __int8)v31;
        v126.m128i_i32[1] = v114;
        v126.m128i_i16[0] = 5390;
        v126.m128i_i64[1] = v26;
        v127.m128i_i64[0] = v111;
        sub_37036E0(v139, &v126);
        if ( v100 == v23 )
        {
LABEL_49:
          v4 = a2;
          v3 = v104;
          break;
        }
      }
      else
      {
        if ( (v32 & 0x40) != 0
          && ((v37 = *(_BYTE *)(v34 - 16), (v37 & 2) == 0)
            ? (v38 = v36 - 8LL * ((v37 >> 2) & 0xF))
            : (v38 = *(_QWORD *)(v34 - 32)),
              (v39 = *(_QWORD *)(v38 + 16)) != 0) )
        {
          v40 = sub_B91420(v39);
          if ( v41 > 5 && *(_DWORD *)v40 == 1953527391 && *(_WORD *)(v40 + 4) == 9330 )
          {
            v91 = *(_BYTE *)(v34 - 16);
            if ( (v91 & 2) != 0 )
              v92 = *(_QWORD *)(v34 - 32);
            else
              v92 = v36 - 8LL * ((v91 >> 2) & 0xF);
            v93 = sub_3206530(a2, *(_QWORD *)(v92 + 24), 0);
            v126.m128i_i16[0] = 5129;
            *(__int32 *)((char *)v126.m128i_i32 + 2) = v93;
            sub_37035B0(v139, &v126);
            goto LABEL_31;
          }
          v33 = v23[1] + *(_QWORD *)(v34 + 32);
          if ( (*(_DWORD *)(v34 + 20) & 0x80000) != 0 )
          {
LABEL_42:
            v42 = sub_AF2D80(v34);
            v43 = 0;
            if ( v42 && *(_BYTE *)v42 == 17 )
            {
              if ( *(_DWORD *)(v42 + 32) <= 0x40u )
                v44 = *(_QWORD *)(v42 + 24);
              else
                v44 = **(_QWORD **)(v42 + 24);
              v45 = v23[1] + v44;
              v43 = v33 - v45;
              v33 = v45;
            }
            v46 = *(_QWORD *)(v34 + 24);
            v126.m128i_i8[7] = v43;
            v126.m128i_i16[0] = 4613;
            v126.m128i_i8[6] = v46;
            *(__int32 *)((char *)v126.m128i_i32 + 2) = v114;
            v47 = sub_370A950(a2 + 648, &v126);
            v114 = sub_3707F80(v97, v47);
          }
        }
        else
        {
          v33 = v23[1] + *(_QWORD *)(v34 + 32);
          if ( (v32 & 0x80000) != 0 )
            goto LABEL_42;
        }
        v126.m128i_i16[1] = (unsigned __int8)v31;
        v126.m128i_i32[1] = v114;
        v126.m128i_i16[0] = 5389;
        v126.m128i_i64[1] = v33 >> 3;
        v127.m128i_i64[0] = v26;
        v127.m128i_i64[1] = v111;
        sub_3703940(v139, &v126);
LABEL_31:
        v23 += 2;
        if ( v100 == v23 )
          goto LABEL_49;
      }
    }
  }
  v95 = &v134[2 * v135];
  if ( v95 != v134 )
  {
    v96 = v134;
    while ( 1 )
    {
      v52 = sub_B91420(*v96);
      v119 = 0;
      v105 = v52;
      v120 = 0;
      v121 = 0;
      v53 = v96[1];
      v101 = v54;
      if ( (v53 & 4) != 0 )
      {
        v55 = v53 & 0xFFFFFFFFFFFFFFF8LL;
        v48 = *(_QWORD **)v55;
        v98 = *(_QWORD *)v55 + 8LL * *(unsigned int *)(v55 + 8);
      }
      else
      {
        v48 = v96 + 1;
        if ( (v53 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          break;
        v98 = (__int64)(v96 + 2);
      }
      if ( (_QWORD *)v98 == v48 )
        break;
      do
      {
        v61 = (_DWORD *)*v48;
        v62 = sub_3207400(v4, *v48, v3);
        v63 = v61[8];
        v64 = -1;
        v118 = v62;
        v65 = v63 & 0x40000;
        if ( (v63 & 0x40000) != 0 )
        {
          v68 = v61[6];
          v116 = v65;
          v69 = sub_AE2980(*(_QWORD *)(*(_QWORD *)(v4 + 16) + 2488LL) + 312LL, 0);
          v65 = v116;
          v64 = v68 * (v69[1] >> 3);
          v63 = v61[8];
        }
        v66 = (4 * v63) & 0x100;
        if ( (v63 & 0x1000) != 0 )
        {
          LOWORD(v56) = 8;
        }
        else
        {
          v67 = v61[9] & 3;
          if ( v67 == 1 )
          {
            v56 = v65 == 0 ? 4 : 16;
          }
          else if ( v67 == 2 )
          {
            v56 = v65 == 0 ? 20 : 24;
          }
          else
          {
            if ( v67 )
              BUG();
            LOWORD(v56) = 0;
          }
        }
        v57 = v63 & 3;
        v112 = v64;
        v115 = v66;
        v58 = sub_AF18C0(v3);
        if ( (unsigned int)(v57 - 1) > 2 )
          LOWORD(v57) = 2 * (v58 != 2) + 1;
        v126.m128i_i32[2] = v112;
        v59 = v115 | v56 | v57;
        v126.m128i_i16[0] = 5393;
        v60 = v120;
        *(__int32 *)((char *)v126.m128i_i32 + 2) = v118;
        v126.m128i_i16[3] = v59;
        v127.m128i_i64[0] = v105;
        v127.m128i_i64[1] = v101;
        if ( v120 == v121 )
        {
          sub_31FCA30(&v119, v120, &v126);
        }
        else
        {
          if ( v120 )
          {
            *v120 = _mm_loadu_si128(&v126);
            v60[1] = _mm_loadu_si128(&v127);
            v60 = v120;
          }
          v120 = v60 + 2;
        }
        ++i;
        ++v48;
      }
      while ( (_QWORD *)v98 != v48 );
      v70 = v120;
      v71 = (const __m128i *)v119;
      v72 = &v120->m128i_i8[-v119];
      if ( (__m128i *)((char *)v120 - v119) != (__m128i *)32 )
      {
        v126 = (__m128i)0x1206uLL;
        v127 = 0u;
        if ( (unsigned __int64)v72 > 0x7FFFFFFFFFFFFFE0LL )
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        if ( v72 )
        {
          v73 = (__m128i *)sub_22077B0((unsigned __int64)v120 - v119);
          v74 = (__m128i *)&v72[(_QWORD)v73];
          v126.m128i_i64[1] = (__int64)v73;
          v127.m128i_i64[1] = (__int64)&v72[(_QWORD)v73];
          if ( v70 == v71 )
          {
            v49 = (__int64)v73;
          }
          else
          {
            v49 = (__int64)&v72[(_QWORD)v73];
            do
            {
              if ( v73 )
              {
                *v73 = _mm_loadu_si128(v71);
                v73[1] = _mm_loadu_si128(v71 + 1);
              }
              v73 += 2;
              v71 += 2;
            }
            while ( v73 != v74 );
          }
          goto LABEL_55;
        }
LABEL_54:
        v127.m128i_i64[1] = 0;
        v49 = 0;
LABEL_55:
        v127.m128i_i64[0] = v49;
        v50 = sub_370B8B0(v4 + 648, &v126);
        v51 = sub_3707F80(v4 + 632, v50);
        v122[0] = 5391;
        v123 = v51;
        v122[1] = (__int64)((__int64)v120->m128i_i64 - v119) >> 5;
        v124 = v105;
        v125 = v101;
        sub_3703810(v139, v122);
        if ( v126.m128i_i64[1] )
          j_j___libc_free_0(v126.m128i_u64[1]);
        goto LABEL_57;
      }
      sub_3703BA0(v139, v119);
LABEL_57:
      if ( v119 )
        j_j___libc_free_0(v119);
      v96 += 2;
      if ( v95 == v96 )
        goto LABEL_95;
    }
    v126.m128i_i64[1] = 0;
    v126.m128i_i16[0] = 4614;
    goto LABEL_54;
  }
LABEL_95:
  v75 = v137;
  for ( j = v138; j != v75; ++i )
  {
    v77 = *v75;
    v81 = *(_BYTE *)(*v75 - 16);
    if ( (v81 & 2) != 0 )
    {
      v76 = *(_QWORD *)(*(_QWORD *)(v77 - 32) + 16LL);
      if ( !v76 )
        goto LABEL_102;
    }
    else
    {
      v76 = *(_QWORD *)(v77 - 8LL * ((v81 >> 2) & 0xF));
      if ( !v76 )
      {
LABEL_102:
        v79 = 0;
        goto LABEL_99;
      }
    }
    v77 = *v75;
    v76 = sub_B91420(v76);
    v79 = v78;
LABEL_99:
    ++v75;
    v80 = sub_3206530(v4, v77, 0);
    v126.m128i_i16[0] = 5392;
    *(__int32 *)((char *)v126.m128i_i32 + 2) = v80;
    v126.m128i_i64[1] = v76;
    v127.m128i_i64[0] = v79;
    sub_3703A70(v139, &v126);
  }
  v82 = sub_37083C0(v4 + 632, v139);
  v83 = v138 == v137;
  v84 = v136;
  *(_DWORD *)(a1 + 4) = i;
  *(_DWORD *)(a1 + 8) = v84;
  *(_DWORD *)(a1 + 12) = v82;
  *(_BYTE *)a1 = !v83;
  sub_3702CE0(v139);
  if ( v137 )
    j_j___libc_free_0((unsigned __int64)v137);
  v85 = v134;
  v86 = &v134[2 * v135];
  if ( v134 != v86 )
  {
    do
    {
      v87 = *(v86 - 1);
      v86 -= 2;
      if ( v87 )
      {
        if ( (v87 & 4) != 0 )
        {
          v88 = (unsigned __int64 *)(v87 & 0xFFFFFFFFFFFFFFF8LL);
          v89 = (unsigned __int64)v88;
          if ( v88 )
          {
            if ( (unsigned __int64 *)*v88 != v88 + 2 )
              _libc_free(*v88);
            j_j___libc_free_0(v89);
          }
        }
      }
    }
    while ( v85 != v86 );
    v86 = v134;
  }
  if ( v86 != (__int64 *)&v136 )
    _libc_free((unsigned __int64)v86);
  sub_C7D6A0(v132, 16LL * v133, 8);
  if ( v130 )
    j_j___libc_free_0((unsigned __int64)v130);
  if ( v128 )
    j_j___libc_free_0((unsigned __int64)v128);
  return a1;
}
