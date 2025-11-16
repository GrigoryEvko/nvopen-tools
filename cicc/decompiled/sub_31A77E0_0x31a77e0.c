// Function: sub_31A77E0
// Address: 0x31a77e0
//
__int64 __fastcall sub_31A77E0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r9
  char *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rdx
  char v16; // r9
  int v17; // r10d
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rcx
  const __m128i *v26; // r14
  unsigned __int64 v27; // rsi
  __m128i *v28; // r13
  __int64 v29; // r14
  unsigned __int64 *v30; // r13
  unsigned __int64 v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 *v34; // rbx
  __int64 *v35; // r13
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rsi
  _QWORD *v39; // rax
  __int64 v41; // r13
  __int64 v42; // rcx
  __int64 v43; // rbx
  char *v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // r9
  char *v49; // r15
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // r12
  __int64 v54; // rsi
  __int64 v55; // r13
  __int64 v56; // r12
  __int64 v57; // rsi
  __int64 v58; // r13
  __int64 v59; // r12
  __int64 v60; // rsi
  __int64 v61; // r13
  __int64 v62; // r12
  __int64 v63; // rsi
  __int64 v64; // r13
  char *v65; // r13
  char *v66; // rax
  __int64 v67; // r15
  char *v68; // r14
  __int64 v69; // rax
  __int64 v70; // r12
  __int64 v71; // rsi
  _BYTE *v72; // rax
  signed __int64 v73; // r12
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // r14
  __int64 v78; // r13
  __int64 v79; // rsi
  __int64 v80; // r12
  __int64 v81; // r13
  __int64 v82; // rsi
  __int64 v83; // r12
  __int64 v84; // r13
  __int64 v85; // rsi
  __int64 v86; // r12
  char *v87; // [rsp+18h] [rbp-218h]
  __int64 v88; // [rsp+20h] [rbp-210h]
  unsigned __int8 v89; // [rsp+2Fh] [rbp-201h]
  char *v90; // [rsp+30h] [rbp-200h]
  __int64 *v91; // [rsp+30h] [rbp-200h]
  char *src; // [rsp+38h] [rbp-1F8h]
  __int64 *v93; // [rsp+40h] [rbp-1F0h]
  __int64 *v94; // [rsp+40h] [rbp-1F0h]
  __int64 *v95; // [rsp+40h] [rbp-1F0h]
  __int64 *v96; // [rsp+40h] [rbp-1F0h]
  __int64 v97; // [rsp+40h] [rbp-1F0h]
  __int64 *v98; // [rsp+40h] [rbp-1F0h]
  __int64 *v99; // [rsp+40h] [rbp-1F0h]
  __int64 *v100; // [rsp+40h] [rbp-1F0h]
  __int64 *v101; // [rsp+48h] [rbp-1E8h]
  __int64 *v102; // [rsp+48h] [rbp-1E8h]
  void *dest; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v104; // [rsp+58h] [rbp-1D8h]
  _QWORD v105[7]; // [rsp+60h] [rbp-1D0h] BYREF
  char v106; // [rsp+98h] [rbp-198h]
  unsigned __int64 *v107; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v108; // [rsp+A8h] [rbp-188h]
  _BYTE v109[324]; // [rsp+B0h] [rbp-180h] BYREF
  int v110; // [rsp+1F4h] [rbp-3Ch]
  __int64 v111; // [rsp+1F8h] [rbp-38h]

  v2 = *(_QWORD *)a1;
  v3 = sub_D440B0(*(_QWORD *)(a1 + 48), *(_QWORD *)a1);
  *(_QWORD *)(a1 + 56) = v3;
  v5 = *(_QWORD *)(v3 + 112);
  if ( v5 )
  {
    v6 = **(_QWORD **)(a1 + 64);
    v101 = *(__int64 **)(a1 + 64);
    v7 = sub_B2BE50(v6);
    if ( sub_B6EA50(v7)
      || (v75 = sub_B2BE50(v6),
          v76 = sub_B6F970(v75),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v76 + 48LL))(v76)) )
    {
      v11 = sub_31A4B60(*(_QWORD *)(a1 + 416), v2, v8, v9, v4, v10);
      v12 = *(_QWORD *)(v5 + 24);
      v13 = *(_QWORD *)(v5 + 32);
      v14 = *(_QWORD *)(v5 + 16);
      v15 = *(_QWORD *)(v5 + 48);
      v16 = *(_BYTE *)(v5 + 12);
      v17 = *(_DWORD *)(v5 + 8);
      v105[5] = *(_QWORD *)(v5 + 56);
      v107 = (unsigned __int64 *)v109;
      v108 = 0x400000000LL;
      v105[0] = v14;
      v105[1] = v12;
      v105[2] = v13;
      v105[4] = v15;
      LODWORD(v104) = v17;
      BYTE4(v104) = v16;
      v105[3] = v11;
      v106 = 0;
      v109[320] = 0;
      v110 = -1;
      dest = &unk_49D9D08;
      v111 = *(_QWORD *)(v5 + 424);
      sub_B18290((__int64)&dest, "loop not vectorized: ", 0x15u);
      v20 = 80LL * *(unsigned int *)(v5 + 88);
      if ( v20 )
      {
        v21 = (unsigned int)v108;
        v22 = *(_QWORD *)(v5 + 80);
        v23 = 0xCCCCCCCCCCCCCCCDLL * (v20 >> 4);
        do
        {
          v24 = (unsigned int)v21;
          v25 = (__int64)v107;
          v26 = (const __m128i *)v22;
          v27 = (unsigned int)v21 + 1LL;
          if ( v27 > HIDWORD(v108) )
          {
            if ( (unsigned __int64)v107 > v22 || (unsigned __int64)&v107[10 * (unsigned int)v21] <= v22 )
            {
              v26 = (const __m128i *)v22;
              sub_11F02D0((__int64)&v107, v27, v21, (__int64)v107, v18, v19);
              v24 = (unsigned int)v108;
              v25 = (__int64)v107;
              LODWORD(v21) = v108;
            }
            else
            {
              v77 = v22 - (_QWORD)v107;
              sub_11F02D0((__int64)&v107, v27, v21, (__int64)v107, v18, v19);
              v25 = (__int64)v107;
              v24 = (unsigned int)v108;
              v26 = (const __m128i *)((char *)v107 + v77);
              LODWORD(v21) = v108;
            }
          }
          v28 = (__m128i *)(v25 + 80 * v24);
          if ( v28 )
          {
            v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
            sub_31A4020(v28->m128i_i64, v26->m128i_i64[0], v26->m128i_i64[0] + v26->m128i_i64[1]);
            v28[2].m128i_i64[0] = (__int64)v28[3].m128i_i64;
            sub_31A4020(v28[2].m128i_i64, (_BYTE *)v26[2].m128i_i64[0], v26[2].m128i_i64[0] + v26[2].m128i_i64[1]);
            v28[4] = _mm_loadu_si128(v26 + 4);
            LODWORD(v21) = v108;
          }
          v21 = (unsigned int)(v21 + 1);
          v22 += 80LL;
          LODWORD(v108) = v21;
          --v23;
        }
        while ( v23 );
      }
      dest = &unk_49D9DE8;
      sub_1049740(v101, (__int64)&dest);
      v29 = (__int64)v107;
      dest = &unk_49D9D40;
      v30 = &v107[10 * (unsigned int)v108];
      if ( v107 != v30 )
      {
        do
        {
          v30 -= 10;
          v31 = v30[4];
          if ( (unsigned __int64 *)v31 != v30 + 6 )
            j_j___libc_free_0(v31);
          if ( (unsigned __int64 *)*v30 != v30 + 2 )
            j_j___libc_free_0(*v30);
        }
        while ( (unsigned __int64 *)v29 != v30 );
        v30 = v107;
      }
      if ( v30 != (unsigned __int64 *)v109 )
        _libc_free((unsigned __int64)v30);
    }
    v3 = *(_QWORD *)(a1 + 56);
  }
  v32 = *(unsigned __int8 *)(v3 + 40);
  v89 = v32;
  if ( !(_BYTE)v32 )
    return sub_31A52A0(a1);
  if ( *(_BYTE *)(v3 + 43) )
  {
    sub_2AB8760(
      (__int64)"We don't allow storing to uniform addresses",
      43,
      "write to a loop invariant address could not be vectorized",
      0x39u,
      (__int64)"CantVectorizeStoreToLoopInvariantAddress",
      40,
      *(__int64 **)(a1 + 64),
      *(_QWORD *)a1,
      0);
    return 0;
  }
  v33 = *(unsigned int *)(v3 + 56);
  if ( !*(_DWORD *)(v3 + 56) )
  {
    v41 = *(_QWORD *)(a1 + 16);
    goto LABEL_74;
  }
  v34 = *(__int64 **)(v3 + 48);
  v35 = &v34[v33];
  do
  {
    v36 = *v34;
    if ( !(unsigned __int8)sub_31A6470(a1, *v34, v33, v32, v4) )
      goto LABEL_34;
    if ( sub_31A6C30(a1, *(_QWORD *)(v36 + 40)) )
    {
      sub_2AB8760(
        (__int64)"We don't allow storing to uniform addresses",
        43,
        "write of conditional recurring variant value to a loop invariant address could not be vectorized",
        0x60u,
        (__int64)"CantVectorizeStoreToLoopInvariantAddress",
        40,
        *(__int64 **)(a1 + 64),
        *(_QWORD *)a1,
        0);
      return 0;
    }
    v37 = *(_QWORD *)(v36 - 32);
    if ( *(_BYTE *)v37 <= 0x1Cu )
      goto LABEL_34;
    v32 = *(_QWORD *)a1;
    v38 = *(_QWORD *)(v37 + 40);
    if ( !*(_BYTE *)(*(_QWORD *)a1 + 84LL) )
    {
      if ( !sub_C8CA60(v32 + 56, v38) )
        goto LABEL_34;
      v32 = *(_QWORD *)a1;
LABEL_31:
      sub_2AB8760(
        (__int64)"Invariant address is calculated inside the loop",
        47,
        "write to a loop invariant address could not be vectorized",
        0x39u,
        (__int64)"CantVectorizeStoreToLoopInvariantAddress",
        40,
        *(__int64 **)(a1 + 64),
        v32,
        0);
      return 0;
    }
    v39 = *(_QWORD **)(v32 + 64);
    v33 = (__int64)&v39[*(unsigned int *)(v32 + 76)];
    if ( v39 != (_QWORD *)v33 )
    {
      while ( v38 != *v39 )
      {
        if ( (_QWORD *)v33 == ++v39 )
          goto LABEL_34;
      }
      goto LABEL_31;
    }
LABEL_34:
    ++v34;
  }
  while ( v35 != v34 );
  v3 = *(_QWORD *)(a1 + 56);
  v41 = *(_QWORD *)(a1 + 16);
  v42 = v3;
  if ( !*(_BYTE *)(v3 + 42) )
    goto LABEL_74;
  v43 = *(_QWORD *)(v41 + 112);
  dest = v105;
  v104 = 0x400000000LL;
  v44 = *(char **)(v3 + 48);
  v87 = &v44[8 * *(unsigned int *)(v3 + 56)];
  if ( v87 == v44 )
    goto LABEL_94;
  v102 = *(__int64 **)(v3 + 48);
  v88 = a1;
  while ( 2 )
  {
    v47 = *v102;
    if ( !(unsigned __int8)sub_31A6470(v88, *v102, (__int64)v44, v42, v4) )
    {
      v45 = (unsigned int)v104;
      v42 = HIDWORD(v104);
      v46 = (unsigned int)v104 + 1LL;
      if ( v46 > HIDWORD(v104) )
      {
        sub_C8D5F0((__int64)&dest, v105, v46, 8u, v4, v48);
        v45 = (unsigned int)v104;
      }
      v44 = (char *)dest;
      *((_QWORD *)dest + v45) = v47;
      LODWORD(v104) = v104 + 1;
      goto LABEL_41;
    }
    v49 = (char *)dest;
    v50 = 8LL * (unsigned int)v104;
    src = (char *)dest + v50;
    v51 = v50 >> 3;
    v52 = v50 >> 5;
    if ( !v52 )
    {
LABEL_96:
      if ( v51 != 2 )
      {
        if ( v51 != 3 )
        {
          if ( v51 != 1 )
          {
LABEL_99:
            v49 = src;
            goto LABEL_69;
          }
LABEL_108:
          v84 = *(_QWORD *)v49;
          if ( v47 == *(_QWORD *)v49
            || (v85 = *(_QWORD *)(v47 - 32), v86 = *(_QWORD *)(v84 - 32), v85 == v86)
            || (v100 = sub_DD8400(v43, v85), v100 == sub_DD8400(v43, v86)) )
          {
            if ( *(_QWORD *)(*(_QWORD *)(v84 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
              goto LABEL_59;
          }
          goto LABEL_99;
        }
        v78 = *(_QWORD *)v49;
        if ( v47 == *(_QWORD *)v49
          || (v79 = *(_QWORD *)(v47 - 32), v80 = *(_QWORD *)(v78 - 32), v79 == v80)
          || (v98 = sub_DD8400(v43, v79), v98 == sub_DD8400(v43, v80)) )
        {
          if ( *(_QWORD *)(*(_QWORD *)(v78 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
            goto LABEL_59;
        }
        v49 += 8;
      }
      v81 = *(_QWORD *)v49;
      if ( v47 == *(_QWORD *)v49
        || (v82 = *(_QWORD *)(v47 - 32), v83 = *(_QWORD *)(v81 - 32), v82 == v83)
        || (v99 = sub_DD8400(v43, v82), v99 == sub_DD8400(v43, v83)) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v81 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
          goto LABEL_59;
      }
      v49 += 8;
      goto LABEL_108;
    }
    v90 = (char *)dest + 32 * v52;
    while ( 1 )
    {
      v62 = *(_QWORD *)v49;
      if ( v47 == *(_QWORD *)v49
        || (v63 = *(_QWORD *)(v47 - 32), v64 = *(_QWORD *)(v62 - 32), v63 == v64)
        || (v93 = sub_DD8400(v43, v63), v93 == sub_DD8400(v43, v64)) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v62 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
          break;
      }
      v53 = *((_QWORD *)v49 + 1);
      if ( v47 == v53
        || (v54 = *(_QWORD *)(v47 - 32), v55 = *(_QWORD *)(v53 - 32), v54 == v55)
        || (v94 = sub_DD8400(v43, v54), v94 == sub_DD8400(v43, v55)) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v53 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
        {
          v49 += 8;
          break;
        }
      }
      v56 = *((_QWORD *)v49 + 2);
      if ( v47 == v56
        || (v57 = *(_QWORD *)(v47 - 32), v58 = *(_QWORD *)(v56 - 32), v57 == v58)
        || (v95 = sub_DD8400(v43, v57), v95 == sub_DD8400(v43, v58)) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v56 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
        {
          v49 += 16;
          break;
        }
      }
      v59 = *((_QWORD *)v49 + 3);
      if ( v47 == v59
        || (v60 = *(_QWORD *)(v47 - 32), v61 = *(_QWORD *)(v59 - 32), v60 == v61)
        || (v96 = sub_DD8400(v43, v60), v96 == sub_DD8400(v43, v61)) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v59 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL) )
        {
          v49 += 24;
          break;
        }
      }
      v49 += 32;
      if ( v49 == v90 )
      {
        v51 = (src - v49) >> 3;
        goto LABEL_96;
      }
    }
LABEL_59:
    if ( src != v49 )
    {
      v65 = v49 + 8;
      if ( src != v49 + 8 )
      {
        v66 = v49;
        v67 = v47;
        v68 = v66;
        do
        {
          while ( 1 )
          {
            v70 = *(_QWORD *)v65;
            if ( v67 == *(_QWORD *)v65
              || (v4 = *(_QWORD *)(v70 - 32), v71 = *(_QWORD *)(v67 - 32), v97 = v4, v71 == v4)
              || (v91 = sub_DD8400(v43, v71), v91 == sub_DD8400(v43, v97)) )
            {
              if ( *(_QWORD *)(*(_QWORD *)(v70 - 64) + 8LL) == *(_QWORD *)(*(_QWORD *)(v67 - 64) + 8LL) )
                break;
            }
            v69 = *(_QWORD *)v65;
            v68 += 8;
            v65 += 8;
            *((_QWORD *)v68 - 1) = v69;
            if ( src == v65 )
              goto LABEL_68;
          }
          v65 += 8;
        }
        while ( src != v65 );
LABEL_68:
        v49 = v68;
      }
    }
LABEL_69:
    v72 = dest;
    v44 = (char *)dest + 8 * (unsigned int)v104;
    v73 = v44 - src;
    if ( src != v44 )
    {
      memmove(v49, src, (_BYTE *)dest + 8 * (unsigned int)v104 - src);
      v72 = dest;
    }
    v42 = (&v49[v73] - v72) >> 3;
    LODWORD(v104) = v42;
LABEL_41:
    if ( v87 != (char *)++v102 )
      continue;
    break;
  }
  if ( (_DWORD)v104 )
  {
    sub_2AB8760(
      (__int64)"We don't allow storing to uniform addresses",
      43,
      "write to a loop invariant address could not be vectorized",
      0x39u,
      (__int64)"CantVectorizeStoreToLoopInvariantAddress",
      40,
      *(__int64 **)(v88 + 64),
      *(_QWORD *)v88,
      0);
    if ( dest != v105 )
      _libc_free((unsigned __int64)dest);
    return 0;
  }
  if ( dest != v105 )
    _libc_free((unsigned __int64)dest);
  v42 = *(_QWORD *)(v88 + 56);
  v41 = *(_QWORD *)(v88 + 16);
LABEL_94:
  v3 = v42;
LABEL_74:
  v74 = sub_D9B120(*(_QWORD *)v3);
  sub_DEF380(v41, v74);
  return v89;
}
