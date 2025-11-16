// Function: sub_3999410
// Address: 0x3999410
//
__int64 __fastcall sub_3999410(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r13
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  const char *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 *v25; // r15
  __int64 v26; // rsi
  __int64 *v27; // r14
  __int64 v28; // rax
  __int64 (*v29)(); // rdx
  __int64 v30; // r15
  char v31; // al
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  _BYTE *v35; // r14
  __int64 v36; // rax
  __m128i *v37; // rdx
  __m128i *v38; // r15
  __int64 v39; // rdi
  __m128i *v40; // rdx
  _BYTE *v41; // r10
  __m128i *v42; // r9
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v49; // r8
  unsigned int v50; // esi
  __int64 v51; // r8
  unsigned int v52; // edi
  __int64 *v53; // rax
  __int64 v54; // rcx
  int v55; // r11d
  __int64 *v56; // rdx
  int v57; // eax
  int v58; // ecx
  __int64 v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // r14
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // r8
  _QWORD *v65; // rax
  unsigned __int64 *v66; // rax
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // rdi
  __m128i *v69; // rax
  __m128i *v70; // rcx
  __m128i *v71; // rdx
  _QWORD *v72; // rdi
  int v73; // r9d
  _QWORD *v74; // rdi
  __int64 v75; // rax
  __m128i v76; // rax
  int v77; // eax
  int v78; // edi
  __int64 v79; // r8
  unsigned int v80; // eax
  __int64 v81; // rsi
  int v82; // r10d
  __int64 *v83; // r9
  int v84; // eax
  int v85; // esi
  __int64 v86; // r8
  __int64 *v87; // rdi
  unsigned int v88; // r12d
  int v89; // r9d
  __int64 v90; // rax
  _BYTE *v91; // [rsp+8h] [rbp-E8h]
  __m128i *v92; // [rsp+10h] [rbp-E0h]
  void *dest; // [rsp+18h] [rbp-D8h]
  __int64 v94; // [rsp+20h] [rbp-D0h]
  const char *v95; // [rsp+28h] [rbp-C8h]
  __int64 v96; // [rsp+38h] [rbp-B8h]
  __int64 v97; // [rsp+38h] [rbp-B8h]
  __m128i *v98; // [rsp+40h] [rbp-B0h]
  __int64 v99; // [rsp+48h] [rbp-A8h]
  __m128i v100; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v101; // [rsp+60h] [rbp-90h] BYREF
  __m128i *v102; // [rsp+68h] [rbp-88h]
  _QWORD v103[2]; // [rsp+70h] [rbp-80h] BYREF
  __m128i *v104; // [rsp+80h] [rbp-70h] BYREF
  size_t v105; // [rsp+88h] [rbp-68h]
  __m128i v106; // [rsp+90h] [rbp-60h] BYREF
  __m128i v107; // [rsp+A0h] [rbp-50h] BYREF
  _QWORD v108[8]; // [rsp+B0h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 544);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 528);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
      {
        v9 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL * *((unsigned int *)v7 + 2) + 8);
        if ( v9 )
          return v9;
      }
    }
    else
    {
      v11 = 1;
      while ( v8 != -8 )
      {
        v73 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v11 = v73;
      }
    }
  }
  v12 = *(unsigned int *)(a2 + 8);
  if ( *(_BYTE *)a2 == 15 )
  {
    v14 = *(_QWORD *)(a2 - 8 * v12);
    if ( !v14 )
    {
      v95 = 0;
      v17 = a2;
      v94 = 0;
      goto LABEL_14;
    }
  }
  else
  {
    v13 = *(_QWORD *)(a2 - 8 * v12);
    if ( !v13 )
    {
      v94 = 0;
      v95 = byte_3F871B3;
      goto LABEL_12;
    }
    v14 = *(_QWORD *)(v13 - 8LL * *(unsigned int *)(v13 + 8));
    if ( !v14 )
    {
      v95 = 0;
      v94 = 0;
      goto LABEL_12;
    }
  }
  v15 = sub_161E970(v14);
  v12 = *(unsigned int *)(a2 + 8);
  v95 = (const char *)v15;
  v94 = v16;
  if ( *(_BYTE *)a2 == 15 )
  {
    v17 = a2;
    goto LABEL_14;
  }
LABEL_12:
  v17 = *(_QWORD *)(a2 - 8 * v12);
  if ( !v17 )
  {
    v19 = 0;
    v18 = byte_3F871B3;
    goto LABEL_16;
  }
  v12 = *(unsigned int *)(v17 + 8);
LABEL_14:
  v18 = *(const char **)(v17 + 8 * (1 - v12));
  if ( v18 )
    v18 = (const char *)sub_161E970((__int64)v18);
  else
    v19 = 0;
LABEL_16:
  *(_QWORD *)(a1 + 4024) = v18;
  v20 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 4032) = v19;
  v21 = *(_DWORD *)(a1 + 4216);
  v96 = v20;
  v22 = sub_22077B0(0x3A8u);
  v9 = v22;
  if ( v22 )
    sub_39C7990(v22, v21, a2, v96, a1, a1 + 4040);
  v107.m128i_i64[0] = v9;
  v97 = v9 + 8;
  sub_39A0610(a1 + 4040, &v107);
  if ( v107.m128i_i64[0] )
    sub_3985790(v107.m128i_u64[0]);
  if ( *(_BYTE *)(a1 + 4513) )
  {
    *(_QWORD *)(v9 + 616) = sub_398B900(a1, v9);
    sub_39A3F30(
      v9,
      v97,
      8496,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 880LL),
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 888LL));
  }
  v23 = *(_QWORD *)(a2 + 8 * (7LL - *(unsigned int *)(a2 + 8)));
  if ( v23 )
  {
    v24 = 8LL * *(unsigned int *)(v23 + 8);
    v25 = (__int64 *)(v23 - v24);
    if ( v23 - v24 != v23 )
    {
      do
      {
        v26 = *v25++;
        sub_3992040(v9, v26);
      }
      while ( (__int64 *)v23 != v25 );
    }
  }
  if ( (unsigned __int16)sub_398C0A0(a1) > 4u )
  {
    v27 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 256LL);
    v28 = *v27;
    v29 = *(__int64 (**)())(*v27 + 88);
    if ( v29 != sub_168DB60 )
    {
      if ( ((unsigned __int8 (__fastcall *)(_QWORD))v29)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL))
        && !*(_BYTE *)(a1 + 5408) )
      {
        goto LABEL_34;
      }
      v27 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 256LL);
      v28 = *v27;
    }
    v30 = *(unsigned int *)(v9 + 600);
    dest = *(void **)(v28 + 576);
    v31 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 == 15 )
    {
      v32 = a2;
    }
    else
    {
      v32 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      if ( !v32 )
      {
        LOBYTE(v108[0]) = 0;
LABEL_33:
        v33 = sub_39A3100(v9, v32);
        ((void (__fastcall *)(__int64 *, _QWORD, _QWORD, const char *, __int64, __int64, __m128i *, __int64))dest)(
          v27,
          *(_QWORD *)(a1 + 4024),
          *(_QWORD *)(a1 + 4032),
          v95,
          v94,
          v33,
          &v107,
          v30);
        goto LABEL_34;
      }
    }
    if ( *(_BYTE *)(v32 + 56) )
    {
      v76.m128i_i64[0] = sub_161E970(*(_QWORD *)(v32 + 48));
      LOBYTE(v108[0]) = 1;
      v107 = v76;
      v31 = *(_BYTE *)a2;
    }
    else
    {
      LOBYTE(v108[0]) = 0;
    }
    v32 = a2;
    if ( v31 != 15 )
      v32 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
    goto LABEL_33;
  }
LABEL_34:
  v34 = *(unsigned int *)(a2 + 8);
  v35 = *(_BYTE **)(a2 + 8 * (1 - v34));
  if ( v35 )
  {
    v36 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v34)));
    v38 = v37;
    v34 = *(unsigned int *)(a2 + 8);
    v35 = (_BYTE *)v36;
  }
  else
  {
    v38 = 0;
  }
  v39 = *(_QWORD *)(a2 + 8 * (2 - v34));
  if ( !v39 || (v41 = (_BYTE *)sub_161E970(v39), (v42 = v40) == 0) )
  {
    sub_39A3F30(v9, v97, 37, v35, v38);
    goto LABEL_46;
  }
  v107.m128i_i64[0] = (__int64)v108;
  if ( v41 )
  {
    v104 = v40;
    if ( (unsigned __int64)v40 > 0xF )
    {
      v92 = v40;
      v91 = v41;
      v75 = sub_22409D0((__int64)&v107, (unsigned __int64 *)&v104, 0);
      v42 = v92;
      v41 = v91;
      v107.m128i_i64[0] = v75;
      v72 = (_QWORD *)v75;
      v108[0] = v104;
    }
    else
    {
      if ( v40 == (__m128i *)1 )
      {
        LOBYTE(v108[0]) = *v41;
        v43 = (__int64)v108;
LABEL_42:
        v107.m128i_i64[1] = (__int64)v42;
        v42->m128i_i8[v43] = 0;
        goto LABEL_84;
      }
      v72 = v108;
    }
    memcpy(v72, v41, (size_t)v42);
    v42 = v104;
    v43 = v107.m128i_i64[0];
    goto LABEL_42;
  }
  v107.m128i_i64[1] = 0;
  LOBYTE(v108[0]) = 0;
LABEL_84:
  if ( !v35 )
  {
    LOBYTE(v103[0]) = 0;
    v101 = v103;
    v102 = 0;
    goto LABEL_91;
  }
  v104 = v38;
  v101 = v103;
  if ( (unsigned __int64)v38 > 0xF )
  {
    v101 = (_QWORD *)sub_22409D0((__int64)&v101, (unsigned __int64 *)&v104, 0);
    v74 = v101;
    v103[0] = v104;
  }
  else
  {
    if ( v38 == (__m128i *)1 )
    {
      LOBYTE(v103[0]) = *v35;
      v65 = v103;
      goto LABEL_88;
    }
    if ( !v38 )
    {
      v65 = v103;
      goto LABEL_88;
    }
    v74 = v103;
  }
  memcpy(v74, v35, (size_t)v38);
  v38 = v104;
  v65 = v101;
LABEL_88:
  v102 = v38;
  v38->m128i_i8[(_QWORD)v65] = 0;
  if ( v102 == (__m128i *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
LABEL_91:
  v66 = sub_2241490((unsigned __int64 *)&v101, " ", 1u);
  v104 = &v106;
  if ( (unsigned __int64 *)*v66 == v66 + 2 )
  {
    v106 = _mm_loadu_si128((const __m128i *)v66 + 1);
  }
  else
  {
    v104 = (__m128i *)*v66;
    v106.m128i_i64[0] = v66[2];
  }
  v105 = v66[1];
  *v66 = (unsigned __int64)(v66 + 2);
  v66[1] = 0;
  *((_BYTE *)v66 + 16) = 0;
  v67 = 15;
  v68 = 15;
  if ( v104 != &v106 )
    v68 = v106.m128i_i64[0];
  if ( v105 + v107.m128i_i64[1] > v68 )
  {
    if ( (_QWORD *)v107.m128i_i64[0] != v108 )
      v67 = v108[0];
    if ( v105 + v107.m128i_i64[1] <= v67 )
    {
      v69 = (__m128i *)sub_2241130((unsigned __int64 *)&v107, 0, 0, v104, v105);
      v98 = &v100;
      v70 = (__m128i *)v69->m128i_i64[0];
      v71 = v69 + 1;
      if ( (__m128i *)v69->m128i_i64[0] != &v69[1] )
        goto LABEL_100;
LABEL_120:
      v100 = _mm_loadu_si128(v69 + 1);
      goto LABEL_101;
    }
  }
  v69 = (__m128i *)sub_2241490((unsigned __int64 *)&v104, (char *)v107.m128i_i64[0], v107.m128i_u64[1]);
  v98 = &v100;
  v70 = (__m128i *)v69->m128i_i64[0];
  v71 = v69 + 1;
  if ( (__m128i *)v69->m128i_i64[0] == &v69[1] )
    goto LABEL_120;
LABEL_100:
  v98 = v70;
  v100.m128i_i64[0] = v69[1].m128i_i64[0];
LABEL_101:
  v99 = v69->m128i_i64[1];
  v69->m128i_i64[0] = (__int64)v71;
  v69->m128i_i64[1] = 0;
  v69[1].m128i_i8[0] = 0;
  if ( v104 != &v106 )
    j_j___libc_free_0((unsigned __int64)v104);
  if ( v101 != v103 )
    j_j___libc_free_0((unsigned __int64)v101);
  if ( (_QWORD *)v107.m128i_i64[0] != v108 )
    j_j___libc_free_0(v107.m128i_u64[0]);
  sub_39A3F30(v9, v97, 37, v98, v99);
  if ( v98 != &v100 )
    j_j___libc_free_0((unsigned __int64)v98);
LABEL_46:
  v44 = *(unsigned int *)(a2 + 24);
  v107.m128i_i32[0] = 65541;
  sub_39A3560(v9, v9 + 16, 19, &v107, v44);
  sub_39A3F30(v9, v97, 3, v95, v94);
  if ( *(_BYTE *)(a1 + 4514) )
  {
    if ( *(_BYTE *)(a1 + 4513) )
      goto LABEL_48;
    sub_39A3E90(v9);
  }
  if ( *(_BYTE *)(a1 + 4513) )
  {
LABEL_48:
    if ( !*(_BYTE *)(a1 + 4512) )
    {
LABEL_76:
      *(_QWORD *)(v9 + 56) = *(_QWORD *)(sub_396DD80(*(_QWORD *)(a1 + 8)) + 224);
      goto LABEL_58;
    }
    goto LABEL_49;
  }
  sub_39C7CA0(v9);
  if ( *(_QWORD *)(a1 + 4032) )
    sub_39A3F30(v9, v97, 27, *(_QWORD *)(a1 + 4024), *(_QWORD *)(a1 + 4032));
  sub_3989C90(a1, v9, v97);
  if ( *(_BYTE *)(a1 + 4512) )
  {
LABEL_49:
    if ( *(_BYTE *)(a2 + 28) )
      sub_39A34D0(v9, v97, 16353);
    v45 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
    if ( v45 )
    {
      v46 = sub_161E970(v45);
      if ( v47 )
        sub_39A3F30(v9, v97, 16354, v46, v47);
    }
    v48 = *(unsigned int *)(a2 + 32);
    if ( (_DWORD)v48 )
    {
      v107.m128i_i32[0] = 65547;
      sub_39A3560(v9, v9 + 16, 16357, &v107, v48);
    }
  }
  if ( *(_BYTE *)(a1 + 4513) )
    goto LABEL_76;
  *(_QWORD *)(v9 + 56) = *(_QWORD *)(sub_396DD80(*(_QWORD *)(a1 + 8)) + 88);
LABEL_58:
  v49 = *(_QWORD *)(a2 + 40);
  if ( v49 )
  {
    v107.m128i_i32[0] = 65543;
    sub_39A3560(v9, v9 + 16, 8497, &v107, v49);
    v59 = *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)));
    if ( v59 )
    {
      sub_161E970(v59);
      if ( v60 )
      {
        v61 = 3LL - *(unsigned int *)(a2 + 8);
        v62 = *(_QWORD *)(a2 + 8 * v61);
        if ( v62 )
        {
          v62 = sub_161E970(*(_QWORD *)(a2 + 8 * v61));
          v64 = v63;
        }
        else
        {
          v64 = 0;
        }
        sub_39A3F30(v9, v97, 8496, v62, v64);
      }
    }
  }
  v107.m128i_i64[0] = a2;
  v107.m128i_i64[1] = v9;
  sub_3999160(a1 + 520, &v107);
  v50 = *(_DWORD *)(a1 + 600);
  if ( !v50 )
  {
    ++*(_QWORD *)(a1 + 576);
    goto LABEL_134;
  }
  v51 = *(_QWORD *)(a1 + 584);
  v52 = (v50 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
  v53 = (__int64 *)(v51 + 16LL * v52);
  v54 = *v53;
  if ( v97 == *v53 )
    return v9;
  v55 = 1;
  v56 = 0;
  while ( v54 != -8 )
  {
    if ( v56 || v54 != -16 )
      v53 = v56;
    v52 = (v50 - 1) & (v55 + v52);
    v54 = *(_QWORD *)(v51 + 16LL * v52);
    if ( v97 == v54 )
      return v9;
    ++v55;
    v56 = v53;
    v53 = (__int64 *)(v51 + 16LL * v52);
  }
  if ( !v56 )
    v56 = v53;
  v57 = *(_DWORD *)(a1 + 592);
  ++*(_QWORD *)(a1 + 576);
  v58 = v57 + 1;
  if ( 4 * (v57 + 1) >= 3 * v50 )
  {
LABEL_134:
    sub_3992350(a1 + 576, 2 * v50);
    v77 = *(_DWORD *)(a1 + 600);
    if ( v77 )
    {
      v78 = v77 - 1;
      v79 = *(_QWORD *)(a1 + 584);
      v58 = *(_DWORD *)(a1 + 592) + 1;
      v80 = (v77 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
      v56 = (__int64 *)(v79 + 16LL * v80);
      v81 = *v56;
      if ( v97 != *v56 )
      {
        v82 = 1;
        v83 = 0;
        while ( v81 != -8 )
        {
          if ( !v83 && v81 == -16 )
            v83 = v56;
          v80 = v78 & (v82 + v80);
          v56 = (__int64 *)(v79 + 16LL * v80);
          v81 = *v56;
          if ( v97 == *v56 )
            goto LABEL_67;
          ++v82;
        }
        if ( v83 )
          v56 = v83;
      }
      goto LABEL_67;
    }
    goto LABEL_164;
  }
  if ( v50 - *(_DWORD *)(a1 + 596) - v58 <= v50 >> 3 )
  {
    sub_3992350(a1 + 576, v50);
    v84 = *(_DWORD *)(a1 + 600);
    if ( v84 )
    {
      v85 = v84 - 1;
      v86 = *(_QWORD *)(a1 + 584);
      v87 = 0;
      v88 = (v84 - 1) & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
      v89 = 1;
      v58 = *(_DWORD *)(a1 + 592) + 1;
      v56 = (__int64 *)(v86 + 16LL * v88);
      v90 = *v56;
      if ( v97 != *v56 )
      {
        while ( v90 != -8 )
        {
          if ( v90 == -16 && !v87 )
            v87 = v56;
          v88 = v85 & (v89 + v88);
          v56 = (__int64 *)(v86 + 16LL * v88);
          v90 = *v56;
          if ( v97 == *v56 )
            goto LABEL_67;
          ++v89;
        }
        if ( v87 )
          v56 = v87;
      }
      goto LABEL_67;
    }
LABEL_164:
    ++*(_DWORD *)(a1 + 592);
    BUG();
  }
LABEL_67:
  *(_DWORD *)(a1 + 592) = v58;
  if ( *v56 != -8 )
    --*(_DWORD *)(a1 + 596);
  v56[1] = v9;
  *v56 = v97;
  return v9;
}
