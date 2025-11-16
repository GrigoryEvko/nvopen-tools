// Function: sub_12A2C80
// Address: 0x12a2c80
//
const char *__fastcall sub_12A2C80(__int64 a1, __int64 a2)
{
  const char *v4; // rax
  __int64 v5; // rdx
  char *v6; // r13
  const char *v7; // rax
  __int64 v8; // rdx
  char *v9; // r14
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // r10
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  __m128i *v17; // rax
  void **v18; // rcx
  __m128i *v19; // rdx
  void **v20; // rdi
  size_t v21; // rdx
  __int64 v22; // rsi
  int v23; // ebx
  __int64 v24; // r14
  char v25; // al
  __int64 v26; // r10
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r10
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // r10
  __int64 v36; // rax
  unsigned int v37; // esi
  __int64 v38; // rdi
  unsigned int v39; // ecx
  __int64 *v40; // rax
  __int64 v41; // rdx
  const char *result; // rax
  unsigned __int64 v43; // rcx
  __int8 *v44; // r14
  size_t v45; // r9
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  char *v50; // rdi
  __int64 v51; // rax
  _QWORD *v52; // rdi
  unsigned int v53; // eax
  int v54; // eax
  int v55; // edi
  __int64 v56; // rsi
  __int64 v57; // rdx
  int v58; // ecx
  __int64 v59; // r8
  int v60; // r11d
  __int64 *v61; // r9
  int v62; // r11d
  __int64 *v63; // r9
  int v64; // ecx
  int v65; // eax
  int v66; // edx
  __int64 v67; // rdi
  __int64 *v68; // r8
  __int64 v69; // rbx
  int v70; // r9d
  __int64 v71; // rsi
  unsigned int v72; // eax
  _QWORD *v73; // [rsp+0h] [rbp-130h]
  __int64 v74; // [rsp+8h] [rbp-128h]
  __int64 v75; // [rsp+8h] [rbp-128h]
  __m128i *v76; // [rsp+10h] [rbp-120h]
  __int64 v77; // [rsp+10h] [rbp-120h]
  int v78; // [rsp+10h] [rbp-120h]
  int v79; // [rsp+10h] [rbp-120h]
  int v80; // [rsp+18h] [rbp-118h]
  size_t v81; // [rsp+18h] [rbp-118h]
  int v82; // [rsp+18h] [rbp-118h]
  _QWORD *v83; // [rsp+20h] [rbp-110h]
  unsigned int v84; // [rsp+20h] [rbp-110h]
  __int64 v85; // [rsp+20h] [rbp-110h]
  int v86; // [rsp+20h] [rbp-110h]
  __int64 v87; // [rsp+20h] [rbp-110h]
  __int64 v88; // [rsp+20h] [rbp-110h]
  unsigned __int8 v89; // [rsp+37h] [rbp-F9h] BYREF
  __int64 v90; // [rsp+38h] [rbp-F8h] BYREF
  void *dest; // [rsp+40h] [rbp-F0h] BYREF
  size_t v92; // [rsp+48h] [rbp-E8h]
  _QWORD v93[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v94[2]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v95[2]; // [rsp+70h] [rbp-C0h] BYREF
  __m128i *v96; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+88h] [rbp-A8h]
  __m128i v98; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v99[2]; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD v100[2]; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD *v101; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v102; // [rsp+C8h] [rbp-68h]
  _QWORD v103[2]; // [rsp+D0h] [rbp-60h] BYREF
  void **p_src; // [rsp+E0h] [rbp-50h] BYREF
  size_t n; // [rsp+E8h] [rbp-48h]
  __m128i src; // [rsp+F0h] [rbp-40h] BYREF

  if ( sub_12A2A10(a1, a2) )
    sub_127B550("unexpected: declaration for variable already exists!", (_DWORD *)(a2 + 64), 1);
  dest = v93;
  v92 = 0;
  LOBYTE(v93[0]) = 0;
  sub_72F9F0(a2, 0, &v89, &v90);
  v76 = sub_740200(a2);
  if ( v89 != 3 && v89 > 1u && !v76 && (v89 != 2 || *(_BYTE *)(*(_QWORD *)v90 + 48LL) > 1u) )
    sub_127B550("function-scope static variable is initialized with non-constant initializer!", (_DWORD *)(a2 + 64), 1);
  if ( !*(_QWORD *)(a2 + 8) )
  {
    v43 = sub_737880(a2);
    if ( !v43 )
    {
      src.m128i_i8[4] = 48;
      v44 = &src.m128i_i8[4];
      v99[0] = (__int64)v100;
LABEL_64:
      v45 = 1;
      LOBYTE(v100[0]) = *v44;
      v46 = v100;
LABEL_82:
      v99[1] = v45;
      *((_BYTE *)v46 + v45) = 0;
      goto LABEL_9;
    }
    v44 = &src.m128i_i8[5];
    do
    {
      *--v44 = v43 % 0xA + 48;
      v49 = v43;
      v43 /= 0xAu;
    }
    while ( v49 > 9 );
    v50 = (char *)(&src.m128i_u8[5] - (unsigned __int8 *)v44);
    v99[0] = (__int64)v100;
    v45 = &src.m128i_u8[5] - (unsigned __int8 *)v44;
    v101 = (_QWORD *)(&src.m128i_u8[5] - (unsigned __int8 *)v44);
    if ( (unsigned __int64)(&src.m128i_u8[5] - (unsigned __int8 *)v44) <= 0xF )
    {
      if ( v50 == (char *)1 )
        goto LABEL_64;
      if ( !v50 )
      {
        v46 = v100;
        goto LABEL_82;
      }
      v52 = v100;
    }
    else
    {
      v51 = sub_22409D0(v99, &v101, 0);
      v45 = &src.m128i_u8[5] - (unsigned __int8 *)v44;
      v99[0] = v51;
      v52 = (_QWORD *)v51;
      v100[0] = v101;
    }
    memcpy(v52, v44, v45);
    v45 = (size_t)v101;
    v46 = (_QWORD *)v99[0];
    goto LABEL_82;
  }
  v4 = (const char *)sub_127B360(a2);
  v5 = -1;
  v99[0] = (__int64)v100;
  v6 = (char *)v4;
  if ( v4 )
    v5 = (__int64)&v4[strlen(v4)];
  sub_12A27A0(v99, v6, v5);
LABEL_9:
  sub_127B670((__int64 *)&v101, v99, a2);
  v7 = (const char *)sub_127B370(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 32LL));
  v8 = -1;
  v9 = (char *)v7;
  v94[0] = (__int64)v95;
  if ( v7 )
    v8 = (__int64)&v7[strlen(v7)];
  sub_12A27A0(v94, v9, v8);
  if ( v94[1] == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v11 = (__m128i *)sub_2241490(v94, "$", 1, v10);
  v96 = &v98;
  if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
  {
    v98 = _mm_loadu_si128(v11 + 1);
  }
  else
  {
    v96 = (__m128i *)v11->m128i_i64[0];
    v98.m128i_i64[0] = v11[1].m128i_i64[0];
  }
  v12 = v11->m128i_i64[1];
  v11[1].m128i_i8[0] = 0;
  v97 = v12;
  v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
  v13 = v96;
  v11->m128i_i64[1] = 0;
  v14 = 15;
  v15 = 15;
  if ( v13 != &v98 )
    v15 = v98.m128i_i64[0];
  v16 = v97 + v102;
  if ( v97 + v102 <= v15 )
    goto LABEL_20;
  if ( v101 != v103 )
    v14 = v103[0];
  if ( v16 <= v14 )
  {
    v17 = (__m128i *)sub_2241130(&v101, 0, 0, v13, v97);
    p_src = (void **)&src;
    v18 = (void **)v17->m128i_i64[0];
    v19 = v17 + 1;
    if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
      goto LABEL_21;
  }
  else
  {
LABEL_20:
    v17 = (__m128i *)sub_2241490(&v96, v101, v102, v16);
    p_src = (void **)&src;
    v18 = (void **)v17->m128i_i64[0];
    v19 = v17 + 1;
    if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
    {
LABEL_21:
      p_src = v18;
      src.m128i_i64[0] = v17[1].m128i_i64[0];
      goto LABEL_22;
    }
  }
  src = _mm_loadu_si128(v17 + 1);
LABEL_22:
  n = v17->m128i_u64[1];
  v17->m128i_i64[0] = (__int64)v19;
  v17->m128i_i64[1] = 0;
  v17[1].m128i_i8[0] = 0;
  v20 = (void **)dest;
  v21 = n;
  if ( p_src == (void **)&src )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src.m128i_i8[0];
      else
        memcpy(dest, &src, n);
      v21 = n;
      v20 = (void **)dest;
    }
    v92 = v21;
    *((_BYTE *)v20 + v21) = 0;
    v20 = p_src;
  }
  else
  {
    if ( dest == v93 )
    {
      dest = p_src;
      v92 = n;
      v93[0] = src.m128i_i64[0];
    }
    else
    {
      v22 = v93[0];
      dest = p_src;
      v92 = n;
      v93[0] = src.m128i_i64[0];
      if ( v20 )
      {
        p_src = v20;
        src.m128i_i64[0] = v22;
        goto LABEL_26;
      }
    }
    p_src = (void **)&src;
    v20 = (void **)&src;
  }
LABEL_26:
  n = 0;
  *(_BYTE *)v20 = 0;
  if ( p_src != (void **)&src )
    j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
  if ( v96 != &v98 )
    j_j___libc_free_0(v96, v98.m128i_i64[0] + 1);
  if ( (_QWORD *)v94[0] != v95 )
    j_j___libc_free_0(v94[0], v95[0] + 1LL);
  if ( v101 != v103 )
    j_j___libc_free_0(v101, v103[0] + 1LL);
  if ( (_QWORD *)v99[0] != v100 )
    j_j___libc_free_0(v99[0], v100[0] + 1LL);
  v23 = *(_BYTE *)(*(_QWORD *)(a1 + 120) + 32LL) & 0xF;
  if ( ((*(_BYTE *)(*(_QWORD *)(a1 + 120) + 32LL) + 14) & 0xFu) > 3 && v23 != 10 && v23 != 9 )
    v23 = 7;
  v24 = sub_127A040(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(a2 + 120));
  if ( (*(_BYTE *)(a2 + 156) & 2) != 0 )
  {
    v26 = sub_127D2C0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 120));
  }
  else
  {
    if ( !v76 )
    {
      v81 = **(_QWORD **)(a1 + 32);
      v86 = sub_15A06D0(v24);
      src.m128i_i16[0] = 260;
      p_src = &dest;
      v78 = sub_127BFC0(a2);
      v47 = sub_1648A60(88, 1);
      v30 = v47;
      if ( v47 )
      {
        v32 = v81;
        sub_15E51E0(v47, v81, v24, 0, v23, v86, (__int64)&p_src, 0, 0, v78, 0);
        v85 = v24;
LABEL_46:
        v34 = sub_127C800(a2, v32, v33);
        sub_15E4CC0(v30, v34);
        v35 = v30;
        if ( v24 != v85 )
        {
          v36 = sub_1646BA0(v24, 0);
          v35 = sub_15A4510(v30, v36, 0);
        }
LABEL_48:
        v37 = *(_DWORD *)(a1 + 24);
        if ( v37 )
          goto LABEL_49;
LABEL_87:
        ++*(_QWORD *)a1;
        goto LABEL_88;
      }
      goto LABEL_121;
    }
    v83 = (_QWORD *)sub_127F650(a1, v76, *(_QWORD *)(a2 + 120));
    v25 = sub_12789C0(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(a2 + 120));
    v26 = (__int64)v83;
    if ( v25 )
    {
      v73 = v83;
      v85 = *v83;
      v75 = **(_QWORD **)(a1 + 32);
      v79 = sub_15A06D0(v85);
      src.m128i_i16[0] = 260;
      p_src = &dest;
      v82 = sub_127BFC0(a2);
      v48 = sub_1648A60(88, 1);
      v29 = (__int64)v73;
      v30 = v48;
      if ( v48 )
      {
        sub_15E51E0(v48, v75, v85, 0, v23, v79, (__int64)&p_src, 0, 0, v82, 0);
        v29 = (__int64)v73;
      }
      goto LABEL_45;
    }
  }
  v74 = v26;
  v77 = **(_QWORD **)(a1 + 32);
  v80 = sub_15A06D0(v24);
  src.m128i_i16[0] = 260;
  p_src = &dest;
  v84 = sub_127BFC0(a2);
  v27 = sub_1648A60(88, 1);
  v28 = v84;
  v29 = v74;
  v30 = v27;
  if ( !v27 )
  {
    v85 = v24;
    if ( v74 )
      goto LABEL_45;
LABEL_121:
    v72 = sub_127C800(a2, 1, v28);
    sub_15E4CC0(0, v72);
    v35 = 0;
    goto LABEL_48;
  }
  sub_15E51E0(v27, v77, v24, 0, v23, v80, (__int64)&p_src, 0, 0, v84, 0);
  v29 = v74;
  v85 = v24;
  if ( v74 )
  {
LABEL_45:
    v32 = v30;
    sub_126A090(*(_QWORD *)(a1 + 32), v30, v29, a2);
    goto LABEL_46;
  }
  v53 = sub_127C800(a2, v77, v31);
  sub_15E4CC0(v30, v53);
  v37 = *(_DWORD *)(a1 + 24);
  v35 = v30;
  if ( !v37 )
    goto LABEL_87;
LABEL_49:
  v38 = *(_QWORD *)(a1 + 8);
  v39 = (v37 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v40 = (__int64 *)(v38 + 16LL * v39);
  v41 = *v40;
  if ( a2 == *v40 )
    goto LABEL_50;
  v62 = 1;
  v63 = 0;
  while ( v41 != -8 )
  {
    if ( !v63 && v41 == -16 )
      v63 = v40;
    v39 = (v37 - 1) & (v62 + v39);
    v40 = (__int64 *)(v38 + 16LL * v39);
    v41 = *v40;
    if ( a2 == *v40 )
      goto LABEL_50;
    ++v62;
  }
  v64 = *(_DWORD *)(a1 + 16);
  if ( v63 )
    v40 = v63;
  ++*(_QWORD *)a1;
  v58 = v64 + 1;
  if ( 4 * v58 >= 3 * v37 )
  {
LABEL_88:
    v87 = v35;
    sub_12A2850(a1, 2 * v37);
    v54 = *(_DWORD *)(a1 + 24);
    if ( v54 )
    {
      v55 = v54 - 1;
      v56 = *(_QWORD *)(a1 + 8);
      v35 = v87;
      LODWORD(v57) = (v54 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v58 = *(_DWORD *)(a1 + 16) + 1;
      v40 = (__int64 *)(v56 + 16LL * (unsigned int)v57);
      v59 = *v40;
      if ( a2 != *v40 )
      {
        v60 = 1;
        v61 = 0;
        while ( v59 != -8 )
        {
          if ( !v61 && v59 == -16 )
            v61 = v40;
          v57 = v55 & (unsigned int)(v57 + v60);
          v40 = (__int64 *)(v56 + 16 * v57);
          v59 = *v40;
          if ( a2 == *v40 )
            goto LABEL_104;
          ++v60;
        }
        if ( v61 )
          v40 = v61;
      }
      goto LABEL_104;
    }
    goto LABEL_134;
  }
  if ( v37 - *(_DWORD *)(a1 + 20) - v58 <= v37 >> 3 )
  {
    v88 = v35;
    sub_12A2850(a1, v37);
    v65 = *(_DWORD *)(a1 + 24);
    if ( v65 )
    {
      v66 = v65 - 1;
      v67 = *(_QWORD *)(a1 + 8);
      v68 = 0;
      LODWORD(v69) = (v65 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v35 = v88;
      v70 = 1;
      v58 = *(_DWORD *)(a1 + 16) + 1;
      v40 = (__int64 *)(v67 + 16LL * (unsigned int)v69);
      v71 = *v40;
      if ( a2 != *v40 )
      {
        while ( v71 != -8 )
        {
          if ( !v68 && v71 == -16 )
            v68 = v40;
          v69 = v66 & (unsigned int)(v69 + v70);
          v40 = (__int64 *)(v67 + 16 * v69);
          v71 = *v40;
          if ( a2 == *v40 )
            goto LABEL_104;
          ++v70;
        }
        if ( v68 )
          v40 = v68;
      }
      goto LABEL_104;
    }
LABEL_134:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_104:
  *(_DWORD *)(a1 + 16) = v58;
  if ( *v40 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v40 = a2;
  v40[1] = 0;
LABEL_50:
  v40[1] = v35;
  if ( sub_127B250(a2) )
  {
    sub_1273CD0(*(_QWORD *)(a1 + 32), v30);
    result = (const char *)&dword_4D046B4;
    if ( !dword_4D046B4 )
      goto LABEL_52;
  }
  else
  {
    result = (const char *)&dword_4D046B4;
    if ( !dword_4D046B4 )
      goto LABEL_52;
  }
  result = sub_12A24A0(*(_QWORD **)(*(_QWORD *)(a1 + 32) + 384LL), v30, a2, 0);
LABEL_52:
  if ( dest != v93 )
    return (const char *)j_j___libc_free_0(dest, v93[0] + 1LL);
  return result;
}
