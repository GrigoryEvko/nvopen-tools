// Function: sub_1CF8EB0
// Address: 0x1cf8eb0
//
_QWORD *__fastcall sub_1CF8EB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rsi
  unsigned int v4; // ebx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  _QWORD *v7; // r14
  __int64 v8; // r12
  int v9; // r8d
  _QWORD *v10; // r15
  unsigned __int8 v11; // al
  unsigned int v13; // esi
  _QWORD *v14; // rdi
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rsi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r10
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned int v33; // ecx
  __int64 v34; // rax
  _QWORD *v35; // r10
  int v36; // r13d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r9
  unsigned int v44; // esi
  __int64 *v45; // rdx
  __int64 v46; // r11
  __int64 v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // r9
  unsigned int v50; // esi
  __int64 v51; // rax
  _QWORD *v52; // rbx
  __int64 v53; // rax
  __int64 v54; // rax
  __m128i *v55; // rax
  __int64 v56; // rdx
  char v57; // al
  __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // r10
  __int64 v61; // r9
  unsigned int v62; // edi
  __int64 *v63; // rsi
  __int64 v64; // r11
  unsigned int v65; // r13d
  __int64 v66; // r12
  unsigned int v67; // ebx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r15
  int v71; // r15d
  __int64 v72; // rax
  __int64 v73; // rdx
  int v74; // eax
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rdi
  unsigned int v78; // esi
  __int64 *v79; // rax
  __int64 v80; // r11
  __int64 v81; // rax
  __int64 v82; // rax
  __m128i *v83; // rax
  int v84; // eax
  int i; // edx
  int v86; // r8d
  int k; // edx
  int v88; // ecx
  int v89; // eax
  int v90; // r8d
  int v91; // eax
  int v92; // ecx
  int j; // esi
  int v94; // ecx
  __int64 v95; // rsi
  int v96; // ecx
  __int64 v97; // [rsp+0h] [rbp-2D0h]
  __int64 v98; // [rsp+8h] [rbp-2C8h]
  __int64 v101[2]; // [rsp+30h] [rbp-2A0h] BYREF
  __int64 v102; // [rsp+40h] [rbp-290h] BYREF
  __int64 v103; // [rsp+48h] [rbp-288h]
  __int64 v104; // [rsp+50h] [rbp-280h]
  __int64 v105; // [rsp+58h] [rbp-278h]
  __m128i v106; // [rsp+60h] [rbp-270h] BYREF
  __m128i v107; // [rsp+70h] [rbp-260h] BYREF
  __int64 v108; // [rsp+80h] [rbp-250h]
  _BYTE *v109; // [rsp+90h] [rbp-240h] BYREF
  __int64 v110; // [rsp+98h] [rbp-238h]
  _BYTE v111[560]; // [rsp+A0h] [rbp-230h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)a1;
  v101[1] = (__int64)&v109;
  v109 = v111;
  v101[0] = (__int64)&v102;
  v110 = 0x2000000000LL;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  sub_1CF8C20(v101, v3, 0);
  v4 = v110;
  if ( !(_DWORD)v110 )
    goto LABEL_13;
  while ( 1 )
  {
    v5 = v4--;
    v6 = (unsigned __int64)&v109[16 * v5 - 16];
    v7 = *(_QWORD **)v6;
    v8 = *(_QWORD *)(v6 + 8);
    LODWORD(v110) = v4;
    v10 = sub_1648700((__int64)v7);
    v11 = *((_BYTE *)v10 + 16);
    if ( v11 <= 0x17u )
      goto LABEL_3;
    if ( v11 == 55 )
    {
      v17 = *(v10 - 3);
      v18 = *(_QWORD *)a1;
      if ( *(_BYTE *)(v17 + 16) == 72 )
      {
        if ( v17 != v18 )
        {
          if ( *(_QWORD *)(v17 - 24) != v18 )
            goto LABEL_3;
          v19 = *(v10 - 6);
          if ( *(_BYTE *)(v19 + 16) != 54 )
            goto LABEL_3;
LABEL_29:
          v20 = *(_QWORD *)(v19 - 24);
          if ( *(_BYTE *)(v20 + 16) != 17
            || !(unsigned __int8)sub_15E0450(*(_QWORD *)(v19 - 24))
            || *(_QWORD *)(a1 + 16) )
          {
            goto LABEL_3;
          }
          v21 = v10[5];
          v22 = *(_QWORD *)(sub_15F2060((__int64)v10) + 80);
          if ( v22 )
            v22 -= 24;
          if ( v22 != v21 )
            goto LABEL_3;
          v23 = *(_QWORD *)(a1 + 8);
          v24 = *(unsigned int *)(v23 + 48);
          if ( !(_DWORD)v24 )
            goto LABEL_3;
          v25 = v10[5];
          v26 = *(_QWORD *)(v23 + 32);
          v27 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v25 != *v28 )
          {
            for ( i = 1; ; i = v86 )
            {
              if ( v29 == -8 )
                goto LABEL_3;
              v86 = i + 1;
              v27 = (v24 - 1) & (i + v27);
              v28 = (__int64 *)(v26 + 16LL * v27);
              v29 = *v28;
              if ( v25 == *v28 )
                break;
            }
          }
          if ( v28 == (__int64 *)(v26 + 16 * v24) )
            goto LABEL_3;
          v30 = v28[1];
          if ( !v30 )
            goto LABEL_3;
          v31 = *(unsigned int *)(a2 + 32);
          v32 = *(_QWORD *)(a2 + 16);
          if ( (_DWORD)v31 )
          {
            v33 = (v31 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v34 = v32 + 16LL * v33;
            v35 = *(_QWORD **)v34;
            if ( v10 == *(_QWORD **)v34 )
            {
LABEL_41:
              v36 = *(_DWORD *)(v34 + 8);
              v37 = sub_22077B0(32);
              if ( v37 )
              {
                v38 = *(_QWORD *)(v30 + 48);
                *(_DWORD *)v37 = v36;
                *(_QWORD *)(v37 + 16) = v20;
                *(_QWORD *)(v37 + 4) = v38;
                *(_QWORD *)(v37 + 24) = v10;
              }
              v39 = *(_QWORD *)(a1 + 16);
              *(_QWORD *)(a1 + 16) = v37;
              if ( v39 )
                j_j___libc_free_0(v39, 32);
              goto LABEL_56;
            }
            v89 = 1;
            while ( v35 != (_QWORD *)-8LL )
            {
              v90 = v89 + 1;
              v33 = (v31 - 1) & (v89 + v33);
              v34 = v32 + 16LL * v33;
              v35 = *(_QWORD **)v34;
              if ( v10 == *(_QWORD **)v34 )
                goto LABEL_41;
              v89 = v90;
            }
          }
          v34 = v32 + 16 * v31;
          goto LABEL_41;
        }
      }
      else if ( v17 != v18 )
      {
        goto LABEL_3;
      }
      v19 = *(v10 - 6);
      if ( *(_BYTE *)(v19 + 16) != 54 )
        goto LABEL_3;
      goto LABEL_29;
    }
    if ( (unsigned __int8)(v11 - 71) <= 1u )
      goto LABEL_8;
    if ( v11 == 56 )
    {
      if ( *v7 != v10[-3 * (*((_DWORD *)v10 + 5) & 0xFFFFFFF)] )
        goto LABEL_3;
      goto LABEL_8;
    }
    if ( v11 == 54 )
      break;
    if ( v11 != 78 )
    {
      if ( v11 != 77 )
        goto LABEL_3;
      v13 = *((_DWORD *)v10 + 5) & 0xFFFFFFF;
      if ( (*((_BYTE *)v10 + 23) & 0x40) != 0 )
        v14 = (_QWORD *)*(v10 - 1);
      else
        v14 = &v10[-3 * v13];
      if ( v13 > 1 )
      {
        v15 = v14 + 3;
        v16 = (__int64)&v14[3 * v13];
        while ( *v14 == *v15 )
        {
          v15 += 3;
          if ( v15 == (_QWORD *)v16 )
            goto LABEL_8;
        }
        goto LABEL_3;
      }
LABEL_8:
      if ( !v8 )
        v8 = (__int64)v7;
      sub_1CF8C20(v101, (__int64)v10, v8);
      v4 = v110;
      goto LABEL_11;
    }
    v56 = *(v10 - 3);
    v57 = *(_BYTE *)(v56 + 16);
    if ( !v57 && (*(_BYTE *)(v56 + 33) & 0x20) != 0 )
    {
      if ( (unsigned int)(*(_DWORD *)(v56 + 36) - 116) <= 1 )
        goto LABEL_11;
      goto LABEL_3;
    }
    v58 = *(_QWORD *)(a1 + 8);
    v59 = *(unsigned int *)(v58 + 48);
    if ( !(_DWORD)v59 )
      goto LABEL_3;
    v60 = v10[5];
    v61 = *(_QWORD *)(v58 + 32);
    v62 = (v59 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
    v63 = (__int64 *)(v61 + 16LL * v62);
    v64 = *v63;
    if ( v60 != *v63 )
    {
      for ( j = 1; ; j = v94 )
      {
        if ( v64 == -8 )
          goto LABEL_3;
        v94 = j + 1;
        v95 = ((_DWORD)v59 - 1) & (v62 + j);
        v62 = v95;
        v63 = (__int64 *)(v61 + 16 * v95);
        v64 = *v63;
        if ( v60 == *v63 )
          break;
      }
    }
    if ( v63 == (__int64 *)(v61 + 16 * v59) )
      goto LABEL_3;
    v97 = v63[1];
    if ( !v97 || v57 == 20 )
      goto LABEL_3;
    v98 = v8;
    v65 = 0;
    v66 = (__int64)v10;
    v67 = *((_DWORD *)v10 + 5) & 0xFFFFFFF;
    while ( 1 )
    {
      if ( *(char *)(v66 + 23) >= 0 )
        goto LABEL_74;
      v68 = sub_1648A40(v66);
      v70 = v68 + v69;
      if ( *(char *)(v66 + 23) >= 0 )
      {
        if ( (unsigned int)(v70 >> 4) )
LABEL_122:
          BUG();
LABEL_74:
        v74 = 0;
        goto LABEL_70;
      }
      if ( !(unsigned int)((v70 - sub_1648A40(v66)) >> 4) )
        goto LABEL_74;
      if ( *(char *)(v66 + 23) >= 0 )
        goto LABEL_122;
      v71 = *(_DWORD *)(sub_1648A40(v66) + 8);
      if ( *(char *)(v66 + 23) >= 0 )
        BUG();
      v72 = sub_1648A40(v66);
      v74 = *(_DWORD *)(v72 + v73 - 4) - v71;
LABEL_70:
      if ( v65 >= v67 - 1 - v74 )
        break;
      v67 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
      if ( *v7 == *(_QWORD *)(v66 + 24 * (v65 - (unsigned __int64)v67)) )
      {
        if ( !(unsigned __int8)sub_1560290((_QWORD *)(v66 + 56), v65, 6) )
        {
          v75 = *(_QWORD *)(v66 - 24);
          if ( *(_BYTE *)(v75 + 16)
            || (v106.m128i_i64[0] = *(_QWORD *)(v75 + 112), !(unsigned __int8)sub_1560290(&v106, v65, 6)) )
          {
            v10 = (_QWORD *)v66;
            goto LABEL_3;
          }
        }
        v67 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
      }
      ++v65;
    }
    v76 = *(unsigned int *)(a2 + 32);
    v77 = *(_QWORD *)(a2 + 16);
    if ( (_DWORD)v76 )
    {
      LODWORD(v61) = v76 - 1;
      v78 = (v76 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
      v79 = (__int64 *)(v77 + 16LL * v78);
      v80 = *v79;
      if ( *v79 == v66 )
        goto LABEL_91;
      v84 = 1;
      while ( v80 != -8 )
      {
        v96 = v84 + 1;
        v78 = v61 & (v84 + v78);
        v79 = (__int64 *)(v77 + 16LL * v78);
        v80 = *v79;
        if ( *v79 == v66 )
          goto LABEL_91;
        v84 = v96;
      }
    }
    v79 = (__int64 *)(v77 + 16 * v76);
LABEL_91:
    v106.m128i_i32[0] = *((_DWORD *)v79 + 2);
    v81 = *(_QWORD *)(v97 + 48);
    v107.m128i_i64[1] = (__int64)v7;
    v107.m128i_i64[0] = 0;
    *(__int64 *)((char *)v106.m128i_i64 + 4) = v81;
    v82 = *(unsigned int *)(a1 + 32);
    v108 = v98;
    if ( (unsigned int)v82 >= *(_DWORD *)(a1 + 36) )
    {
      sub_16CD150(a1 + 24, (const void *)(a1 + 40), 0, 40, v9, v61);
      v82 = *(unsigned int *)(a1 + 32);
    }
    v83 = (__m128i *)(*(_QWORD *)(a1 + 24) + 40 * v82);
    *v83 = _mm_loadu_si128(&v106);
    v4 = v110;
    v83[1] = _mm_loadu_si128(&v107);
    v83[2].m128i_i64[0] = v108;
    ++*(_DWORD *)(a1 + 32);
LABEL_11:
    if ( !v4 )
    {
      v2 = a1;
LABEL_13:
      v10 = 0;
      if ( !*(_QWORD *)(v2 + 16) )
        v10 = *(_QWORD **)v2;
      goto LABEL_3;
    }
  }
  v40 = *(_QWORD *)(a1 + 8);
  v41 = *(unsigned int *)(v40 + 48);
  if ( !(_DWORD)v41 )
    goto LABEL_3;
  v42 = v10[5];
  v43 = *(_QWORD *)(v40 + 32);
  v44 = (v41 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
  v45 = (__int64 *)(v43 + 16LL * v44);
  v46 = *v45;
  if ( v42 != *v45 )
  {
    for ( k = 1; ; k = v88 )
    {
      if ( v46 == -8 )
        goto LABEL_3;
      v88 = k + 1;
      v44 = (v41 - 1) & (k + v44);
      v45 = (__int64 *)(v43 + 16LL * v44);
      v46 = *v45;
      if ( v42 == *v45 )
        break;
    }
  }
  if ( v45 != (__int64 *)(v43 + 16 * v41) )
  {
    v47 = v45[1];
    if ( v47 )
    {
      v48 = *(unsigned int *)(a2 + 32);
      v49 = *(_QWORD *)(a2 + 16);
      if ( (_DWORD)v48 )
      {
        v50 = (v48 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v51 = v49 + 16LL * v50;
        v52 = *(_QWORD **)v51;
        if ( v10 == *(_QWORD **)v51 )
          goto LABEL_53;
        v91 = 1;
        while ( v52 != (_QWORD *)-8LL )
        {
          v92 = v91 + 1;
          v50 = (v48 - 1) & (v91 + v50);
          v51 = v49 + 16LL * v50;
          v52 = *(_QWORD **)v51;
          if ( v10 == *(_QWORD **)v51 )
            goto LABEL_53;
          v91 = v92;
        }
      }
      v51 = v49 + 16 * v48;
LABEL_53:
      v106.m128i_i32[0] = *(_DWORD *)(v51 + 8);
      v53 = *(_QWORD *)(v47 + 48);
      v107.m128i_i64[0] = 0;
      *(__int64 *)((char *)v106.m128i_i64 + 4) = v53;
      v54 = *(unsigned int *)(a1 + 32);
      v107.m128i_i64[1] = (__int64)v7;
      v108 = v8;
      if ( (unsigned int)v54 >= *(_DWORD *)(a1 + 36) )
      {
        sub_16CD150(a1 + 24, (const void *)(a1 + 40), 0, 40, v9, v49);
        v54 = *(unsigned int *)(a1 + 32);
      }
      v55 = (__m128i *)(*(_QWORD *)(a1 + 24) + 40 * v54);
      *v55 = _mm_loadu_si128(&v106);
      v55[1] = _mm_loadu_si128(&v107);
      v55[2].m128i_i64[0] = v108;
      ++*(_DWORD *)(a1 + 32);
LABEL_56:
      v4 = v110;
      goto LABEL_11;
    }
  }
LABEL_3:
  j___libc_free_0(v103);
  if ( v109 != v111 )
    _libc_free((unsigned __int64)v109);
  return v10;
}
