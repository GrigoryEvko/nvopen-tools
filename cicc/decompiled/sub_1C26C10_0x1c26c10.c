// Function: sub_1C26C10
// Address: 0x1c26c10
//
__int64 *__fastcall sub_1C26C10(__int64 *a1, __int64 *a2, __int64 a3, __m128i a4)
{
  __int64 *v4; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  char v11; // al
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned int v19; // esi
  __int64 v20; // rdi
  __int64 i; // rcx
  char v22; // r9
  char v23; // dl
  char v24; // al
  bool v25; // cf
  __int8 v26; // dl
  __int8 v27; // al
  char v28; // r8
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // r14
  unsigned __int64 v32; // rdx
  unsigned __int64 *v33; // r15
  unsigned __int64 *v34; // r13
  unsigned __int64 v35; // rdi
  unsigned __int64 *v36; // r15
  unsigned __int64 v37; // r13
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  __int64 *v40; // rax
  __int64 v41; // r10
  const void *v42; // rsi
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __m128i *v48; // rcx
  __m128i *v49; // rdx
  char *(*v50)(); // rax
  char v51; // al
  unsigned __int64 v52; // rax
  __m128i *v53; // rax
  __m128i *v54; // r8
  __int64 v55; // rax
  __int64 v56; // rsi
  char v57; // al
  __m128i *v58; // r8
  __int64 *v59; // r12
  __int64 v60; // rdx
  _QWORD **v61; // r14
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r14
  __int64 v67; // r15
  int v68; // eax
  unsigned __int64 *v69; // rbx
  unsigned __int64 *v70; // r13
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rdi
  unsigned __int64 *v73; // rbx
  unsigned __int64 *v74; // r13
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // rdi
  _BYTE *v77; // rax
  size_t v78; // r10
  __int64 v79; // rdx
  _QWORD *v80; // r14
  __int64 v81; // [rsp-8h] [rbp-238h]
  __int64 *v82; // [rsp+10h] [rbp-220h]
  void *src; // [rsp+18h] [rbp-218h]
  __int64 v84; // [rsp+20h] [rbp-210h]
  size_t v85; // [rsp+20h] [rbp-210h]
  __int64 *v86; // [rsp+28h] [rbp-208h]
  unsigned int v87; // [rsp+28h] [rbp-208h]
  __int64 v88; // [rsp+30h] [rbp-200h]
  __int64 v89; // [rsp+30h] [rbp-200h]
  __m128i *v90; // [rsp+30h] [rbp-200h]
  int na; // [rsp+38h] [rbp-1F8h]
  _BYTE *n; // [rsp+38h] [rbp-1F8h]
  size_t nb; // [rsp+38h] [rbp-1F8h]
  __int64 v94; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 v95; // [rsp+48h] [rbp-1E8h] BYREF
  __int64 v96; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v97; // [rsp+58h] [rbp-1D8h] BYREF
  __int64 v98; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 v99; // [rsp+68h] [rbp-1C8h] BYREF
  __int64 v100[2]; // [rsp+70h] [rbp-1C0h] BYREF
  _QWORD **v101; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v102; // [rsp+88h] [rbp-1A8h]
  unsigned __int64 v103; // [rsp+90h] [rbp-1A0h]
  unsigned __int64 v104; // [rsp+98h] [rbp-198h]
  int v105; // [rsp+A0h] [rbp-190h]
  _QWORD *v106; // [rsp+A8h] [rbp-188h] BYREF
  unsigned __int64 v107; // [rsp+B0h] [rbp-180h]
  _QWORD v108[2]; // [rsp+B8h] [rbp-178h] BYREF
  unsigned __int8 v109; // [rsp+C8h] [rbp-168h]
  __int64 v110; // [rsp+D0h] [rbp-160h]
  __int64 v111; // [rsp+E0h] [rbp-150h] BYREF
  unsigned __int64 v112; // [rsp+E8h] [rbp-148h]
  char *v113; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v114; // [rsp+F8h] [rbp-138h]

  v4 = a1;
  if ( !a3 || *(_QWORD *)(*a2 + 16) - *(_QWORD *)(*a2 + 8) <= 3u )
  {
    *a1 = 0;
    return v4;
  }
  v8 = sub_22077B0(104);
  if ( v8 )
  {
    memset((void *)v8, 0, 0x68u);
    *(_QWORD *)(v8 + 88) = 1;
    *(_QWORD *)(v8 + 16) = v8 + 32;
    *(_QWORD *)(v8 + 24) = 0x400000000LL;
    *(_QWORD *)(v8 + 64) = v8 + 80;
  }
  v9 = *a2;
  v10 = *(_QWORD *)(*a2 + 8);
  if ( *(_DWORD *)v10 != 2135835629 )
  {
    v101 = 0;
    v102 = 0;
    v106 = v108;
    v103 = 0;
    v104 = 0;
    v107 = 0;
    LOBYTE(v108[0]) = 0;
    v100[0] = a3;
    v100[1] = v8;
    sub_16E40A0(
      (__int64)&v111,
      *(_QWORD *)(v9 + 8),
      *(_QWORD *)(v9 + 16) - *(_QWORD *)(v9 + 8),
      0,
      (__int64)nullsub_668,
      (__int64)v100);
    sub_16E4090((__int64)&v111, (__int64)v100);
    sub_16E7420((__int64)&v111, (__int64)v100);
    sub_16E3830((__int64)&v111);
    sub_1C23370((__int64)&v111, (__int64)&v101);
    sub_16E46D0((__int64)&v111);
    v11 = sub_1C14BC0((__int64)v101, v102, v103, v104, v109);
    if ( v109 )
      v104 = 7;
    if ( v11 )
    {
      sub_16E3EB0((__int64)&v111);
LABEL_12:
      *a1 = 0;
LABEL_13:
      if ( v106 != v108 )
        j_j___libc_free_0(v106, v108[0] + 1LL);
      goto LABEL_15;
    }
    na = sub_16E4240((__int64)&v111);
    sub_16E3EB0((__int64)&v111);
    if ( na )
      goto LABEL_12;
    v112 = 0;
    v111 = (__int64)&v113;
    LOBYTE(v113) = 0;
    if ( v109 )
    {
      if ( (v107 & 1) != 0 )
        goto LABEL_12;
      v88 = v107 & 1;
      sub_22410F0(&v111, v107 >> 1, 0);
      v19 = 0;
      v20 = (unsigned int)(v107 >> 1);
      if ( (unsigned int)(v107 >> 1) )
      {
        for ( i = v88; v20 != i; ++i )
        {
          v28 = *((_BYTE *)v106 + v19);
          if ( (unsigned __int8)(v28 - 48) > 9u )
          {
            if ( (unsigned __int8)(v28 - 97) <= 5u )
            {
              v22 = 16 * (v28 - 87);
            }
            else
            {
              v22 = -16;
              if ( (unsigned __int8)(v28 - 65) <= 5u )
                v22 = 16 * (v28 - 55);
            }
          }
          else
          {
            v22 = 16 * (v28 - 48);
          }
          v23 = *((_BYTE *)v106 + v19 + 1);
          v24 = v23 - 48;
          if ( (unsigned __int8)(v23 - 48) > 9u )
          {
            if ( (unsigned __int8)(v23 - 97) > 5u )
            {
              v25 = (unsigned __int8)(v23 - 65) < 6u;
              v26 = v22 | (v23 - 55);
              v27 = -1;
              if ( v25 )
                v27 = v26;
              goto LABEL_40;
            }
            v24 = v23 - 87;
          }
          v27 = v22 | v24;
LABEL_40:
          v19 += 2;
          *(_BYTE *)(v111 + i) = v27;
        }
      }
    }
    else
    {
      sub_2240AE0(&v111, &v106);
    }
    sub_16C2450(&v97, v111, v112, (__int64)byte_3F871B3, 0);
    v29 = (_QWORD *)v97;
    v97 = 0;
    v99 = (__int64)v29;
    sub_1C17660(&v98, (_QWORD **)&v99, v110, a3, v105, a4);
    if ( v99 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v99 + 8LL))(v99);
    v30 = (_QWORD *)v98;
    if ( v98 )
    {
      v31 = *(_QWORD *)(v98 + 144);
      *(_QWORD *)(v98 + 40) = v101;
      v30[6] = v102;
      v30[7] = v103;
      v32 = v104;
      v30[18] = v8;
      v30[8] = v32;
      if ( v31 )
      {
        v33 = *(unsigned __int64 **)(v31 + 16);
        v34 = &v33[*(unsigned int *)(v31 + 24)];
        while ( v34 != v33 )
        {
          v35 = *v33++;
          _libc_free(v35);
        }
        v36 = *(unsigned __int64 **)(v31 + 64);
        v37 = (unsigned __int64)&v36[2 * *(unsigned int *)(v31 + 72)];
        if ( v36 != (unsigned __int64 *)v37 )
        {
          do
          {
            v38 = *v36;
            v36 += 2;
            _libc_free(v38);
          }
          while ( (unsigned __int64 *)v37 != v36 );
          v37 = *(_QWORD *)(v31 + 64);
        }
        if ( v37 != v31 + 80 )
          _libc_free(v37);
        v39 = *(_QWORD *)(v31 + 16);
        if ( v39 != v31 + 32 )
          _libc_free(v39);
        j_j___libc_free_0(v31, 104);
        v30 = (_QWORD *)v98;
      }
      *v4 = (__int64)v30;
      v8 = 0;
    }
    else
    {
      *v4 = 0;
    }
    if ( v97 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v97 + 8LL))(v97);
    if ( (char **)v111 != &v113 )
      j_j___libc_free_0(v111, v113 + 1);
    goto LABEL_13;
  }
  if ( !(unsigned __int8)sub_1C14BC0(
                           ((unsigned __int64)*(unsigned __int8 *)(v10 + 5) << 32) | *(unsigned __int8 *)(v10 + 4),
                           *(unsigned __int8 *)(v10 + 6) | ((unsigned __int64)*(unsigned __int8 *)(v10 + 7) << 32),
                           *(unsigned __int8 *)(v10 + 8) | ((unsigned __int64)*(unsigned __int8 *)(v10 + 9) << 32),
                           *(unsigned __int8 *)(v10 + 10) | ((unsigned __int64)*(unsigned __int8 *)(v10 + 11) << 32),
                           1u) )
  {
    v89 = sub_1C17CB0(*a2, (__int64 *)v8);
    v40 = (__int64 *)sub_22077B0(8);
    v86 = v40;
    if ( v40 )
      sub_1602D10(v40);
    v41 = 0;
    v42 = *(const void **)(*a2 + 16);
    v43 = *(_QWORD *)(*a2 + 8);
    v44 = *(unsigned int *)(v10 + 20);
    v45 = *(_QWORD *)(*a2 + 16) - v43;
    if ( v44 <= v45 )
    {
      v42 = (const void *)(v43 + v44);
      v41 = v45 - v44;
    }
    n = 0;
    if ( *(_DWORD *)(v89 + 228) )
    {
      nb = v41;
      v77 = (_BYTE *)sub_2207820(v41);
      v78 = nb;
      n = v77;
      v85 = v78;
      memcpy(v77, v42, v78);
      v80 = sub_16886D0(*(unsigned int *)(v89 + 228), (__int64)v42, v79);
      sub_16887A0((__int64)v80, n, v85);
      sub_1688720(v80);
      v42 = n;
      v41 = v85;
    }
    sub_16C2450(&v111, (__int64)v42, v41, (__int64)byte_3F871B3, 0);
    v48 = (__m128i *)v111;
    v49 = *(__m128i **)(v111 + 8);
    src = (void *)v111;
    v84 = *(_QWORD *)(v111 + 16);
    v111 = (__int64)v49;
    v112 = v84 - (_QWORD)v49;
    v50 = *(char *(**)())(*(_QWORD *)src + 16LL);
    if ( v50 == sub_12BCB10 )
    {
      v114 = 14;
      v113 = "Unknown buffer";
    }
    else
    {
      v113 = (char *)((__int64 (__fastcall *)(void *))v50)(src);
      v114 = (__int64)v49;
    }
    v81 = v114;
    sub_1509BC0((__int64)&v101, (__int64)v86, (__int64)v49, (__int64)v48, v46, v47, a4, (__m128i *)v111, v112);
    v51 = v102;
    LOBYTE(v102) = v102 & 0xFD;
    if ( (v51 & 1) != 0 )
    {
      v52 = (unsigned __int64)v101;
      v101 = 0;
      v53 = (__m128i *)(v52 & 0xFFFFFFFFFFFFFFFELL);
      v54 = v53;
      if ( v53 )
      {
        v55 = v53->m128i_i64[0];
        v90 = v54;
        v94 = 0;
        v56 = (__int64)&unk_4FA032A;
        v95 = 0;
        v96 = 0;
        v57 = (*(__int64 (__fastcall **)(__m128i *, void *))(v55 + 48))(v54, &unk_4FA032A);
        v58 = v90;
        if ( v57 )
        {
          v97 = 1;
          v82 = (__int64 *)v90[1].m128i_i64[0];
          if ( (__int64 *)v90->m128i_i64[1] != v82 )
          {
            v59 = (__int64 *)v90->m128i_i64[1];
            do
            {
              v100[0] = *v59;
              *v59 = 0;
              sub_1C13FF0(&v99, v100);
              v56 = (__int64)&v111;
              v111 = v97 | 1;
              sub_12BEC00((unsigned __int64 *)&v98, (unsigned __int64 *)&v111, (unsigned __int64 *)&v99);
              v97 = v98 | 1;
              if ( (v111 & 1) != 0 || (v111 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_16BCAE0(&v111, (__int64)&v111, v111);
              if ( (v99 & 1) != 0 || (v99 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_16BCAE0(&v99, (__int64)&v111, v99);
              if ( v100[0] )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v100[0] + 8LL))(v100[0]);
              ++v59;
            }
            while ( v82 != v59 );
            v4 = a1;
            v58 = v90;
          }
          v100[0] = v97 | 1;
          (*(void (__fastcall **)(__m128i *))(v58->m128i_i64[0] + 8))(v58);
        }
        else
        {
          v56 = (__int64)&v111;
          v111 = (__int64)v90;
          sub_1C13FF0(v100, &v111);
          if ( v111 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v111 + 8LL))(v111);
        }
        v100[0] = ((v100[0] & 0xFFFFFFFFFFFFFFFELL) != 0) | v100[0] & 0xFFFFFFFFFFFFFFFELL;
        sub_14ECA90(v100);
        sub_14ECA90(&v96);
        sub_14ECA90(&v95);
        if ( v86 )
        {
          sub_16025D0(v86);
          v56 = 8;
          j_j___libc_free_0(v86, 8);
        }
        (*(void (__fastcall **)(void *))(*(_QWORD *)src + 8LL))(src);
        if ( n )
          j_j___libc_free_0_0(n);
        *v4 = 0;
        sub_14ECA90(&v94);
LABEL_93:
        if ( (v102 & 2) != 0 )
          sub_1264230(&v101, v56, v60);
        v61 = v101;
        if ( (v102 & 1) != 0 )
        {
          if ( v101 )
            ((void (__fastcall *)(_QWORD **))(*v101)[1])(v101);
        }
        else if ( v101 )
        {
          sub_1633490(v101);
          j_j___libc_free_0(v61, 736);
        }
        goto LABEL_15;
      }
      v56 = 0;
    }
    else
    {
      v111 = 0;
      sub_14ECA90(&v111);
      v56 = (__int64)v101;
    }
    v94 = 0;
    sub_14ECA90(&v94);
    v101 = 0;
    v87 = *(unsigned __int16 *)(v10 + 14);
    v62 = sub_22077B0(152);
    v65 = v87;
    v66 = v62;
    if ( v62 )
    {
      sub_1C17480(v62, v56, v89, a3, v87, 1, 1);
      v63 = v81;
    }
    v67 = *(_QWORD *)(v66 + 144);
    *(_DWORD *)(v66 + 40) = *(unsigned __int8 *)(v10 + 4);
    *(_DWORD *)(v66 + 44) = *(unsigned __int8 *)(v10 + 5);
    *(_DWORD *)(v66 + 48) = *(unsigned __int8 *)(v10 + 6);
    *(_DWORD *)(v66 + 52) = *(unsigned __int8 *)(v10 + 7);
    *(_DWORD *)(v66 + 56) = *(unsigned __int8 *)(v10 + 8);
    v68 = *(unsigned __int8 *)(v10 + 9);
    *(_QWORD *)(v66 + 64) = 7;
    *(_DWORD *)(v66 + 60) = v68;
    *(_QWORD *)(v66 + 144) = v8;
    if ( v67 )
    {
      v69 = *(unsigned __int64 **)(v67 + 16);
      v70 = &v69[*(unsigned int *)(v67 + 24)];
      while ( v70 != v69 )
      {
        v71 = *v69++;
        _libc_free(v71);
      }
      v72 = *(_QWORD *)(v67 + 64);
      v73 = (unsigned __int64 *)v72;
      v74 = (unsigned __int64 *)(v72 + 16LL * *(unsigned int *)(v67 + 72));
      if ( (unsigned __int64 *)v72 != v74 )
      {
        do
        {
          v75 = *v73;
          v73 += 2;
          _libc_free(v75);
        }
        while ( v73 != v74 );
        v72 = *(_QWORD *)(v67 + 64);
      }
      if ( v72 != v67 + 80 )
        _libc_free(v72);
      v76 = *(_QWORD *)(v67 + 16);
      if ( v76 != v67 + 32 )
        _libc_free(v76);
      v56 = 104;
      j_j___libc_free_0(v67, 104);
    }
    (*(void (__fastcall **)(void *, __int64, __int64, __int64, __int64))(*(_QWORD *)src + 8LL))(src, v56, v63, v64, v65);
    if ( n )
      j_j___libc_free_0_0(n);
    *v4 = v66;
    v8 = 0;
    goto LABEL_93;
  }
  *a1 = 0;
LABEL_15:
  if ( v8 )
  {
    v12 = *(unsigned __int64 **)(v8 + 16);
    v13 = &v12[*(unsigned int *)(v8 + 24)];
    while ( v13 != v12 )
    {
      v14 = *v12++;
      _libc_free(v14);
    }
    v15 = *(unsigned __int64 **)(v8 + 64);
    v16 = (unsigned __int64)&v15[2 * *(unsigned int *)(v8 + 72)];
    if ( v15 != (unsigned __int64 *)v16 )
    {
      do
      {
        v17 = *v15;
        v15 += 2;
        _libc_free(v17);
      }
      while ( v15 != (unsigned __int64 *)v16 );
      v16 = *(_QWORD *)(v8 + 64);
    }
    if ( v16 != v8 + 80 )
      _libc_free(v16);
    v18 = *(_QWORD *)(v8 + 16);
    if ( v18 != v8 + 32 )
      _libc_free(v18);
    j_j___libc_free_0(v8, 104);
  }
  return v4;
}
