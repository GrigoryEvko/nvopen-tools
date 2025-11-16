// Function: sub_CE0430
// Address: 0xce0430
//
__int64 *__fastcall sub_CE0430(__int64 *a1, __int64 *a2, _QWORD **a3, __m128i a4)
{
  __int64 *v4; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rsi
  char v12; // al
  __int64 *v13; // r14
  __int64 *v14; // rbx
  __int64 k; // rax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 *v18; // rbx
  __int64 *v19; // r13
  __int64 v20; // rdi
  __int64 v21; // rdi
  int v22; // r14d
  unsigned __int64 v23; // r9
  size_t v24; // rax
  unsigned int v25; // edx
  __int16 v26; // si
  __int64 v27; // rcx
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // r13
  unsigned __int64 v31; // rdx
  __int64 *v32; // r15
  __int64 *v33; // rbx
  __int64 i; // rdx
  __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 *v39; // rbx
  __int64 *v40; // r15
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int64 *v43; // rax
  __int64 v44; // r10
  char *v45; // rsi
  __int64 v46; // rdx
  char *v47; // rax
  char *v48; // rcx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  char v52; // al
  unsigned __int64 v53; // rsi
  _QWORD *v54; // rsi
  __int64 v55; // rax
  _QWORD *v56; // rdi
  char v57; // al
  _QWORD *v58; // r8
  __int64 *v59; // rax
  __int64 *v60; // r12
  __int64 *v61; // r13
  __int64 v62; // rdx
  __int64 v63; // rcx
  _QWORD **v64; // r13
  __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // rbx
  __int64 v68; // r13
  int v69; // eax
  __int64 *v70; // r15
  __int64 *v71; // r14
  __int64 j; // rdx
  __int64 v73; // rdi
  unsigned int v74; // ecx
  __int64 *v75; // rdi
  __int64 *v76; // r15
  __int64 *v77; // r14
  __int64 v78; // rdi
  __int64 v79; // rdi
  void *v80; // rax
  size_t v81; // r10
  __int64 v82; // rbx
  unsigned __int64 v83; // [rsp-8h] [rbp-3C8h]
  __int64 *v84; // [rsp+8h] [rbp-3B8h]
  __int64 v85; // [rsp+20h] [rbp-3A0h]
  size_t v86; // [rsp+20h] [rbp-3A0h]
  __int64 *v87; // [rsp+28h] [rbp-398h]
  _QWORD **v88; // [rsp+28h] [rbp-398h]
  unsigned int v89; // [rsp+28h] [rbp-398h]
  size_t n; // [rsp+30h] [rbp-390h]
  size_t na; // [rsp+30h] [rbp-390h]
  size_t nb; // [rsp+30h] [rbp-390h]
  __int64 v93; // [rsp+38h] [rbp-388h]
  _QWORD *v94; // [rsp+38h] [rbp-388h]
  __int64 v95; // [rsp+40h] [rbp-380h] BYREF
  __int64 v96; // [rsp+48h] [rbp-378h] BYREF
  __int64 v97; // [rsp+50h] [rbp-370h] BYREF
  __int64 v98; // [rsp+58h] [rbp-368h] BYREF
  __int64 v99; // [rsp+60h] [rbp-360h] BYREF
  unsigned __int64 v100; // [rsp+68h] [rbp-358h] BYREF
  _QWORD **v101; // [rsp+70h] [rbp-350h] BYREF
  __int64 v102; // [rsp+78h] [rbp-348h]
  __int64 v103; // [rsp+80h] [rbp-340h] BYREF
  unsigned __int64 v104; // [rsp+88h] [rbp-338h]
  unsigned __int64 v105; // [rsp+90h] [rbp-330h]
  unsigned __int64 v106; // [rsp+98h] [rbp-328h]
  int v107; // [rsp+A0h] [rbp-320h]
  _QWORD *v108; // [rsp+A8h] [rbp-318h] BYREF
  unsigned __int64 v109; // [rsp+B0h] [rbp-310h]
  _QWORD v110[2]; // [rsp+B8h] [rbp-308h] BYREF
  unsigned __int8 v111; // [rsp+C8h] [rbp-2F8h]
  __int64 v112; // [rsp+D0h] [rbp-2F0h]
  __int64 v113; // [rsp+E0h] [rbp-2E0h] BYREF
  __int64 v114; // [rsp+E8h] [rbp-2D8h]
  _QWORD v115[90]; // [rsp+F0h] [rbp-2D0h] BYREF

  v4 = a1;
  if ( !a3 || *(_QWORD *)(*a2 + 16) - *(_QWORD *)(*a2 + 8) <= 3u )
  {
    *a1 = 0;
    return v4;
  }
  v8 = sub_22077B0(96);
  if ( v8 )
  {
    memset((void *)v8, 0, 0x60u);
    *(_QWORD *)(v8 + 88) = 1;
    *(_QWORD *)(v8 + 16) = v8 + 32;
    *(_QWORD *)(v8 + 24) = 0x400000000LL;
    *(_QWORD *)(v8 + 64) = v8 + 80;
  }
  v9 = *a2;
  v10 = *(_QWORD *)(*a2 + 8);
  if ( *(_DWORD *)v10 != 2135835629 )
  {
    v103 = 0;
    v108 = v110;
    v104 = 0;
    v105 = 0;
    v106 = 0;
    v109 = 0;
    LOBYTE(v110[0]) = 0;
    v101 = a3;
    v102 = v8;
    sub_CB0A90(
      (__int64)&v113,
      *(_QWORD *)(v9 + 8),
      *(_QWORD *)(v9 + 16) - *(_QWORD *)(v9 + 8),
      0,
      (__int64)nullsub_179,
      (__int64)&v101);
    sub_CB0A80((__int64)&v113, (__int64)&v101);
    sub_CB4D10((__int64)&v113, (__int64)&v101);
    sub_CB0300((__int64)&v113);
    sub_CDCA30((__int64)&v113, (__int64)&v103);
    sub_CB1A30((__int64)&v113);
    v11 = v104;
    v12 = sub_CCD5F0(v103, v104, v105, v106, v111);
    if ( v111 )
      v106 = 20;
    if ( v12 )
    {
      sub_CB34B0((__int64)&v113, v11);
    }
    else
    {
      v22 = sub_CB0000((__int64)&v113);
      sub_CB34B0((__int64)&v113, v11);
      if ( !v22 )
      {
        v114 = 0;
        v113 = (__int64)v115;
        LOBYTE(v115[0]) = 0;
        if ( v111 )
        {
          v11 = v109;
          if ( (v109 & 1) != 0 )
            goto LABEL_12;
          n = v109 & 1;
          sub_22410F0(&v113, v109 >> 1, 0);
          v23 = v109 >> 1;
          if ( (unsigned int)(v109 >> 1) )
          {
            v24 = n;
            v25 = 0;
            do
            {
              v26 = 16 * word_3F64060[*((unsigned __int8 *)v108 + v25)];
              v27 = v25 + 1;
              v25 += 2;
              *(_BYTE *)(v113 + v24++) = v26 | LOBYTE(word_3F64060[*((unsigned __int8 *)v108 + v27)]);
            }
            while ( (unsigned int)v23 != v24 );
          }
        }
        else
        {
          sub_2240AE0(&v113, &v108);
        }
        sub_C7DA90(&v98, v113, v114, byte_3F871B3, 0, 0);
        v28 = (_QWORD *)v98;
        v11 = (__int64)&v100;
        v98 = 0;
        v100 = (unsigned __int64)v28;
        sub_CD02C0(&v99, (_QWORD **)&v100, v112, (__int64)a3, v107);
        if ( v100 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v100 + 8LL))(v100);
        v29 = (_QWORD *)v99;
        if ( v99 )
        {
          v30 = *(_QWORD *)(v99 + 144);
          *(_QWORD *)(v99 + 40) = v103;
          v29[6] = v104;
          v29[7] = v105;
          v31 = v106;
          v29[18] = v8;
          v29[8] = v31;
          if ( v30 )
          {
            v32 = *(__int64 **)(v30 + 16);
            v33 = &v32[*(unsigned int *)(v30 + 24)];
            if ( v32 != v33 )
            {
              for ( i = *(_QWORD *)(v30 + 16); ; i = *(_QWORD *)(v30 + 16) )
              {
                v35 = *v32;
                v36 = (unsigned int)(((__int64)v32 - i) >> 3) >> 7;
                v11 = 4096LL << v36;
                if ( v36 >= 0x1E )
                  v11 = 0x40000000000LL;
                ++v32;
                sub_C7D6A0(v35, v11, 16);
                if ( v33 == v32 )
                  break;
              }
            }
            v37 = *(_QWORD *)(v30 + 64);
            v38 = 16LL * *(unsigned int *)(v30 + 72);
            v39 = (__int64 *)(v37 + v38);
            if ( v37 != v37 + v38 )
            {
              v40 = *(__int64 **)(v30 + 64);
              do
              {
                v11 = v40[1];
                v41 = *v40;
                v40 += 2;
                sub_C7D6A0(v41, v11, 16);
              }
              while ( v39 != v40 );
              v39 = *(__int64 **)(v30 + 64);
            }
            if ( v39 != (__int64 *)(v30 + 80) )
              _libc_free(v39, v11);
            v42 = *(_QWORD *)(v30 + 16);
            if ( v42 != v30 + 32 )
              _libc_free(v42, v11);
            v11 = 96;
            j_j___libc_free_0(v30, 96);
            v29 = (_QWORD *)v99;
          }
          *v4 = (__int64)v29;
          v8 = 0;
        }
        else
        {
          *a1 = 0;
        }
        if ( v98 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v98 + 8LL))(v98);
        if ( (_QWORD *)v113 != v115 )
        {
          v11 = v115[0] + 1LL;
          j_j___libc_free_0(v113, v115[0] + 1LL);
        }
        goto LABEL_13;
      }
    }
LABEL_12:
    *a1 = 0;
LABEL_13:
    if ( v108 != v110 )
    {
      v11 = v110[0] + 1LL;
      j_j___libc_free_0(v108, v110[0] + 1LL);
    }
    goto LABEL_15;
  }
  v11 = *(unsigned __int8 *)(v10 + 6) | ((unsigned __int64)*(unsigned __int8 *)(v10 + 7) << 32);
  if ( !(unsigned __int8)sub_CCD5F0(
                           ((unsigned __int64)*(unsigned __int8 *)(v10 + 5) << 32) | *(unsigned __int8 *)(v10 + 4),
                           v11,
                           *(unsigned __int8 *)(v10 + 8) | ((unsigned __int64)*(unsigned __int8 *)(v10 + 9) << 32),
                           *(unsigned __int8 *)(v10 + 10) | ((unsigned __int64)*(unsigned __int8 *)(v10 + 11) << 32),
                           1u) )
  {
    v93 = sub_CD1D80(*a2, (__int64 *)v8);
    v43 = (__int64 *)sub_22077B0(8);
    v87 = v43;
    if ( v43 )
      sub_B6EEA0(v43);
    v44 = 0;
    v45 = *(char **)(*a2 + 16);
    v46 = *(_QWORD *)(*a2 + 8);
    v47 = (char *)*(unsigned int *)(v10 + 20);
    v48 = &v45[-v46];
    if ( v47 <= &v45[-v46] )
    {
      v45 = &v47[v46];
      v44 = v48 - v47;
    }
    na = 0;
    if ( *(_DWORD *)(v93 + 228) )
    {
      nb = v44;
      v80 = (void *)sub_2207820(v44);
      v81 = nb;
      na = (size_t)v80;
      v86 = v81;
      memcpy(v80, v45, v81);
      v82 = sub_16886D0(*(unsigned int *)(v93 + 228));
      sub_16887A0(v82, na, (unsigned int)v86);
      sub_1688720(v82);
      v45 = (char *)na;
      v44 = v86;
    }
    sub_C7DA90(&v113, (__int64)v45, v44, byte_3F871B3, 0, 0);
    memset(&v115[2], 0, 0x58u);
    v85 = v113;
    sub_C7EC60(&v103, (_QWORD *)v113);
    v83 = v106;
    sub_A01950((__int64)&v101, (__int64)v87, (__int64)&v113, v49, v50, v51, a4, (const __m128i *)v103, v104);
    if ( LOBYTE(v115[12]) )
    {
      LOBYTE(v115[12]) = 0;
      if ( v115[10] )
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v115[10])(&v115[8], &v115[8], 3);
    }
    if ( LOBYTE(v115[7]) )
    {
      LOBYTE(v115[7]) = 0;
      if ( v115[5] )
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v115[5])(&v115[3], &v115[3], 3);
    }
    if ( LOBYTE(v115[2]) )
    {
      LOBYTE(v115[2]) = 0;
      if ( v115[0] )
        ((void (__fastcall *)(__int64 *, __int64 *, __int64))v115[0])(&v113, &v113, 3);
    }
    v52 = v102;
    v53 = (unsigned __int64)v101;
    LOBYTE(v102) = v102 & 0xFD;
    if ( (v52 & 1) != 0 )
    {
      v101 = 0;
      v54 = (_QWORD *)(v53 & 0xFFFFFFFFFFFFFFFELL);
      if ( v54 )
      {
        v55 = *v54;
        v56 = v54;
        v94 = v54;
        v95 = 0;
        v11 = (__int64)&unk_4F84052;
        v96 = 0;
        v97 = 0;
        v57 = (*(__int64 (__fastcall **)(_QWORD *, void *))(v55 + 48))(v56, &unk_4F84052);
        v58 = v94;
        if ( v57 )
        {
          v98 = 1;
          v84 = (__int64 *)v94[2];
          if ( (__int64 *)v94[1] != v84 )
          {
            v59 = v4;
            v60 = (__int64 *)v94[1];
            v61 = v59;
            do
            {
              v103 = *v60;
              *v60 = 0;
              sub_CCD240((__int64 *)&v100, &v103);
              v11 = (__int64)&v113;
              v113 = v98 | 1;
              sub_9CDB40((unsigned __int64 *)&v99, (unsigned __int64 *)&v113, &v100);
              v98 = v99 | 1;
              if ( (v113 & 1) != 0 || (v113 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v113, (__int64)&v113);
              if ( (v100 & 1) != 0 || (v100 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v100, (__int64)&v113);
              if ( v103 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v103 + 8LL))(v103);
              ++v60;
            }
            while ( v84 != v60 );
            v58 = v94;
            v4 = v61;
          }
          v103 = v98 | 1;
          (*(void (__fastcall **)(_QWORD *))(*v58 + 8LL))(v58);
        }
        else
        {
          v11 = (__int64)&v113;
          v113 = (__int64)v94;
          sub_CCD240(&v103, &v113);
          if ( v113 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v113 + 8LL))(v113);
        }
        if ( (v103 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          BUG();
        v103 = 0;
        sub_9C66B0(&v103);
        sub_9C66B0(&v97);
        sub_9C66B0(&v96);
        if ( v87 )
        {
          sub_B6E710(v87);
          v11 = 8;
          j_j___libc_free_0(v87, 8);
        }
        if ( v85 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v85 + 8LL))(v85);
        if ( na )
          j_j___libc_free_0_0(na);
        *v4 = 0;
        sub_9C66B0(&v95);
LABEL_93:
        if ( (v102 & 2) != 0 )
          sub_904700(&v101);
        v64 = v101;
        if ( (v102 & 1) != 0 )
        {
          if ( v101 )
            ((void (__fastcall *)(_QWORD **))(*v101)[1])(v101);
        }
        else if ( v101 )
        {
          sub_BA9C10(v101, v11, v62, v63);
          v11 = 880;
          j_j___libc_free_0(v64, 880);
        }
        goto LABEL_15;
      }
      v11 = 0;
    }
    else
    {
      v88 = v101;
      v113 = 0;
      sub_9C66B0(&v113);
      v11 = (__int64)v88;
    }
    v95 = 0;
    sub_9C66B0(&v95);
    v101 = 0;
    v89 = *(unsigned __int16 *)(v10 + 14);
    v65 = sub_22077B0(152);
    v66 = v89;
    v67 = v65;
    if ( v65 )
    {
      sub_CCFFF0(v65, (__int64 *)v11, v93, (__int64)a3, v89, 1, 1);
      v62 = v83;
    }
    v68 = *(_QWORD *)(v67 + 144);
    *(_DWORD *)(v67 + 40) = *(unsigned __int8 *)(v10 + 4);
    *(_DWORD *)(v67 + 44) = *(unsigned __int8 *)(v10 + 5);
    *(_DWORD *)(v67 + 48) = *(unsigned __int8 *)(v10 + 6);
    *(_DWORD *)(v67 + 52) = *(unsigned __int8 *)(v10 + 7);
    *(_DWORD *)(v67 + 56) = *(unsigned __int8 *)(v10 + 8);
    v69 = *(unsigned __int8 *)(v10 + 9);
    *(_QWORD *)(v67 + 64) = 20;
    *(_DWORD *)(v67 + 60) = v69;
    *(_QWORD *)(v67 + 144) = v8;
    if ( v68 )
    {
      v70 = *(__int64 **)(v68 + 16);
      v71 = &v70[*(unsigned int *)(v68 + 24)];
      if ( v70 != v71 )
      {
        for ( j = *(_QWORD *)(v68 + 16); ; j = *(_QWORD *)(v68 + 16) )
        {
          v73 = *v70;
          v74 = (unsigned int)(((__int64)v70 - j) >> 3) >> 7;
          v11 = 4096LL << v74;
          if ( v74 >= 0x1E )
            v11 = 0x40000000000LL;
          ++v70;
          sub_C7D6A0(v73, v11, 16);
          if ( v71 == v70 )
            break;
        }
      }
      v75 = *(__int64 **)(v68 + 64);
      v76 = v75;
      v77 = &v75[2 * *(unsigned int *)(v68 + 72)];
      if ( v75 != v77 )
      {
        do
        {
          v11 = v76[1];
          v78 = *v76;
          v76 += 2;
          sub_C7D6A0(v78, v11, 16);
        }
        while ( v77 != v76 );
        v75 = *(__int64 **)(v68 + 64);
      }
      if ( v75 != (__int64 *)(v68 + 80) )
        _libc_free(v75, v11);
      v79 = *(_QWORD *)(v68 + 16);
      if ( v79 != v68 + 32 )
        _libc_free(v79, v11);
      v11 = 96;
      j_j___libc_free_0(v68, 96);
    }
    if ( v85 )
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v85 + 8LL))(
        v85,
        v11,
        v62,
        v63,
        v66);
    if ( na )
      j_j___libc_free_0_0(na);
    *v4 = v67;
    v8 = 0;
    goto LABEL_93;
  }
  *a1 = 0;
LABEL_15:
  if ( v8 )
  {
    v13 = *(__int64 **)(v8 + 16);
    v14 = &v13[*(unsigned int *)(v8 + 24)];
    if ( v13 != v14 )
    {
      for ( k = *(_QWORD *)(v8 + 16); ; k = *(_QWORD *)(v8 + 16) )
      {
        v16 = *v13;
        v17 = (unsigned int)(((__int64)v13 - k) >> 3) >> 7;
        v11 = 4096LL << v17;
        if ( v17 >= 0x1E )
          v11 = 0x40000000000LL;
        ++v13;
        sub_C7D6A0(v16, v11, 16);
        if ( v14 == v13 )
          break;
      }
    }
    v18 = *(__int64 **)(v8 + 64);
    v19 = &v18[2 * *(unsigned int *)(v8 + 72)];
    if ( v18 != v19 )
    {
      do
      {
        v11 = v18[1];
        v20 = *v18;
        v18 += 2;
        sub_C7D6A0(v20, v11, 16);
      }
      while ( v19 != v18 );
      v19 = *(__int64 **)(v8 + 64);
    }
    if ( v19 != (__int64 *)(v8 + 80) )
      _libc_free(v19, v11);
    v21 = *(_QWORD *)(v8 + 16);
    if ( v21 != v8 + 32 )
      _libc_free(v21, v11);
    j_j___libc_free_0(v8, 96);
  }
  return v4;
}
