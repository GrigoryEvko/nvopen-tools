// Function: sub_183A180
// Address: 0x183a180
//
__int64 __fastcall sub_183A180(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, __int64),
        __int64 a3,
        unsigned int a4,
        const __m128i *a5,
        __m128 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  int v14; // r8d
  int v15; // r9d
  char v16; // r12
  __int64 v17; // r15
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r14
  unsigned __int64 v22; // rax
  unsigned __int8 v23; // dl
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // r15
  __int64 i; // r13
  __int64 v29; // rdi
  __int64 v30; // r12
  char v31; // r15
  double v32; // xmm4_8
  double v33; // xmm5_8
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  unsigned __int8 v36; // al
  __int64 *v37; // rax
  __int64 v38; // rdx
  _QWORD *v39; // rsi
  _QWORD *v40; // rdx
  __int64 v41; // rdi
  unsigned __int64 v42; // rcx
  __int64 *v43; // rax
  int v44; // r14d
  __int64 v45; // r13
  int v46; // r15d
  unsigned __int64 v47; // rbx
  __int64 *v48; // rax
  __int64 v49; // rax
  __int64 *v50; // rax
  unsigned __int64 v51; // rax
  unsigned __int8 v52; // dl
  unsigned __int64 v53; // rax
  __int64 *v54; // rsi
  __int64 *v55; // rcx
  __int64 v56; // rdx
  int v57; // r8d
  int v58; // r9d
  __int64 v59; // rdx
  char *v60; // rdi
  __int64 v61; // rax
  __int64 v62; // r13
  unsigned __int8 v63; // al
  unsigned __int64 v64; // rsi
  _QWORD *v65; // r13
  __int64 v66; // r14
  _QWORD *v67; // rax
  _QWORD *v68; // r12
  __int64 v69; // rax
  __int64 *v70; // rax
  int v71; // edx
  __int64 *v72; // rdi
  __int64 *v73; // rsi
  __int64 *v74; // rdi
  __int64 *v75; // rdx
  _QWORD *v76; // rdx
  __int64 v77; // [rsp+18h] [rbp-3C8h]
  _BYTE *v78; // [rsp+20h] [rbp-3C0h]
  __int64 v79; // [rsp+28h] [rbp-3B8h]
  __int64 v80; // [rsp+30h] [rbp-3B0h]
  _QWORD *v81; // [rsp+38h] [rbp-3A8h]
  __int64 *v82; // [rsp+40h] [rbp-3A0h]
  __int64 v84; // [rsp+50h] [rbp-390h]
  __int64 *v85; // [rsp+58h] [rbp-388h]
  char v87; // [rsp+6Bh] [rbp-375h]
  __int64 v89; // [rsp+70h] [rbp-370h] BYREF
  __int64 *v90; // [rsp+78h] [rbp-368h]
  __int64 *v91; // [rsp+80h] [rbp-360h]
  __int64 v92; // [rsp+88h] [rbp-358h]
  int v93; // [rsp+90h] [rbp-350h]
  _BYTE v94[72]; // [rsp+98h] [rbp-348h] BYREF
  __int64 v95; // [rsp+E0h] [rbp-300h] BYREF
  __int64 *v96; // [rsp+E8h] [rbp-2F8h]
  __int64 *v97; // [rsp+F0h] [rbp-2F0h]
  __int64 v98; // [rsp+F8h] [rbp-2E8h]
  int v99; // [rsp+100h] [rbp-2E0h]
  _BYTE v100[72]; // [rsp+108h] [rbp-2D8h] BYREF
  __int64 *v101; // [rsp+150h] [rbp-290h] BYREF
  __int64 v102; // [rsp+158h] [rbp-288h]
  _BYTE v103[128]; // [rsp+160h] [rbp-280h] BYREF
  _BYTE *v104; // [rsp+1E0h] [rbp-200h] BYREF
  __int64 v105; // [rsp+1E8h] [rbp-1F8h]
  _BYTE v106[128]; // [rsp+1F0h] [rbp-1F0h] BYREF
  char *v107; // [rsp+270h] [rbp-170h] BYREF
  __int64 v108; // [rsp+278h] [rbp-168h]
  char v109[128]; // [rsp+280h] [rbp-160h] BYREF
  __m128 v110; // [rsp+300h] [rbp-E0h] BYREF
  _QWORD *v111; // [rsp+310h] [rbp-D0h]
  __int64 v112; // [rsp+318h] [rbp-C8h]
  int v113; // [rsp+320h] [rbp-C0h]
  _QWORD v114[23]; // [rsp+328h] [rbp-B8h] BYREF

  if ( (unsigned __int8)sub_1560180(a1 + 112, 18) )
    return 0;
  v16 = 0;
  v17 = 0;
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 <= 1 && !(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 8) )
  {
    v101 = (__int64 *)v103;
    v102 = 0x1000000000LL;
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(a1, 18);
      v18 = *(_QWORD *)(a1 + 88);
      v20 = v18 + 40LL * *(_QWORD *)(a1 + 96);
      if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
      {
        sub_15E08E0(a1, 18);
        v18 = *(_QWORD *)(a1 + 88);
      }
      v19 = (unsigned int)v102;
      if ( v20 != v18 )
        goto LABEL_8;
    }
    else
    {
      v18 = *(_QWORD *)(a1 + 88);
      v19 = 0;
      v20 = v18 + 40LL * *(_QWORD *)(a1 + 96);
      if ( v20 == v18 )
        return v17;
      do
      {
LABEL_8:
        while ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 15 )
        {
          v18 += 40;
          if ( v20 == v18 )
            goto LABEL_12;
        }
        if ( HIDWORD(v102) <= (unsigned int)v19 )
        {
          sub_16CD150((__int64)&v101, v103, 0, 8, v14, v15);
          v19 = (unsigned int)v102;
        }
        v101[v19] = v18;
        v18 += 40;
        v19 = (unsigned int)(v102 + 1);
        LODWORD(v102) = v102 + 1;
      }
      while ( v20 != v18 );
    }
LABEL_12:
    if ( !(_DWORD)v19 )
    {
LABEL_17:
      v17 = 0;
      goto LABEL_18;
    }
    v21 = *(_QWORD *)(a1 + 8);
    while ( v21 )
    {
      v22 = (unsigned __int64)sub_1648700(v21);
      v23 = *(_BYTE *)(v22 + 16);
      if ( v23 <= 0x17u )
        goto LABEL_17;
      if ( v23 == 78 )
      {
        v25 = v22 | 4;
      }
      else
      {
        if ( v23 != 29 )
          goto LABEL_17;
        v25 = v22 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v26 = v25 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_17;
      if ( (v25 & 4) != 0 )
      {
        if ( v21 != v26 - 24 || (*(_WORD *)(v26 + 18) & 3) == 2 )
          goto LABEL_17;
      }
      else if ( v21 != v26 - 72 )
      {
        goto LABEL_17;
      }
      v21 = *(_QWORD *)(v21 + 8);
      if ( a1 == *(_QWORD *)(*(_QWORD *)(v26 + 40) + 56LL) )
        v16 = 1;
    }
    v27 = *(_QWORD *)(a1 + 80);
    for ( i = a1 + 72; i != v27; v27 = *(_QWORD *)(v27 + 8) )
    {
      v29 = v27 - 24;
      if ( !v27 )
        v29 = 0;
      if ( sub_157EBE0(v29) )
        goto LABEL_17;
    }
    v80 = sub_1632FA0(*(_QWORD *)(a1 + 40));
    v89 = 0;
    v81 = (_QWORD *)a2(a3, a1);
    v90 = (__int64 *)v94;
    v91 = (__int64 *)v94;
    v96 = (__int64 *)v100;
    v97 = (__int64 *)v100;
    v92 = 8;
    v93 = 0;
    v95 = 0;
    v98 = 8;
    v99 = 0;
    v82 = &v101[(unsigned int)v102];
    if ( v101 == v82 )
      goto LABEL_150;
    v85 = v101;
    v87 = v16;
    v79 = a1;
LABEL_36:
    v30 = *v85;
    v84 = *(_QWORD *)(*(_QWORD *)*v85 + 24LL);
    if ( (unsigned __int8)sub_15E04F0(*v85) )
    {
      v44 = *(_DWORD *)(v30 + 32);
      sub_15E0F40(v79, v44, 53);
      sub_15E0DF0(v79, v44, 20);
      v45 = *(_QWORD *)(v79 + 8);
      if ( v45 )
      {
        v46 = v44 + 1;
        do
        {
          v51 = (unsigned __int64)sub_1648700(v45);
          v52 = *(_BYTE *)(v51 + 16);
          if ( v52 <= 0x17u )
          {
            v47 = 0;
            goto LABEL_81;
          }
          if ( v52 == 78 )
          {
            v53 = v51 | 4;
          }
          else
          {
            v47 = 0;
            if ( v52 != 29 )
              goto LABEL_81;
            v53 = v51 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v47 = v53 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v53 & 4) != 0 )
          {
            v110.m128_u64[0] = *(_QWORD *)((v53 & 0xFFFFFFFFFFFFFFF8LL) + 56);
            goto LABEL_82;
          }
LABEL_81:
          v110.m128_u64[0] = *(_QWORD *)(v47 + 56);
LABEL_82:
          v48 = (__int64 *)sub_16498A0(v47);
          v49 = sub_1563C10((__int64 *)&v110, v48, v46, 53);
          *(_QWORD *)(v47 + 56) = v49;
          v110.m128_u64[0] = v49;
          v50 = (__int64 *)sub_16498A0(v47);
          *(_QWORD *)(v47 + 56) = sub_1563AB0((__int64 *)&v110, v50, v46, 20);
          v45 = *(_QWORD *)(v45 + 8);
        }
        while ( v45 );
      }
    }
    v31 = sub_15E0450(v30);
    if ( !v31 )
      goto LABEL_38;
    if ( (unsigned __int8)sub_1833160(v84, v80) )
      goto LABEL_65;
    v113 = 0;
    v110.m128_u64[1] = (unsigned __int64)v114;
    v111 = v114;
    v112 = 0x100000010LL;
    v105 = 0x1000000000LL;
    v108 = 0x1000000000LL;
    v114[0] = v30;
    v110.m128_u64[0] = 1;
    v56 = *(_QWORD *)(v30 + 8);
    v104 = v106;
    v107 = v109;
    sub_1832530(&v107, v109, v56, 0);
    LODWORD(v59) = v108;
    if ( (_DWORD)v108 )
    {
      do
      {
        v60 = v107;
        v61 = (unsigned int)v59;
        LODWORD(v59) = v59 - 1;
        v62 = *(_QWORD *)&v107[8 * v61 - 8];
        LODWORD(v108) = v59;
        v63 = *(_BYTE *)(v62 + 16);
        if ( v63 <= 0x17u )
          goto LABEL_98;
        if ( v63 != 56 && v63 != 77 )
        {
          if ( v63 == 55 )
          {
            v69 = (unsigned int)v105;
            if ( (unsigned int)v105 >= HIDWORD(v105) )
            {
              sub_16CD150((__int64)&v104, v106, 0, 8, v57, v58);
              v69 = (unsigned int)v105;
            }
            *(_QWORD *)&v104[8 * v69] = v62;
            LODWORD(v59) = v108;
            LODWORD(v105) = v105 + 1;
          }
          else if ( v63 != 54 )
          {
            goto LABEL_98;
          }
          continue;
        }
        v70 = (__int64 *)v110.m128_u64[1];
        if ( v111 == (_QWORD *)v110.m128_u64[1] )
        {
          v72 = (__int64 *)(v110.m128_u64[1] + 8LL * HIDWORD(v112));
          v57 = HIDWORD(v112);
          if ( (__int64 *)v110.m128_u64[1] != v72 )
          {
            v73 = 0;
            do
            {
              if ( v62 == *v70 )
                goto LABEL_110;
              if ( *v70 == -2 )
                v73 = v70;
              ++v70;
            }
            while ( v72 != v70 );
            if ( v73 )
            {
              *v73 = v62;
              v59 = (unsigned int)v108;
              --v113;
              ++v110.m128_u64[0];
LABEL_127:
              sub_1832530(&v107, &v107[8 * v59], *(_QWORD *)(v62 + 8), 0);
              LODWORD(v59) = v108;
              continue;
            }
          }
          if ( HIDWORD(v112) < (unsigned int)v112 )
          {
            ++HIDWORD(v112);
            *v72 = v62;
            v59 = (unsigned int)v108;
            ++v110.m128_u64[0];
            goto LABEL_127;
          }
        }
        sub_16CCBA0((__int64)&v110, v62);
        v57 = v71;
        v59 = (unsigned int)v108;
        if ( (_BYTE)v57 )
          goto LABEL_127;
LABEL_110:
        ;
      }
      while ( (_DWORD)v59 );
    }
    v64 = (unsigned __int64)v111;
    v78 = &v104[8 * (unsigned int)v105];
    if ( v104 == v78 )
      goto LABEL_124;
    v77 = v30;
    v65 = v104;
    while ( 1 )
    {
      v66 = *(_QWORD *)(*v65 - 48LL);
      v67 = (_QWORD *)v110.m128_u64[1];
      if ( v64 == v110.m128_u64[1] )
        break;
      v68 = (_QWORD *)(v64 + 8LL * (unsigned int)v112);
      v67 = sub_16CC9F0((__int64)&v110, *(_QWORD *)(*v65 - 48LL));
      if ( v66 == *v67 )
      {
        v64 = (unsigned __int64)v111;
        if ( v111 == (_QWORD *)v110.m128_u64[1] )
          v76 = &v111[HIDWORD(v112)];
        else
          v76 = &v111[(unsigned int)v112];
        goto LABEL_133;
      }
      v64 = (unsigned __int64)v111;
      if ( v111 == (_QWORD *)v110.m128_u64[1] )
      {
        v67 = &v111[HIDWORD(v112)];
        v76 = v67;
        goto LABEL_133;
      }
      v67 = &v111[(unsigned int)v112];
LABEL_117:
      if ( v68 != v67 )
      {
        v30 = v77;
        v60 = v107;
        goto LABEL_98;
      }
      if ( v78 == (_BYTE *)++v65 )
      {
        v30 = v77;
LABEL_124:
        v60 = v107;
        v31 = 0;
LABEL_98:
        if ( v60 != v109 )
          _libc_free((unsigned __int64)v60);
        if ( v104 != v106 )
          _libc_free((unsigned __int64)v104);
        if ( v111 != (_QWORD *)v110.m128_u64[1] )
          _libc_free((unsigned __int64)v111);
        if ( v31 )
        {
LABEL_38:
          if ( !v87 )
            goto LABEL_61;
          if ( *(_BYTE *)(v84 + 8) != 13 )
            goto LABEL_61;
          v34 = *(_QWORD **)(v84 + 16);
          v35 = &v34[*(unsigned int *)(v84 + 12)];
          if ( v35 == v34 )
            goto LABEL_61;
LABEL_43:
          while ( *v34 != *(_QWORD *)v30 )
          {
            if ( v35 == ++v34 )
              goto LABEL_61;
          }
          goto LABEL_44;
        }
LABEL_65:
        if ( *(_BYTE *)(v84 + 8) != 13 )
          goto LABEL_61;
        v38 = *(unsigned int *)(v84 + 12);
        if ( !a4 || a4 >= (unsigned int)v38 )
        {
          v34 = *(_QWORD **)(v84 + 16);
          v39 = &v34[v38];
          if ( v39 == v34 )
          {
LABEL_75:
            v43 = v96;
            if ( v97 == v96 )
            {
              v74 = &v96[HIDWORD(v98)];
              if ( v96 != v74 )
              {
                v75 = 0;
                while ( v30 != *v43 )
                {
                  if ( *v43 == -2 )
                    v75 = v43;
                  if ( v74 == ++v43 )
                  {
                    if ( !v75 )
                      goto LABEL_166;
                    *v75 = v30;
                    --v99;
                    ++v95;
                    goto LABEL_44;
                  }
                }
                goto LABEL_44;
              }
LABEL_166:
              if ( HIDWORD(v98) < (unsigned int)v98 )
              {
                ++HIDWORD(v98);
                *v74 = v30;
                ++v95;
                goto LABEL_44;
              }
            }
            sub_16CCBA0((__int64)&v95, v30);
            goto LABEL_44;
          }
          v40 = *(_QWORD **)(v84 + 16);
          v41 = 100990;
          while ( 1 )
          {
            v42 = *(unsigned __int8 *)(*v40 + 8LL);
            if ( (unsigned __int8)v42 > 0x10u || !_bittest64(&v41, v42) )
              break;
            if ( v39 == ++v40 )
              goto LABEL_75;
          }
          if ( v87 )
          {
            v35 = &v34[*(unsigned int *)(v84 + 12)];
            goto LABEL_43;
          }
LABEL_61:
          v36 = sub_15E0300(v30);
          if ( (unsigned __int8)sub_1839110(v30, v36, v81, a4) )
          {
            v37 = v90;
            if ( v91 != v90 )
              goto LABEL_63;
            v54 = &v90[HIDWORD(v92)];
            if ( v90 == v54 )
            {
LABEL_151:
              if ( HIDWORD(v92) >= (unsigned int)v92 )
              {
LABEL_63:
                sub_16CCBA0((__int64)&v89, v30);
              }
              else
              {
                ++HIDWORD(v92);
                *v54 = v30;
                ++v89;
              }
            }
            else
            {
              v55 = 0;
              while ( v30 != *v37 )
              {
                if ( *v37 == -2 )
                  v55 = v37;
                if ( v54 == ++v37 )
                {
                  if ( !v55 )
                    goto LABEL_151;
                  *v55 = v30;
                  --v93;
                  ++v89;
                  break;
                }
              }
            }
          }
        }
LABEL_44:
        if ( v82 == ++v85 )
        {
          if ( HIDWORD(v92) == v93 && v99 == HIDWORD(v98) )
          {
LABEL_150:
            v17 = 0;
          }
          else
          {
            LOBYTE(v111) = a5[1].m128i_i8[0];
            if ( (_BYTE)v111 )
            {
              a6 = (__m128)_mm_loadu_si128(a5);
              v110 = a6;
            }
            v17 = sub_1834E90(v79, (__int64)&v89, (__int64)&v95, (__int64)&v110, a6, a7, a8, a9, v32, v33, a12, a13);
          }
          if ( v97 != v96 )
            _libc_free((unsigned __int64)v97);
          if ( v91 != v90 )
            _libc_free((unsigned __int64)v91);
LABEL_18:
          if ( v101 != (__int64 *)v103 )
            _libc_free((unsigned __int64)v101);
          return v17;
        }
        goto LABEL_36;
      }
    }
    v68 = (_QWORD *)(v64 + 8LL * HIDWORD(v112));
    if ( (_QWORD *)v64 == v68 )
    {
      v76 = (_QWORD *)v64;
    }
    else
    {
      do
      {
        if ( v66 == *v67 )
          break;
        ++v67;
      }
      while ( v68 != v67 );
      v76 = (_QWORD *)(v64 + 8LL * HIDWORD(v112));
    }
LABEL_133:
    while ( v76 != v67 )
    {
      if ( *v67 < 0xFFFFFFFFFFFFFFFELL )
        break;
      ++v67;
    }
    goto LABEL_117;
  }
  return v17;
}
