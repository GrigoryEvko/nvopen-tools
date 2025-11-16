// Function: sub_25F0930
// Address: 0x25f0930
//
__int64 __fastcall sub_25F0930(__int64 a1, __int64 a2, __int64 a3, __int64 **a4)
{
  int v6; // r15d
  unsigned __int64 v7; // rbx
  __m128i *v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r8
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __int64 v16; // rdx
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64); // rax
  __int64 v18; // r9
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  int v21; // ecx
  unsigned __int64 v22; // rax
  bool v23; // cf
  __int64 v24; // rax
  __int64 v25; // r11
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // r8
  unsigned __int8 **v30; // rsi
  int v31; // ecx
  unsigned __int8 **v32; // r10
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  int v35; // edx
  bool v36; // of
  __int64 v37; // rsi
  __int64 *v38; // r12
  __int64 v39; // rax
  __int64 *v40; // r15
  __int64 *v41; // r14
  unsigned __int64 v42; // rdx
  __int64 v43; // rbx
  unsigned int i; // r14d
  __int64 *v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  _QWORD *v49; // rax
  __int64 *v50; // rax
  __int64 v51; // r13
  __int64 v52; // rdi
  int v53; // eax
  __int32 v54; // edx
  unsigned int v55; // r12d
  __int64 *v57; // r14
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // r9
  int v62; // ebx
  __int64 *v63; // rsi
  __int64 *v64; // r11
  __int64 v65; // r14
  int v66; // r13d
  _QWORD *v67; // r8
  __int64 v68; // r8
  _QWORD *v69; // r10
  __int64 v70; // rax
  __int64 *v71; // rax
  __int64 v73; // [rsp+18h] [rbp-248h]
  int v74; // [rsp+18h] [rbp-248h]
  int v75; // [rsp+30h] [rbp-230h]
  __int64 v76; // [rsp+30h] [rbp-230h]
  signed __int64 v77; // [rsp+38h] [rbp-228h]
  __int64 v78; // [rsp+38h] [rbp-228h]
  unsigned int v79; // [rsp+40h] [rbp-220h]
  __int64 v80; // [rsp+40h] [rbp-220h]
  __int64 *v81; // [rsp+48h] [rbp-218h]
  int v82; // [rsp+48h] [rbp-218h]
  int v83; // [rsp+50h] [rbp-210h]
  int v84; // [rsp+58h] [rbp-208h]
  char v85; // [rsp+58h] [rbp-208h]
  __int64 v86; // [rsp+58h] [rbp-208h]
  __int64 v87; // [rsp+60h] [rbp-200h]
  __int64 *v88; // [rsp+60h] [rbp-200h]
  __int64 v89; // [rsp+60h] [rbp-200h]
  __int64 *v90; // [rsp+60h] [rbp-200h]
  __int64 v91; // [rsp+68h] [rbp-1F8h]
  unsigned __int64 v92; // [rsp+68h] [rbp-1F8h]
  int v93; // [rsp+68h] [rbp-1F8h]
  int v94; // [rsp+68h] [rbp-1F8h]
  __int64 v95; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v96; // [rsp+78h] [rbp-1E8h]
  __int64 v97; // [rsp+80h] [rbp-1E0h]
  __int64 v98; // [rsp+88h] [rbp-1D8h]
  __int64 *v99; // [rsp+90h] [rbp-1D0h]
  __int64 v100; // [rsp+98h] [rbp-1C8h]
  __int64 v101; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v102; // [rsp+A8h] [rbp-1B8h]
  __int64 v103; // [rsp+B0h] [rbp-1B0h]
  __int64 v104; // [rsp+B8h] [rbp-1A8h]
  __int64 *v105; // [rsp+C0h] [rbp-1A0h]
  __int64 v106; // [rsp+C8h] [rbp-198h]
  __int64 v107; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v108; // [rsp+D8h] [rbp-188h]
  __int64 v109; // [rsp+E0h] [rbp-180h]
  __int64 v110; // [rsp+E8h] [rbp-178h]
  unsigned __int64 *v111; // [rsp+F0h] [rbp-170h]
  __int64 v112; // [rsp+F8h] [rbp-168h]
  unsigned __int8 **v113; // [rsp+100h] [rbp-160h] BYREF
  __int64 v114; // [rsp+108h] [rbp-158h]
  _BYTE v115[32]; // [rsp+110h] [rbp-150h] BYREF
  __m128i v116; // [rsp+130h] [rbp-130h]
  __m128i v117; // [rsp+140h] [rbp-120h]
  _BYTE v118[16]; // [rsp+150h] [rbp-110h] BYREF
  void (__fastcall *v119)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64); // [rsp+160h] [rbp-100h]
  unsigned __int8 (__fastcall *v120)(_BYTE *, __int64, __int64, __int64, __int64, __int64); // [rsp+168h] [rbp-F8h]
  _OWORD v121[2]; // [rsp+170h] [rbp-F0h] BYREF
  _BYTE v122[16]; // [rsp+190h] [rbp-D0h] BYREF
  void (__fastcall *v123)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64); // [rsp+1A0h] [rbp-C0h]
  __int64 v124; // [rsp+1A8h] [rbp-B8h]
  __m128i v125; // [rsp+1B0h] [rbp-B0h] BYREF
  __m128i v126; // [rsp+1C0h] [rbp-A0h] BYREF
  _BYTE v127[16]; // [rsp+1D0h] [rbp-90h] BYREF
  void (__fastcall *v128)(_BYTE *, _BYTE *, __int64); // [rsp+1E0h] [rbp-80h]
  unsigned __int8 (__fastcall *v129)(_BYTE *, __int64, __int64, __int64, __int64, __int64); // [rsp+1E8h] [rbp-78h]
  __m128i v130; // [rsp+1F0h] [rbp-70h] BYREF
  __m128i v131; // [rsp+200h] [rbp-60h] BYREF
  _BYTE v132[16]; // [rsp+210h] [rbp-50h] BYREF
  void (__fastcall *v133)(_BYTE *, _BYTE *, __int64); // [rsp+220h] [rbp-40h]
  __int64 v134; // [rsp+228h] [rbp-38h]

  v111 = (unsigned __int64 *)&v113;
  v99 = &v101;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = &v107;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v112 = 0;
  sub_29B2CD0(a2, &v95, &v101, &v107, 0);
  v79 = *(_DWORD *)(a3 + 8);
  v73 = *(_QWORD *)a3 + 8LL * v79;
  if ( *(_QWORD *)a3 != v73 )
  {
    v81 = *(__int64 **)a3;
    v6 = 0;
    v7 = 0;
    while ( 1 )
    {
      v8 = &v125;
      v9 = *v81;
      sub_AA72C0(&v125, *v81, 1);
      v12 = _mm_loadu_si128(&v125);
      v119 = 0;
      v13 = _mm_loadu_si128(&v126);
      v116 = v12;
      v117 = v13;
      if ( v128 )
      {
        v8 = (__m128i *)v118;
        v128(v118, v127, 2);
        v120 = v129;
        v119 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64))v128;
      }
      v14 = _mm_loadu_si128(&v130);
      v15 = _mm_loadu_si128(&v131);
      v123 = 0;
      v121[0] = v14;
      v121[1] = v15;
      if ( v133 )
      {
        v8 = (__m128i *)v122;
        v133(v122, v132, 2);
        v16 = v116.m128i_i64[0];
        v124 = v134;
        v17 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64))v133;
        v18 = v116.m128i_i64[0];
        v123 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64))v133;
        if ( *(_QWORD *)&v121[0] != v116.m128i_i64[0] )
          break;
LABEL_38:
        if ( v17 )
          v17(v122, v122, 3, v10, v11, v18);
        goto LABEL_40;
      }
      v16 = v116.m128i_i64[0];
      v18 = v116.m128i_i64[0];
      if ( v116.m128i_i64[0] != *(_QWORD *)&v121[0] )
        break;
LABEL_40:
      if ( v119 )
        v119(v118, v118, 3, v10, v11, v18);
      if ( v133 )
        ((void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64))v133)(v132, v132, 3, v10, v11, v18);
      if ( v128 )
        ((void (__fastcall *)(_BYTE *, _BYTE *, __int64, __int64, __int64, __int64))v128)(v127, v127, 3, v10, v11, v18);
      if ( (__int64 *)v73 == ++v81 )
      {
        v77 = v7;
        v74 = v6;
        v83 = v106;
        v75 = v100;
        v82 = qword_4FF1CA8;
        if ( (int)qword_4FF1CA8 <= 0 )
          goto LABEL_79;
        v125.m128i_i64[0] = 0;
        v126.m128i_i64[0] = 2;
        v38 = *(__int64 **)a3;
        v39 = *(unsigned int *)(a3 + 8);
        v126.m128i_i32[2] = 0;
        v126.m128i_i8[12] = 1;
        v79 = v39;
        v40 = &v38[v39];
        v125.m128i_i64[1] = (__int64)v127;
        if ( v38 != v40 )
        {
          v85 = 1;
          v41 = v38;
          while ( 1 )
          {
            while ( 1 )
            {
              v42 = *(_QWORD *)(*v41 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v42 == *v41 + 48 )
                goto LABEL_133;
              if ( !v42 )
                BUG();
              v92 = *(_QWORD *)(*v41 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              v43 = v42 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
                goto LABEL_133;
              if ( (unsigned int)sub_B46E30(v42 - 24) )
                break;
              v85 &= *(_BYTE *)(v92 - 24) == 36;
LABEL_51:
              if ( v40 == ++v41 )
                goto LABEL_68;
            }
            v93 = sub_B46E30(v43);
            if ( !v93 )
              goto LABEL_51;
            v88 = v41;
            for ( i = 0; i != v93; ++i )
            {
              while ( 1 )
              {
                *(_QWORD *)&v121[0] = sub_B46EC0(v43, i);
                if ( v40 == sub_25EFB40(v38, (__int64)v40, (__int64 *)v121) )
                  break;
                if ( v93 == ++i )
                  goto LABEL_67;
              }
              if ( !v126.m128i_i8[12] )
              {
LABEL_96:
                sub_C8CC70((__int64)&v125, v47, (__int64)v45, v46, v47, v48);
                v85 = 0;
                continue;
              }
              v49 = (_QWORD *)v125.m128i_i64[1];
              v45 = (__int64 *)(v125.m128i_i64[1] + 8LL * v126.m128i_u32[1]);
              if ( (__int64 *)v125.m128i_i64[1] == v45 )
              {
LABEL_97:
                if ( v126.m128i_i32[1] >= (unsigned __int32)v126.m128i_i32[0] )
                  goto LABEL_96;
                v85 = 0;
                ++v126.m128i_i32[1];
                *v45 = v47;
                ++v125.m128i_i64[0];
              }
              else
              {
                while ( v47 != *v49 )
                {
                  if ( v45 == ++v49 )
                    goto LABEL_97;
                }
                v85 = 0;
              }
            }
LABEL_67:
            v41 = v88 + 1;
            if ( v40 == v88 + 1 )
            {
LABEL_68:
              v50 = (__int64 *)v125.m128i_i64[1];
              if ( v126.m128i_i8[12] )
                v51 = v125.m128i_i64[1] + 8LL * v126.m128i_u32[1];
              else
                v51 = v125.m128i_i64[1] + 8LL * v126.m128i_u32[0];
              if ( v125.m128i_i64[1] != v51 )
              {
                while ( 1 )
                {
                  v52 = *v50;
                  if ( (unsigned __int64)*v50 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( (__int64 *)v51 == ++v50 )
                    goto LABEL_73;
                }
                if ( v50 != (__int64 *)v51 )
                {
                  v94 = 0;
                  v57 = v40;
                  v90 = v50;
                  while ( 1 )
                  {
                    v58 = sub_AA5930(v52);
                    v60 = v59;
                    v61 = v58;
                    if ( v58 != v59 )
                      break;
LABEL_117:
                    v71 = v90 + 1;
                    if ( v90 + 1 != (__int64 *)v51 )
                    {
                      while ( 1 )
                      {
                        v52 = *v71;
                        if ( (unsigned __int64)*v71 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( (__int64 *)v51 == ++v71 )
                          goto LABEL_120;
                      }
                      v90 = v71;
                      if ( (__int64 *)v51 != v71 )
                        continue;
                    }
LABEL_120:
                    v83 += v94;
                    goto LABEL_73;
                  }
                  v62 = v94;
                  v63 = v57;
                  v64 = (__int64 *)v121;
                  v65 = v51;
LABEL_106:
                  if ( (*(_DWORD *)(v61 + 4) & 0x7FFFFFF) == 0 )
                    goto LABEL_112;
                  v66 = 0;
                  v67 = (_QWORD *)(*(_QWORD *)(v61 - 8) + 32LL * *(unsigned int *)(v61 + 72));
                  while ( 1 )
                  {
                    *(_QWORD *)&v121[0] = *v67;
                    if ( v63 != sub_25EFB40(v38, (__int64)v63, v64) )
                    {
                      if ( v66 == 1 )
                      {
                        ++v62;
LABEL_112:
                        v70 = *(_QWORD *)(v61 + 32);
                        if ( !v70 )
                          goto LABEL_133;
                        v61 = 0;
                        if ( *(_BYTE *)(v70 - 24) == 84 )
                          v61 = v70 - 24;
                        if ( v60 == v61 )
                        {
                          v94 = v62;
                          v51 = v65;
                          v57 = v63;
                          goto LABEL_117;
                        }
                        goto LABEL_106;
                      }
                      v66 = 1;
                    }
                    v67 = (_QWORD *)(v68 + 8);
                    if ( v69 == v67 )
                      goto LABEL_112;
                  }
                }
              }
LABEL_73:
              if ( v83 + v75 > (int)qword_4FF19E8 )
              {
                v82 = 0x7FFFFFFF;
                if ( v126.m128i_i8[12] )
                  goto LABEL_79;
              }
              else
              {
                v53 = 3 * v83 + v82 + 2 * (v83 + v75) - v79;
                if ( !v85 )
                  v53 = 3 * v83 + v82 + 2 * (v83 + v75);
                v54 = v126.m128i_i32[1] - v126.m128i_i32[2] + v53 - 1;
                if ( (unsigned int)(v126.m128i_i32[1] - v126.m128i_i32[2]) <= 1 )
                  v54 = v53;
                v82 = v54;
                if ( v126.m128i_i8[12] )
                {
LABEL_79:
                  if ( !v74 )
                    goto LABEL_80;
                  goto LABEL_101;
                }
              }
              _libc_free(v125.m128i_u64[1]);
              goto LABEL_79;
            }
          }
        }
LABEL_129:
        v85 = 1;
        goto LABEL_73;
      }
    }
    v91 = v9 + 48;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v9 + 48);
      if ( v18 )
      {
        v18 -= 24;
        v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v20 == v91 )
          goto LABEL_14;
      }
      else
      {
        v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v91 == v20 )
          goto LABEL_28;
      }
      if ( !v20 )
LABEL_133:
        BUG();
      v21 = *(unsigned __int8 *)(v20 - 24);
      v22 = v20 - 24;
      v23 = (unsigned int)(v21 - 30) < 0xB;
      v10 = 0;
      if ( !v23 )
        v22 = 0;
      if ( v22 != v18 )
      {
LABEL_14:
        v24 = 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
        {
          v25 = *(_QWORD *)(v18 - 8);
          v26 = v25 + v24;
        }
        else
        {
          v25 = v18 - v24;
          v26 = v18;
        }
        v27 = v26 - v25;
        v113 = (unsigned __int8 **)v115;
        v28 = v27 >> 5;
        v114 = 0x400000000LL;
        v29 = v27 >> 5;
        if ( (unsigned __int64)v27 > 0x80 )
        {
          v76 = v18;
          v78 = v27 >> 5;
          v80 = v27;
          v86 = v25;
          v89 = v27 >> 5;
          sub_C8D5F0((__int64)&v113, v115, v28, 8u, v29, v18);
          v32 = v113;
          v31 = v114;
          LODWORD(v28) = v89;
          v25 = v86;
          v27 = v80;
          v29 = v78;
          v30 = &v113[(unsigned int)v114];
          v18 = v76;
        }
        else
        {
          v30 = (unsigned __int8 **)v115;
          v31 = 0;
          v32 = (unsigned __int8 **)v115;
        }
        if ( v27 > 0 )
        {
          v33 = 0;
          do
          {
            v30[v33 / 8] = *(unsigned __int8 **)(v25 + 4 * v33);
            v33 += 8LL;
            --v29;
          }
          while ( v29 );
          v32 = v113;
          v31 = v114;
        }
        LODWORD(v114) = v28 + v31;
        v34 = sub_DFCEF0(a4, (unsigned __int8 *)v18, v32, (unsigned int)(v28 + v31), 2);
        v8 = (__m128i *)v113;
        if ( v113 != (unsigned __int8 **)v115 )
        {
          v84 = v35;
          v87 = v34;
          _libc_free((unsigned __int64)v113);
          v35 = v84;
          v34 = v87;
        }
        if ( v35 == 1 )
          v6 = 1;
        v36 = __OFADD__(v34, v7);
        v7 += v34;
        if ( v36 && (v7 = 0x7FFFFFFFFFFFFFFFLL, v34 <= 0) )
        {
          v16 = v116.m128i_i64[0];
          v7 = 0x8000000000000000LL;
        }
        else
        {
          v16 = v116.m128i_i64[0];
        }
      }
LABEL_28:
      v16 = *(_QWORD *)(v16 + 8);
      v116.m128i_i16[4] = 0;
      v116.m128i_i64[0] = v16;
      v18 = v16;
      if ( v16 == v117.m128i_i64[0] )
        goto LABEL_36;
      v37 = v16;
      do
      {
        if ( v37 )
          v37 -= 24;
        if ( !v119 )
          sub_4263D6(v8, v37, v16);
        v8 = (__m128i *)v118;
        if ( v120(v118, v37, v16, v10, v11, v18) )
        {
          v18 = v116.m128i_i64[0];
          v16 = v116.m128i_i64[0];
          goto LABEL_36;
        }
        v10 = 0;
        v37 = *(_QWORD *)(v116.m128i_i64[0] + 8);
        v116.m128i_i16[4] = 0;
        v116.m128i_i64[0] = v37;
        v16 = v37;
      }
      while ( v117.m128i_i64[0] != v37 );
      v18 = v37;
LABEL_36:
      if ( *(_QWORD *)&v121[0] == v18 )
      {
        v17 = v123;
        goto LABEL_38;
      }
    }
  }
  v83 = v106;
  v75 = v100;
  v82 = qword_4FF1CA8;
  if ( (int)qword_4FF1CA8 > 0 )
  {
    *(__int64 *)((char *)v126.m128i_i64 + 4) = 0;
    v125.m128i_i64[1] = (__int64)v127;
    v126.m128i_i8[12] = 1;
    v74 = 0;
    v77 = 0;
    goto LABEL_129;
  }
  v77 = 0;
LABEL_80:
  if ( v82 >= v77 )
LABEL_101:
    v55 = 0;
  else
    v55 = 1;
  if ( v111 != (unsigned __int64 *)&v113 )
    _libc_free((unsigned __int64)v111);
  sub_C7D6A0(v108, 8LL * (unsigned int)v110, 8);
  if ( v105 != &v107 )
    _libc_free((unsigned __int64)v105);
  sub_C7D6A0(v102, 8LL * (unsigned int)v104, 8);
  if ( v99 != &v101 )
    _libc_free((unsigned __int64)v99);
  sub_C7D6A0(v96, 8LL * (unsigned int)v98, 8);
  return v55;
}
