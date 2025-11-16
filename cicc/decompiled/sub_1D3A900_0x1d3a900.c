// Function: sub_1D3A900
// Address: 0x1d3a900
//
__int64 *__fastcall sub_1D3A900(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        unsigned __int64 a4,
        const void **a5,
        __int64 a6,
        __m128 a7,
        double a8,
        __m128i a9,
        unsigned __int64 a10,
        __int16 *a11,
        __int128 a12,
        __int64 a13,
        __int64 a14)
{
  unsigned int v14; // r11d
  unsigned __int16 v15; // r14
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm3
  int v23; // edx
  __int64 v24; // rsi
  int v25; // r13d
  __int64 v26; // rbx
  unsigned __int8 *v27; // rsi
  __int64 *result; // rax
  int v29; // edx
  __int64 *v30; // r15
  __int64 v31; // rsi
  __int64 *v32; // rax
  int v33; // eax
  __m128i v34; // xmm4
  __m128i v35; // xmm6
  char v36; // al
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rax
  char v40; // dl
  __int64 v41; // rax
  unsigned int v42; // eax
  _QWORD *v43; // rbx
  int v44; // eax
  int v45; // esi
  int v46; // eax
  __int64 *v47; // rsi
  __int64 v48; // rdx
  unsigned int v49; // r11d
  __int64 v50; // r10
  __int16 *v51; // rax
  unsigned int v52; // r11d
  __int64 v53; // r10
  int v54; // eax
  unsigned int v55; // r11d
  __int64 v56; // r10
  __int64 v57; // rsi
  int v58; // ecx
  int v59; // r8d
  unsigned __int8 *v60; // rsi
  __int64 v61; // rdx
  _QWORD *v62; // rax
  __int64 v63; // rax
  _BYTE *v64; // rax
  __int64 *v66; // rdi
  float v67; // xmm0_4
  __int64 *v68; // rdi
  float v69; // xmm0_4
  __int64 *v70; // rdi
  float v71; // xmm0_4
  unsigned __int64 v72; // [rsp-10h] [rbp-180h]
  __int64 v73; // [rsp+0h] [rbp-170h]
  unsigned int v74; // [rsp+8h] [rbp-168h]
  __int64 v75; // [rsp+10h] [rbp-160h]
  __int16 *v76; // [rsp+10h] [rbp-160h]
  __int64 v77; // [rsp+10h] [rbp-160h]
  __int16 *v78; // [rsp+10h] [rbp-160h]
  __int64 v79; // [rsp+10h] [rbp-160h]
  int v80; // [rsp+18h] [rbp-158h]
  unsigned int v81; // [rsp+18h] [rbp-158h]
  __int64 v82; // [rsp+18h] [rbp-158h]
  float v83; // [rsp+18h] [rbp-158h]
  __int64 v84; // [rsp+20h] [rbp-150h]
  __int64 v85; // [rsp+20h] [rbp-150h]
  __int64 v86; // [rsp+20h] [rbp-150h]
  __int64 v87; // [rsp+20h] [rbp-150h]
  float v88; // [rsp+20h] [rbp-150h]
  float v89; // [rsp+20h] [rbp-150h]
  __int64 v90; // [rsp+20h] [rbp-150h]
  char v91; // [rsp+28h] [rbp-148h]
  __int64 v92; // [rsp+28h] [rbp-148h]
  __int64 v93; // [rsp+28h] [rbp-148h]
  __int64 v94; // [rsp+28h] [rbp-148h]
  __int16 *v95; // [rsp+28h] [rbp-148h]
  unsigned int v96; // [rsp+28h] [rbp-148h]
  int v97; // [rsp+28h] [rbp-148h]
  unsigned int v98; // [rsp+28h] [rbp-148h]
  int v99; // [rsp+28h] [rbp-148h]
  unsigned int v100; // [rsp+28h] [rbp-148h]
  unsigned int v101; // [rsp+28h] [rbp-148h]
  __int64 v102; // [rsp+30h] [rbp-140h]
  unsigned int v103; // [rsp+30h] [rbp-140h]
  unsigned int v104; // [rsp+30h] [rbp-140h]
  __int64 v105; // [rsp+30h] [rbp-140h]
  unsigned __int64 v106; // [rsp+30h] [rbp-140h]
  __int16 v107; // [rsp+3Eh] [rbp-132h]
  __int64 v108; // [rsp+40h] [rbp-130h]
  __int64 *v109; // [rsp+40h] [rbp-130h]
  _QWORD *v110; // [rsp+40h] [rbp-130h]
  __int64 *v111; // [rsp+58h] [rbp-118h] BYREF
  __int64 v112[4]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v113; // [rsp+80h] [rbp-F0h] BYREF
  __int16 *v114; // [rsp+88h] [rbp-E8h] BYREF
  __m128i v115; // [rsp+90h] [rbp-E0h]
  __int64 v116; // [rsp+A0h] [rbp-D0h]
  __int64 v117; // [rsp+A8h] [rbp-C8h]
  unsigned __int8 *v118; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v119; // [rsp+B8h] [rbp-B8h] BYREF
  __m128i v120; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v121; // [rsp+D0h] [rbp-A0h]
  __int64 v122; // [rsp+D8h] [rbp-98h]

  v14 = a4;
  v15 = a2;
  v107 = a6;
  v19 = a13;
  v108 = a14;
  if ( a2 <= 0x89 )
  {
    if ( a2 <= 0x62 )
      goto LABEL_6;
    switch ( a2 )
    {
      case 0x63u:
        v44 = *(unsigned __int16 *)(a10 + 24);
        a4 = a10;
        if ( v44 != 11 )
        {
          a4 = 0;
          if ( v44 == 33 )
            a4 = a10;
        }
        v45 = *(unsigned __int16 *)(a13 + 24);
        v46 = *(unsigned __int16 *)(a12 + 24);
        if ( v46 != 33 && v46 != 11 || v45 != 11 && v45 != 33 || !a4 )
          goto LABEL_6;
        v75 = a13;
        v81 = v14;
        v85 = a12;
        v105 = *(_QWORD *)(a4 + 88);
        v95 = (__int16 *)sub_16982C0();
        v47 = (__int64 *)(v105 + 32);
        if ( *(__int16 **)(v105 + 32) == v95 )
        {
          sub_169C6E0(&v114, (__int64)v47);
          v50 = v75;
          v49 = v81;
          v48 = v85;
        }
        else
        {
          sub_16986C0(&v114, v47);
          v48 = v85;
          v49 = v81;
          v50 = v75;
        }
        v86 = *(_QWORD *)(v48 + 88);
        v76 = v114;
        v82 = *(_QWORD *)(v50 + 88);
        if ( v114 == v95 )
        {
          v79 = v50;
          v100 = v49;
          v54 = sub_169F930(&v114, v86 + 32, v82 + 32, 0);
          v56 = v79;
          v55 = v100;
        }
        else
        {
          v73 = v50;
          v74 = v49;
          v51 = (__int16 *)sub_1698270();
          v52 = v74;
          v53 = v73;
          if ( v76 == v51 )
          {
            v78 = v51;
            v64 = sub_16D40F0((__int64)qword_4FBB490);
            v52 = v74;
            v53 = v73;
            if ( v64 ? *v64 : LOBYTE(qword_4FBB490[2]) )
            {
              if ( v95 == *(__int16 **)(v82 + 32) )
                v66 = (__int64 *)(*(_QWORD *)(v82 + 40) + 8LL);
              else
                v66 = (__int64 *)(v82 + 32);
              v67 = sub_169D890(v66);
              if ( v95 == *(__int16 **)(v86 + 32) )
                v68 = (__int64 *)(*(_QWORD *)(v86 + 40) + 8LL);
              else
                v68 = (__int64 *)(v86 + 32);
              v83 = v67;
              v69 = sub_169D890(v68);
              v70 = (__int64 *)&v114;
              if ( v95 == v114 )
                v70 = (__int64 *)(v115.m128i_i64[0] + 8);
              v88 = v69;
              v71 = sub_169D890(v70);
              a9 = (__m128i)LODWORD(v83);
              *(_QWORD *)&a8 = LODWORD(v88);
              v89 = sub_1C40E60(v71, v88, v83, (__int64)&v111, 1, 1);
              a7.m128_u64[0] = LODWORD(v89);
              if ( !sub_1C40EE0(v89) )
              {
                sub_169D3B0((__int64)v112, (__m128i)LODWORD(v89));
                sub_169E320(&v119, v112, v78);
                sub_1698460((__int64)v112);
                sub_1D14E50((void **)&v114, (void **)&v119);
                sub_127D120(&v119);
                v55 = v74;
LABEL_65:
                v110 = sub_1D36490((__int64)a1, (__int64)&v113, a3, v55, a5, 0, *(double *)a7.m128_u64, a8, a9);
                sub_127D120(&v114);
                return v110;
              }
              if ( v95 == v114 )
                sub_169CAA0((__int64)&v114, 0, 0, 0, v89);
              else
                sub_16986F0(&v114, 0, 0, 0);
              v55 = v74;
              v56 = v73;
              if ( !*(_BYTE *)(a1[2] + 57) )
                goto LABEL_65;
LABEL_97:
              v90 = v56;
              v101 = v55;
              sub_127D120(&v114);
              v14 = v101;
              v19 = v90;
              goto LABEL_6;
            }
          }
          v77 = v53;
          v96 = v52;
          v54 = sub_169DD30(&v114, (_BYTE *)(v86 + 32), (_BYTE *)(v82 + 32), 0);
          v55 = v96;
          v56 = v77;
        }
        if ( *(_BYTE *)(a1[2] + 57) != 1 || v54 != 1 )
          goto LABEL_65;
        goto LABEL_97;
      case 0x69u:
        v37 = *(unsigned __int16 *)(a13 + 24);
        if ( v37 != 10 && v37 != 32 )
          goto LABEL_6;
        v38 = *(_QWORD *)(a13 + 88);
        a4 = *(_QWORD *)(v38 + 24);
        if ( *(_DWORD *)(v38 + 32) > 0x40u )
          a4 = **(_QWORD **)(v38 + 24);
        v39 = *(_QWORD *)(a10 + 40) + 16LL * (unsigned int)a11;
        v40 = *(_BYTE *)v39;
        v41 = *(_QWORD *)(v39 + 8);
        LOBYTE(v118) = v40;
        v119 = v41;
        if ( v40 )
        {
          v42 = word_42E7700[(unsigned __int8)(v40 - 14)];
        }
        else
        {
          v87 = a13;
          v98 = v14;
          v106 = a4;
          v42 = sub_1F58D30(&v118);
          v19 = v87;
          v14 = v98;
          a4 = v106;
        }
        if ( v42 > a4 )
          goto LABEL_6;
        v118 = 0;
        LODWORD(v119) = 0;
        v43 = sub_1D2B300(a1, 0x30u, (__int64)&v118, v14, (__int64)a5, a6);
        if ( v118 )
          sub_161E7C0((__int64)&v118, (__int64)v118);
        return v43;
      case 0x6Bu:
        v34 = _mm_loadu_si128((const __m128i *)&a12);
        v119 = (__int64)a11;
        v121 = a13;
        v93 = a13;
        v103 = a4;
        v118 = (unsigned __int8 *)a10;
        v122 = a14;
        v120 = v34;
        result = sub_1D374B0(a3, (_BYTE *)(unsigned int)a4, a5, (unsigned int *)&v118, 3, a1, a7, a8, a9);
        v14 = v103;
        v19 = v93;
        if ( !result )
          goto LABEL_6;
        return result;
      case 0x6Cu:
        if ( !(_BYTE)a4 )
        {
          v92 = a13;
          v113 = a10;
          v102 = sub_1D29190((__int64)a1, a4, (__int64)a5, a4, (__int64)a5, a6);
          v80 = v29;
          v114 = a11;
          v116 = v92;
          v117 = v108;
          v115 = _mm_loadu_si128((const __m128i *)&a12);
          goto LABEL_17;
        }
        if ( *(_BYTE *)(*(_QWORD *)(a10 + 40) + 16LL * (unsigned int)a11) )
        {
          v36 = *(_BYTE *)(*(_QWORD *)(a12 + 40) + 16LL * DWORD2(a12));
          if ( v36 )
          {
            if ( v36 == (_BYTE)a4 )
              return (__int64 *)a12;
          }
        }
        goto LABEL_6;
      case 0x86u:
        v33 = *(unsigned __int16 *)(a10 + 24);
        if ( v33 == 10 || v33 == 32 )
        {
          v61 = *(_QWORD *)(a10 + 88);
          v62 = *(_QWORD **)(v61 + 24);
          if ( *(_DWORD *)(v61 + 32) > 0x40u )
            v62 = (_QWORD *)*v62;
          if ( !v62 )
            return (__int64 *)a13;
        }
        else if ( DWORD2(a12) != (_DWORD)a14 || (_QWORD)a12 != a13 )
        {
          goto LABEL_6;
        }
        return (__int64 *)a12;
      case 0x89u:
        v94 = a13;
        v104 = a4;
        result = (__int64 *)sub_1D3B650(
                              (_DWORD)a1,
                              a4,
                              (_DWORD)a5,
                              a10,
                              (_DWORD)a11,
                              *(_DWORD *)(a13 + 84),
                              a12,
                              *((__int64 *)&a12 + 1),
                              a3);
        if ( !result )
        {
          v35 = _mm_loadu_si128((const __m128i *)&a12);
          v119 = (__int64)a11;
          v122 = v108;
          v121 = v94;
          v118 = (unsigned __int8 *)a10;
          v120 = v35;
          result = sub_1D39800(a1, 0x89u, a3, v104, a5, 0, *(double *)a7.m128_u64, a8, a9, (__int64 *)&v118, 3);
          a4 = v72;
          v14 = v104;
          v19 = v94;
          if ( !result )
            goto LABEL_6;
        }
        return result;
      default:
        goto LABEL_6;
    }
  }
  if ( a2 == 158 )
  {
    v20 = *(_QWORD *)(a10 + 40) + 16LL * (unsigned int)a11;
    if ( *(_BYTE *)v20 == (_BYTE)a4 && (*(const void ***)(v20 + 8) == a5 || (_BYTE)a4) )
      return (__int64 *)a10;
  }
LABEL_6:
  v84 = v19;
  v91 = v14;
  v21 = sub_1D29190((__int64)a1, v14, (__int64)a5, a4, (__int64)a5, a6);
  v22 = _mm_loadu_si128((const __m128i *)&a12);
  v113 = a10;
  v102 = v21;
  v80 = v23;
  v114 = a11;
  v116 = v84;
  v117 = v108;
  v115 = v22;
  if ( v91 == 111 )
  {
    v24 = *(_QWORD *)a3;
    v25 = *(_DWORD *)(a3 + 8);
    v118 = (unsigned __int8 *)v24;
    if ( v24 )
      sub_1623A60((__int64)&v118, v24, 2);
    v26 = a1[26];
    if ( v26 )
      a1[26] = *(_QWORD *)v26;
    else
      v26 = sub_145CBF0(a1 + 27, 112, 8);
    *(_QWORD *)v26 = 0;
    *(_QWORD *)(v26 + 8) = 0;
    *(_QWORD *)(v26 + 16) = 0;
    *(_WORD *)(v26 + 24) = v15;
    *(_DWORD *)(v26 + 28) = -1;
    *(_QWORD *)(v26 + 32) = 0;
    *(_QWORD *)(v26 + 40) = v102;
    *(_QWORD *)(v26 + 48) = 0;
    *(_DWORD *)(v26 + 56) = 0;
    *(_DWORD *)(v26 + 60) = v80;
    *(_DWORD *)(v26 + 64) = v25;
    v27 = v118;
    *(_QWORD *)(v26 + 72) = v118;
    if ( v27 )
      sub_1623210((__int64)&v118, v27, v26 + 72);
    *(_WORD *)(v26 + 80) &= 0xF000u;
    *(_WORD *)(v26 + 26) = 0;
    sub_1D23B60((__int64)a1, v26, (__int64)&v113, 3);
LABEL_14:
    sub_1D172A0((__int64)a1, v26);
    return (__int64 *)v26;
  }
LABEL_17:
  v118 = (unsigned __int8 *)&v120;
  v119 = 0x2000000000LL;
  sub_16BD430((__int64)&v118, v15);
  sub_16BD4C0((__int64)&v118, v102);
  v30 = &v113;
  do
  {
    v31 = *v30;
    v30 += 2;
    sub_16BD4C0((__int64)&v118, v31);
    sub_16BD430((__int64)&v118, *((_DWORD *)v30 - 2));
  }
  while ( v30 != (__int64 *)&v118 );
  v111 = 0;
  v32 = sub_1D17920((__int64)a1, (__int64)v30, a3, (__int64 *)&v111);
  if ( !v32 )
  {
    v57 = *(_QWORD *)a3;
    v58 = *(_DWORD *)(a3 + 8);
    v112[0] = v57;
    if ( v57 )
    {
      v97 = v58;
      sub_1623A60((__int64)v112, v57, 2);
      v58 = v97;
    }
    v26 = a1[26];
    v59 = v80;
    if ( v26 )
    {
      a1[26] = *(_QWORD *)v26;
    }
    else
    {
      v99 = v58;
      v63 = sub_145CBF0(a1 + 27, 112, 8);
      v58 = v99;
      v59 = v80;
      v26 = v63;
    }
    *(_QWORD *)v26 = 0;
    *(_QWORD *)(v26 + 8) = 0;
    *(_QWORD *)(v26 + 16) = 0;
    *(_WORD *)(v26 + 24) = v15;
    *(_DWORD *)(v26 + 28) = -1;
    *(_QWORD *)(v26 + 32) = 0;
    *(_QWORD *)(v26 + 40) = v102;
    *(_QWORD *)(v26 + 48) = 0;
    *(_DWORD *)(v26 + 56) = 0;
    *(_DWORD *)(v26 + 60) = v59;
    *(_DWORD *)(v26 + 64) = v58;
    v60 = (unsigned __int8 *)v112[0];
    *(_QWORD *)(v26 + 72) = v112[0];
    if ( v60 )
      sub_1623210((__int64)v112, v60, v26 + 72);
    *(_WORD *)(v26 + 26) = 0;
    *(_WORD *)(v26 + 80) = v107;
    sub_1D23B60((__int64)a1, v26, (__int64)&v113, 3);
    sub_16BDA20(a1 + 40, (__int64 *)v26, v111);
    if ( v118 != (unsigned __int8 *)&v120 )
      _libc_free((unsigned __int64)v118);
    goto LABEL_14;
  }
  v109 = v32;
  sub_1D19330((__int64)v32, v107);
  result = v109;
  if ( v118 != (unsigned __int8 *)&v120 )
  {
    _libc_free((unsigned __int64)v118);
    return v109;
  }
  return result;
}
