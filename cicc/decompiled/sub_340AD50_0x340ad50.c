// Function: sub_340AD50
// Address: 0x340ad50
//
unsigned __int8 *__fastcall sub_340AD50(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        unsigned __int8 a9,
        char a10,
        unsigned __int8 a11,
        __int64 a12,
        __int16 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        __int64 a19,
        const __m128i *a20,
        __int64 *a21)
{
  __int64 v24; // r15
  unsigned __int8 v25; // bl
  int v26; // eax
  __int64 v27; // rdi
  __int64 (*v28)(); // rax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  unsigned __int8 *result; // rax
  __int64 v32; // rdx
  _QWORD *v33; // rax
  char v34; // si
  unsigned int v35; // eax
  unsigned int v36; // eax
  __int64 *v37; // rdi
  __int64 v38; // rax
  __m128i v39; // xmm0
  __int64 v40; // rax
  __int64 v41; // rax
  __m128i v42; // xmm1
  __m128i *v43; // rsi
  char v44; // bl
  __int64 v45; // rax
  int v46; // eax
  unsigned __int8 v47; // dl
  __int64 v48; // rsi
  __int64 *v49; // rdi
  int v50; // eax
  __int64 v51; // r14
  __int64 (__fastcall *v52)(__int64, __int64, unsigned int); // r13
  __int64 v53; // rax
  int v54; // edx
  unsigned __int16 v55; // ax
  __int64 v56; // rax
  __int64 *v57; // rsi
  __int64 v58; // r13
  __int64 v59; // rdx
  unsigned __int16 *v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 v66; // rcx
  unsigned __int64 v67; // rdi
  unsigned __int32 v68; // r9d
  const __m128i *v69; // rax
  void (***v70)(); // rdi
  void (*v71)(); // rax
  _WORD *v72; // rsi
  __m128i v73; // xmm7
  __m128i *v74; // rsi
  __int64 v75; // [rsp+0h] [rbp-1270h]
  bool v76; // [rsp+Fh] [rbp-1261h]
  unsigned int v77; // [rsp+10h] [rbp-1260h]
  __int64 v78; // [rsp+18h] [rbp-1258h]
  unsigned __int8 v79; // [rsp+20h] [rbp-1250h]
  __int64 v81; // [rsp+28h] [rbp-1248h]
  __m128i v82; // [rsp+30h] [rbp-1240h] BYREF
  __int64 v83; // [rsp+40h] [rbp-1230h]
  __int64 v84; // [rsp+48h] [rbp-1228h]
  unsigned __int64 v85; // [rsp+50h] [rbp-1220h]
  unsigned __int64 v86; // [rsp+58h] [rbp-1218h]
  __m128i v87; // [rsp+60h] [rbp-1210h]
  __m128i v88; // [rsp+70h] [rbp-1200h]
  __m128i v89; // [rsp+90h] [rbp-11E0h]
  unsigned __int64 v90; // [rsp+A0h] [rbp-11D0h] BYREF
  __m128i *v91; // [rsp+A8h] [rbp-11C8h]
  const __m128i *v92; // [rsp+B0h] [rbp-11C0h]
  __int16 v93; // [rsp+C0h] [rbp-11B0h] BYREF
  __int64 v94; // [rsp+C8h] [rbp-11A8h]
  __int64 v95; // [rsp+D0h] [rbp-11A0h]
  __m128i v96; // [rsp+E0h] [rbp-1190h] BYREF
  __m128i v97; // [rsp+F0h] [rbp-1180h] BYREF
  __m128i v98; // [rsp+100h] [rbp-1170h] BYREF
  unsigned __int64 v99; // [rsp+110h] [rbp-1160h] BYREF
  __int64 v100; // [rsp+118h] [rbp-1158h]
  __int64 v101; // [rsp+120h] [rbp-1150h]
  unsigned __int64 v102; // [rsp+128h] [rbp-1148h]
  __int64 v103; // [rsp+130h] [rbp-1140h]
  __int64 v104; // [rsp+138h] [rbp-1138h]
  __int64 v105; // [rsp+140h] [rbp-1130h]
  unsigned __int64 v106; // [rsp+148h] [rbp-1128h] BYREF
  __m128i *v107; // [rsp+150h] [rbp-1120h]
  const __m128i *v108; // [rsp+158h] [rbp-1118h]
  __int64 v109; // [rsp+160h] [rbp-1110h]
  __int64 v110; // [rsp+168h] [rbp-1108h] BYREF
  int v111; // [rsp+170h] [rbp-1100h]
  __int64 v112; // [rsp+178h] [rbp-10F8h]
  _BYTE *v113; // [rsp+180h] [rbp-10F0h]
  __int64 v114; // [rsp+188h] [rbp-10E8h]
  _BYTE v115[1792]; // [rsp+190h] [rbp-10E0h] BYREF
  _BYTE *v116; // [rsp+890h] [rbp-9E0h]
  __int64 v117; // [rsp+898h] [rbp-9D8h]
  _BYTE v118[512]; // [rsp+8A0h] [rbp-9D0h] BYREF
  _BYTE *v119; // [rsp+AA0h] [rbp-7D0h]
  __int64 v120; // [rsp+AA8h] [rbp-7C8h]
  _BYTE v121[1792]; // [rsp+AB0h] [rbp-7C0h] BYREF
  _BYTE *v122; // [rsp+11B0h] [rbp-C0h]
  __int64 v123; // [rsp+11B8h] [rbp-B8h]
  _BYTE v124[64]; // [rsp+11C0h] [rbp-B0h] BYREF
  __int64 v125; // [rsp+1200h] [rbp-70h]
  __int64 v126; // [rsp+1208h] [rbp-68h]
  int v127; // [rsp+1210h] [rbp-60h]
  char v128; // [rsp+1230h] [rbp-40h]

  v24 = a8;
  v82.m128i_i64[0] = a5;
  v25 = a10;
  v79 = a11;
  v26 = *(_DWORD *)(a8 + 24);
  v82.m128i_i64[1] = a6;
  if ( v26 == 35 || v26 == 11 )
  {
    v32 = *(_QWORD *)(a8 + 96);
    if ( *(_DWORD *)(v32 + 32) <= 0x40u )
    {
      if ( !*(_QWORD *)(v32 + 24) )
        return (unsigned __int8 *)a2;
      v33 = *(_QWORD **)(v32 + 24);
      v34 = a10;
    }
    else
    {
      v77 = *(_DWORD *)(v32 + 32);
      v78 = *(_QWORD *)(a8 + 96);
      if ( v77 == (unsigned int)sub_C444A0(v32 + 24) )
        return (unsigned __int8 *)a2;
      v33 = *(_QWORD **)(v78 + 24);
      v34 = v25;
      if ( v77 > 0x40 )
        v33 = (_QWORD *)*v33;
    }
    result = sub_3409510(
               a1,
               a4,
               a2,
               a3,
               v82.m128i_i64[0],
               v82.m128i_i64[1],
               a7,
               *((__int64 *)&a7 + 1),
               (unsigned __int64)v33,
               a9,
               v34,
               0,
               a14,
               a15,
               a16,
               a17,
               a18,
               a19,
               a20,
               a21);
    if ( result )
      return result;
    goto LABEL_4;
  }
  v24 = 0;
LABEL_4:
  v27 = *(_QWORD *)(a1 + 8);
  if ( !v27
    || (v28 = *(__int64 (**)())(*(_QWORD *)v27 + 40LL), v28 == sub_33C7CE0)
    || (result = (unsigned __int8 *)((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64, unsigned __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64))v28)(
                                      v27,
                                      a1,
                                      a4,
                                      a2,
                                      a3,
                                      a9,
                                      v82.m128i_i64[0],
                                      v82.m128i_i64[1],
                                      a7,
                                      *((_QWORD *)&a7 + 1),
                                      a8,
                                      *((_QWORD *)&a8 + 1),
                                      v25,
                                      v79,
                                      a14,
                                      a15,
                                      a16,
                                      a17,
                                      a18,
                                      a19)) == 0 )
  {
    if ( v79 )
    {
      v29 = *(_QWORD *)(v24 + 96);
      v30 = *(_QWORD **)(v29 + 24);
      if ( *(_DWORD *)(v29 + 32) > 0x40u )
        v30 = (_QWORD *)*v30;
      return sub_3409510(
               a1,
               a4,
               a2,
               a3,
               v82.m128i_i64[0],
               v82.m128i_i64[1],
               a7,
               *((__int64 *)&a7 + 1),
               (unsigned __int64)v30,
               a9,
               v25,
               1,
               a14,
               a15,
               a16,
               a17,
               a18,
               a19,
               a20,
               a21);
    }
    else
    {
      v35 = sub_2EAC1E0((__int64)&a14);
      sub_33C8580(*(_QWORD *)(a1 + 16), v35);
      v36 = sub_2EAC1E0((__int64)&a17);
      sub_33C8580(*(_QWORD *)(a1 + 16), v36);
      v37 = *(__int64 **)(a1 + 64);
      v90 = 0;
      v91 = 0;
      v92 = 0;
      v96 = 0u;
      v97 = 0u;
      v98 = 0u;
      v38 = sub_BCE3C0(v37, 0);
      v39 = _mm_load_si128(&v82);
      v97.m128i_i64[1] = v38;
      v89 = v39;
      v96.m128i_i64[1] = v82.m128i_i64[0];
      v97.m128i_i32[0] = v39.m128i_i32[2];
      sub_332CDC0(&v90, 0, &v96);
      v73 = _mm_loadu_si128((const __m128i *)&a7);
      v74 = v91;
      v96.m128i_i64[1] = a7;
      v88 = v73;
      v97.m128i_i32[0] = v73.m128i_i32[2];
      if ( v91 == v92 )
      {
        sub_332CDC0(&v90, v91, &v96);
      }
      else
      {
        if ( v91 )
        {
          *v91 = _mm_loadu_si128(&v96);
          v74[1] = _mm_loadu_si128(&v97);
          v74[2] = _mm_loadu_si128(&v98);
          v74 = v91;
        }
        v91 = v74 + 3;
      }
      v40 = sub_2E79000(*(__int64 **)(a1 + 40));
      v41 = sub_AE4420(v40, *(_QWORD *)(a1 + 64), 0);
      v42 = _mm_loadu_si128((const __m128i *)&a8);
      v43 = v91;
      v97.m128i_i64[1] = v41;
      v87 = v42;
      v96.m128i_i64[1] = a8;
      v97.m128i_i32[0] = v42.m128i_i32[2];
      if ( v91 == v92 )
      {
        sub_332CDC0(&v90, v91, &v96);
      }
      else
      {
        if ( v91 )
        {
          *v91 = _mm_loadu_si128(&v96);
          v43[1] = _mm_loadu_si128(&v97);
          v43[2] = _mm_loadu_si128(&v98);
          v43 = v91;
        }
        v91 = v43 + 3;
      }
      v128 = 0;
      v102 = 0xFFFFFFFF00000020LL;
      v113 = v115;
      v114 = 0x2000000000LL;
      v117 = 0x2000000000LL;
      v120 = 0x2000000000LL;
      v123 = 0x400000000LL;
      v116 = v118;
      v119 = v121;
      v44 = HIBYTE(a13);
      v99 = 0;
      v100 = 0;
      v101 = 0;
      v103 = 0;
      v104 = 0;
      v105 = 0;
      v106 = 0;
      v107 = 0;
      v108 = 0;
      v109 = a1;
      v110 = 0;
      v111 = 0;
      v112 = 0;
      v122 = v124;
      v125 = 0;
      v126 = 0;
      v127 = 0;
      if ( HIBYTE(a13) )
      {
        v44 = a13;
      }
      else
      {
        v45 = *(_QWORD *)(a1 + 16);
        v76 = 0;
        if ( *(_QWORD *)(v45 + 529032) )
        {
          v75 = *(_QWORD *)(v45 + 529032);
          if ( strlen((const char *)v75) == 6 )
          {
            if ( *(_DWORD *)v75 != 1668113773 || (v46 = 0, *(_WORD *)(v75 + 4) != 31088) )
              v46 = 1;
            v76 = v46 == 0;
          }
        }
        if ( a12 )
        {
          v47 = sub_34B9CE0(a12);
          if ( (*(_WORD *)(a12 + 2) & 3u) - 1 <= 1 )
            v44 = sub_34B9AF0(a12, *(_QWORD *)a1, v47 & v76);
          if ( v110 )
            sub_B91220((__int64)&v110, v110);
        }
      }
      v48 = *(_QWORD *)a4;
      v110 = v48;
      if ( v48 )
        sub_B96E90((__int64)&v110, v48, 1);
      v99 = a2;
      LODWORD(v100) = a3;
      v49 = *(__int64 **)(a1 + 40);
      v50 = *(_DWORD *)(a4 + 8);
      v85 = a2;
      v51 = *(_QWORD *)(a1 + 16);
      v86 = a3;
      v111 = v50;
      v52 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v51 + 32LL);
      v53 = sub_2E79000(v49);
      if ( v52 == sub_2D42F30 )
      {
        v54 = sub_AE2980(v53, 0)[1];
        v55 = 2;
        if ( v54 != 1 )
        {
          v55 = 3;
          if ( v54 != 2 )
          {
            v55 = 4;
            if ( v54 != 4 )
            {
              v55 = 5;
              if ( v54 != 8 )
              {
                v55 = 6;
                if ( v54 != 16 )
                {
                  v55 = 7;
                  if ( v54 != 32 )
                  {
                    v55 = 8;
                    if ( v54 != 64 )
                      v55 = 9 * (v54 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v55 = v52(v51, v53, 0);
      }
      v56 = sub_33EED90(a1, *(const char **)(*(_QWORD *)(a1 + 16) + 529032LL), v55, 0);
      v57 = *(__int64 **)(a1 + 64);
      v58 = v56;
      v81 = v59;
      v60 = (unsigned __int16 *)(*(_QWORD *)(v82.m128i_i64[0] + 48) + 16LL * v82.m128i_u32[2]);
      v61 = *v60;
      v62 = *((_QWORD *)v60 + 1);
      v93 = v61;
      v94 = v62;
      v65 = sub_3007410((__int64)&v93, v57, v82.m128i_i64[1], v61, v63, v64);
      v66 = *(_QWORD *)(a1 + 16);
      v101 = v65;
      v67 = v106;
      v68 = *(_DWORD *)(v66 + 533000);
      v84 = v81;
      v83 = v58;
      v104 = v58;
      LODWORD(v105) = v81;
      LODWORD(v103) = v68;
      v106 = v90;
      LODWORD(v65) = -1431655765 * ((__int64)((__int64)v91->m128i_i64 - v90) >> 4);
      v107 = v91;
      v90 = 0;
      v91 = 0;
      HIDWORD(v102) = v65;
      v69 = v92;
      v92 = 0;
      v108 = v69;
      if ( v67 )
      {
        v82.m128i_i32[0] = v68;
        j_j___libc_free_0(v67);
        v68 = v82.m128i_i32[0];
      }
      v70 = *(void (****)())(v109 + 16);
      v71 = **v70;
      if ( v71 != nullsub_1688 )
        ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v71)(
          v70,
          *(_QWORD *)(v109 + 40),
          v68,
          &v106);
      v72 = *(_WORD **)(a1 + 16);
      LOBYTE(v102) = v102 & 0xDF;
      BYTE2(v102) = v44;
      sub_3377410((__int64)&v93, v72, (__int64)&v99);
      result = (unsigned __int8 *)v95;
      if ( v122 != v124 )
      {
        v82.m128i_i64[0] = v95;
        _libc_free((unsigned __int64)v122);
        result = (unsigned __int8 *)v82.m128i_i64[0];
      }
      if ( v119 != v121 )
      {
        v82.m128i_i64[0] = (__int64)result;
        _libc_free((unsigned __int64)v119);
        result = (unsigned __int8 *)v82.m128i_i64[0];
      }
      if ( v116 != v118 )
      {
        v82.m128i_i64[0] = (__int64)result;
        _libc_free((unsigned __int64)v116);
        result = (unsigned __int8 *)v82.m128i_i64[0];
      }
      if ( v113 != v115 )
      {
        v82.m128i_i64[0] = (__int64)result;
        _libc_free((unsigned __int64)v113);
        result = (unsigned __int8 *)v82.m128i_i64[0];
      }
      if ( v110 )
      {
        v82.m128i_i64[0] = (__int64)result;
        sub_B91220((__int64)&v110, v110);
        result = (unsigned __int8 *)v82.m128i_i64[0];
      }
      if ( v106 )
      {
        v82.m128i_i64[0] = (__int64)result;
        j_j___libc_free_0(v106);
        result = (unsigned __int8 *)v82.m128i_i64[0];
      }
      if ( v90 )
      {
        v82.m128i_i64[0] = (__int64)result;
        j_j___libc_free_0(v90);
        return (unsigned __int8 *)v82.m128i_i64[0];
      }
    }
  }
  return result;
}
