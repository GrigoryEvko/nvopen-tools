// Function: sub_3405C90
// Address: 0x3405c90
//
unsigned __int8 *__fastcall sub_3405C90(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __m128i a7,
        __int128 a8,
        __int128 a9)
{
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // rbx
  int v17; // eax
  __int128 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r9
  int v21; // r9d
  __m128i v22; // xmm2
  unsigned __int8 *result; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __m128i *v26; // rax
  __m128i v27; // xmm3
  __m128i v28; // xmm4
  __int64 v29; // rdx
  unsigned __int8 *v30; // rax
  int v31; // esi
  int v32; // r9d
  __int128 v33; // rax
  int v34; // r9d
  unsigned __int64 v35; // r12
  __int16 v36; // r15
  __int64 v37; // rbx
  __int64 v38; // rdx
  int v39; // r9d
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rcx
  int v45; // edx
  unsigned __int16 *v46; // rax
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 *v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // rbx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int16 v54; // dx
  __int64 v55; // r15
  __int64 v56; // rbx
  unsigned int v57; // r15d
  __m128i v58; // rax
  unsigned int v59; // edx
  __int64 v60; // rax
  __int64 v61; // rcx
  __m128i v62; // xmm6
  __int64 v63; // rdi
  char (__fastcall *v64)(__int64, unsigned int); // rax
  __int64 v65; // rcx
  _QWORD *v66; // rdx
  __int64 v67; // r15
  __int64 v68; // rcx
  int v69; // eax
  __int64 v70; // rcx
  _QWORD *v71; // rax
  __int64 v72; // rsi
  _QWORD *v73; // rcx
  __int64 v74; // rax
  bool v75; // al
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  bool v79; // cc
  unsigned __int64 v80; // rax
  __int64 v81; // rax
  __int16 v82; // dx
  __int64 v83; // r15
  unsigned __int64 v84; // r12
  __int64 v85; // rax
  unsigned __int64 v86; // rax
  __int64 v87; // rdx
  unsigned __int8 *v88; // rsi
  int v89; // r9d
  __int64 v90; // rax
  unsigned __int64 v91; // rax
  __int64 v92; // rax
  __int16 v93; // bx
  __int64 v94; // r15
  int v95; // r9d
  __int64 v96; // rdx
  _QWORD *v97; // rax
  bool v98; // al
  char v99; // al
  __int64 v100; // rsi
  __int64 v101; // rdx
  __int128 v102; // [rsp-10h] [rbp-150h]
  __int64 v103; // [rsp-8h] [rbp-148h]
  int v104; // [rsp+8h] [rbp-138h]
  __int64 v105; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v106; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v107; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v108; // [rsp+10h] [rbp-130h]
  __int64 v109; // [rsp+10h] [rbp-130h]
  __int64 v110; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v112; // [rsp+18h] [rbp-128h]
  __int64 v113; // [rsp+30h] [rbp-110h] BYREF
  __int64 v114; // [rsp+38h] [rbp-108h]
  __int64 *v115; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v116[2]; // [rsp+50h] [rbp-F0h] BYREF
  _OWORD v117[2]; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v118; // [rsp+80h] [rbp-C0h] BYREF
  _OWORD v119[11]; // [rsp+90h] [rbp-B0h] BYREF

  v113 = a4;
  v114 = a5;
  sub_33E24E0((__int64)a1, a2, (__m128i *)&a8, (__int64)&a9);
  v14 = a8;
  v15 = *(_DWORD *)(a8 + 24);
  if ( v15 != 35 && v15 != 11 )
    v14 = 0;
  v16 = a9;
  v17 = *(_DWORD *)(a9 + 24);
  if ( v17 != 11 && v17 != 35 )
    v16 = 0;
  v18 = a9;
  v19 = sub_33DFBC0(v18, DWORD2(v18), 0, 1u, v12, v13);
  v20 = v19;
  switch ( a2 )
  {
    case 2u:
      if ( *(_DWORD *)(a8 + 24) == 1 )
        return (unsigned __int8 *)a9;
      if ( *(_DWORD *)(a9 + 24) == 1 || (_QWORD)a9 == (_QWORD)a8 && DWORD2(a9) == DWORD2(a8) )
        return (unsigned __int8 *)a8;
      goto LABEL_9;
    case 3u:
    case 4u:
      v36 = *(_WORD *)(a9 + 96);
      v37 = *(_QWORD *)(a9 + 104);
      if ( v36 != sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) || !v36 && v37 != v38 )
        goto LABEL_9;
      return (unsigned __int8 *)a8;
    case 0x35u:
      if ( *(_DWORD *)(a8 + 24) != 54 )
      {
        if ( v14 )
        {
          v58.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v113);
          v118 = v58;
          v59 = sub_CA1930(&v118);
          v60 = *(_QWORD *)(v16 + 96);
          if ( *(_DWORD *)(v60 + 32) <= 0x40u )
            v61 = *(_QWORD *)(v60 + 24);
          else
            v61 = **(_QWORD **)(v60 + 24);
          sub_C440A0((__int64)&v118, (__int64 *)(*(_QWORD *)(v14 + 96) + 24LL), v59, v59 * v61);
          v107 = sub_34007B0((__int64)a1, (__int64)&v118, a3, v113, v114, 0, a7, 0);
          sub_969240(v118.m128i_i64);
          return v107;
        }
        goto LABEL_9;
      }
      v65 = *(_QWORD *)(v16 + 96);
      v66 = *(_QWORD **)(v65 + 24);
      if ( *(_DWORD *)(v65 + 32) > 0x40u )
        v66 = (_QWORD *)*v66;
      return *(unsigned __int8 **)(*(_QWORD *)(a8 + 40) + 40LL * (unsigned int)v66);
    case 0x38u:
    case 0x39u:
    case 0xBBu:
    case 0xBCu:
      if ( v19 && sub_9867B0(*(_QWORD *)(v19 + 96) + 24LL) )
        return (unsigned __int8 *)a8;
      if ( a2 - 56 > 1 || sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) != 2 )
        goto LABEL_9;
      return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 188, a3, v113, v114, v21, a8, a9);
    case 0x3Au:
      if ( sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) == 2 )
        return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 186, a3, v113, v114, v39, a8, a9);
      if ( !v16 || *(_DWORD *)(a8 + 24) != 373 || (a6 & 2) == 0 )
        goto LABEL_9;
      sub_C472A0(
        (__int64)&v118,
        *(_QWORD *)(**(_QWORD **)(a8 + 40) + 96LL) + 24LL,
        (__int64 *)(*(_QWORD *)(v16 + 96) + 24LL));
      v108 = sub_3401900((__int64)a1, a3, v113, v114, (__int64)&v118, 1, a7);
      sub_969240(v118.m128i_i64);
      return v108;
    case 0x3Bu:
    case 0x3Cu:
    case 0x3Du:
    case 0x3Eu:
    case 0x52u:
    case 0x53u:
    case 0x54u:
    case 0x55u:
    case 0xACu:
    case 0xADu:
      if ( sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) != 2 )
        goto LABEL_9;
      if ( a2 - 82 <= 1 )
        return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 187, a3, v113, v114, v32, a8, a9);
      if ( a2 - 84 <= 1 )
      {
        *(_QWORD *)&v33 = sub_34074A0(a1, a3, a9, *((_QWORD *)&a9 + 1), (unsigned int)v113, v114);
        return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 186, a3, v113, v114, v34, a8, v33);
      }
      goto LABEL_9;
    case 0x60u:
    case 0x61u:
    case 0x62u:
    case 0x63u:
    case 0x64u:
      result = (unsigned __int8 *)sub_33FE9E0(a1, a2, a8, *((__int64 *)&a8 + 1), a9, *((__int64 *)&a9 + 1), a6);
      if ( !result )
        goto LABEL_9;
      return result;
    case 0x9Cu:
      v62 = _mm_loadu_si128((const __m128i *)&a9);
      v118 = _mm_loadu_si128((const __m128i *)&a8);
      v119[0] = v62;
      result = (unsigned __int8 *)sub_33F2070(v113, v114, v118.m128i_i8, 2, a1);
      if ( !result )
        goto LABEL_9;
      return result;
    case 0x9Eu:
      v44 = a8;
      v45 = *(_DWORD *)(a8 + 24);
      if ( v45 == 51 || *(_DWORD *)(a9 + 24) == 51 )
        return (unsigned __int8 *)sub_3288990((__int64)a1, (unsigned int)v113, v114);
      if ( !v16 )
        goto LABEL_138;
      v46 = (unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * DWORD2(a8));
      *((_QWORD *)&v18 + 1) = *v46;
      v47 = *((_QWORD *)v46 + 1);
      LOWORD(v117[0]) = WORD4(v18);
      *((_QWORD *)&v117[0] + 1) = v47;
      if ( WORD4(v18) )
      {
        if ( (unsigned __int16)(WORD4(v18) - 17) > 0x9Eu )
          goto LABEL_70;
      }
      else
      {
        v104 = v45;
        v109 = a8;
        v75 = sub_30070D0((__int64)v117);
        v44 = v109;
        v45 = v104;
        *((_QWORD *)&v18 + 1) = DWORD2(v18);
        if ( !v75 )
          goto LABEL_70;
      }
      v76 = *(_QWORD *)(v16 + 96);
      v118.m128i_i16[0] = WORD4(v18);
      v118.m128i_i64[1] = v47;
      v110 = v76 + 24;
      *((_QWORD *)&v18 + 1) = (unsigned int)sub_3281500(&v118, *((__int64 *)&v18 + 1));
      if ( !sub_986EE0(v110, *((unsigned __int64 *)&v18 + 1)) )
        return (unsigned __int8 *)sub_3288990((__int64)a1, (unsigned int)v113, v114);
      v44 = a8;
      v45 = *(_DWORD *)(a8 + 24);
LABEL_70:
      if ( v45 == 159 )
      {
        v81 = *(_QWORD *)(**(_QWORD **)(v44 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v44 + 40) + 8LL);
        v82 = *(_WORD *)v81;
        v83 = *(_QWORD *)(v81 + 8);
        v118.m128i_i16[0] = v82;
        v118.m128i_i64[1] = v83;
        if ( v82 )
        {
          if ( (unsigned __int16)(v82 - 17) > 0x9Eu )
            goto LABEL_9;
        }
        else
        {
          v98 = sub_30070D0((__int64)&v118);
          v82 = 0;
          if ( !v98 )
            goto LABEL_9;
        }
        v118.m128i_i16[0] = v82;
        v118.m128i_i64[1] = v83;
        v84 = (unsigned int)sub_3281500(&v118, *((__int64 *)&v18 + 1));
        v85 = *(_QWORD *)(v16 + 96);
        v79 = *(_DWORD *)(v85 + 32) <= 0x40u;
        v86 = *(_QWORD *)(v85 + 24);
        if ( !v79 )
          v86 = *(_QWORD *)v86;
        v88 = sub_3400EE0((__int64)a1, v86 % v84, a3, 0, a7);
        v90 = *(_QWORD *)(v16 + 96);
        v79 = *(_DWORD *)(v90 + 32) <= 0x40u;
        v91 = *(_QWORD *)(v90 + 24);
        if ( !v79 )
          v91 = *(_QWORD *)v91;
        *((_QWORD *)&v102 + 1) = v87;
        *(_QWORD *)&v102 = v88;
        return (unsigned __int8 *)sub_3406EB0(
                                    (_DWORD)a1,
                                    158,
                                    a3,
                                    v113,
                                    v114,
                                    v89,
                                    *(_OWORD *)(*(_QWORD *)(a8 + 40) + 40LL * (unsigned int)(v91 / v84)),
                                    v102);
      }
      else
      {
        if ( v45 == 156 )
        {
          v96 = *(_QWORD *)(v16 + 96);
          v97 = *(_QWORD **)(v96 + 24);
          if ( *(_DWORD *)(v96 + 32) > 0x40u )
            v97 = (_QWORD *)*v97;
          v48 = 40LL * (unsigned int)v97;
          goto LABEL_73;
        }
        v48 = 0;
        if ( v45 == 168 )
        {
LABEL_73:
          v49 = (__int64 *)(*(_QWORD *)(v44 + 40) + v48);
          v50 = *v49;
          v51 = v49[1];
          result = (unsigned __int8 *)*v49;
          v52 = *(_QWORD *)(*v49 + 48) + 16LL * *((unsigned int *)v49 + 2);
          if ( (_WORD)v113 != *(_WORD *)v52 || !(_WORD)v113 && v114 != *(_QWORD *)(v52 + 8) )
            return sub_33FAFB0((__int64)a1, v50, v51, a3, (unsigned int)v113, v114, a7);
          return result;
        }
LABEL_138:
        if ( v45 != 157 )
        {
          if ( v45 != 161 )
            goto LABEL_9;
          v92 = *(_QWORD *)(v44 + 48) + 16LL * DWORD2(a8);
          v93 = *(_WORD *)v92;
          v94 = *(_QWORD *)(v92 + 8);
          LOWORD(v117[0]) = v93;
          *((_QWORD *)&v117[0] + 1) = v94;
          if ( v93 )
          {
            if ( (unsigned __int16)(v93 - 17) > 0x9Eu )
              goto LABEL_9;
          }
          else if ( !sub_30070D0((__int64)v117) )
          {
            goto LABEL_9;
          }
          v118.m128i_i16[0] = v93;
          v118.m128i_i64[1] = v94;
          if ( (unsigned int)sub_3281500(&v118, *((__int64 *)&v18 + 1)) == 1 )
            return (unsigned __int8 *)sub_3406EB0(
                                        (_DWORD)a1,
                                        158,
                                        a3,
                                        v113,
                                        v114,
                                        v95,
                                        *(_OWORD *)*(_QWORD *)(a8 + 40),
                                        *(_OWORD *)(*(_QWORD *)(a8 + 40) + 40LL));
LABEL_9:
          v22 = _mm_loadu_si128((const __m128i *)&a9);
          v118 = _mm_loadu_si128((const __m128i *)&a8);
          v119[0] = v22;
          result = sub_3402EA0(
                     (__int64)a1,
                     a2,
                     (unsigned __int64 *)a3,
                     (unsigned int)v113,
                     v114,
                     a6,
                     a7,
                     (unsigned int *)&v118,
                     2);
          if ( result )
            return result;
          if ( *(_DWORD *)(a8 + 24) != 51 )
          {
LABEL_12:
            if ( *(_DWORD *)(a9 + 24) != 51 )
              goto LABEL_22;
            if ( a2 > 0x55 )
            {
LABEL_14:
              if ( a2 != 188 )
              {
                if ( a2 > 0xBC )
                  goto LABEL_22;
                if ( a2 != 186 )
                {
                  if ( a2 == 187 )
                    return sub_34015B0((__int64)a1, a3, (unsigned int)v113, v114, 0, 0, a7);
LABEL_22:
                  v26 = sub_33ED250((__int64)a1, (unsigned int)v113, v114);
                  v27 = _mm_loadu_si128((const __m128i *)&a8);
                  v28 = _mm_loadu_si128((const __m128i *)&a9);
                  v116[0] = (__int64)v26;
                  v116[1] = v29;
                  v117[0] = v27;
                  v117[1] = v28;
                  if ( (_WORD)v113 == 262 )
                  {
                    v35 = sub_33E6540(a1, a2, *(_DWORD *)(a3 + 8), (__int64 *)a3, v116);
                    sub_33E4EC0((__int64)a1, v35, (__int64)v117, 2);
                  }
                  else
                  {
                    v118.m128i_i64[1] = 0x2000000000LL;
                    v118.m128i_i64[0] = (__int64)v119;
                    sub_33C9670((__int64)&v118, a2, v116[0], (unsigned __int64 *)v117, 2, (__int64)v117);
                    v115 = 0;
                    v30 = (unsigned __int8 *)sub_33CCCF0((__int64)a1, (__int64)&v118, a3, (__int64 *)&v115);
                    if ( v30 )
                    {
                      v31 = a6;
                      v112 = v30;
                      sub_33D00A0((__int64)v30, v31);
                      result = v112;
                      if ( (_OWORD *)v118.m128i_i64[0] != v119 )
                      {
                        _libc_free(v118.m128i_u64[0]);
                        return v112;
                      }
                      return result;
                    }
                    v35 = sub_33E6540(a1, a2, *(_DWORD *)(a3 + 8), (__int64 *)a3, v116);
                    *(_DWORD *)(v35 + 28) = a6;
                    sub_33E4EC0((__int64)a1, v35, (__int64)v117, 2);
                    sub_C657C0(a1 + 65, (__int64 *)v35, v115, (__int64)off_4A367D0);
                    if ( (_OWORD *)v118.m128i_i64[0] != v119 )
                      _libc_free(v118.m128i_u64[0]);
                  }
                  sub_33CC420((__int64)a1, v35);
                  return (unsigned __int8 *)v35;
                }
                return sub_3400BD0((__int64)a1, 0, a3, (unsigned int)v113, v114, 0, a7, 0);
              }
              if ( *(_DWORD *)(a8 + 24) == 51 )
                return sub_3400BD0((__int64)a1, 0, a3, (unsigned int)v113, v114, 0, a7, 0);
              return (unsigned __int8 *)sub_3288990((__int64)a1, (unsigned int)v113, v114);
            }
            goto LABEL_119;
          }
          v63 = a1[2];
          v64 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v63 + 1360LL);
          if ( v64 == sub_2FE3400 )
          {
            if ( a2 <= 0x62 )
            {
              if ( a2 > 0x37 )
              {
                switch ( a2 )
                {
                  case '8':
                  case ':':
                  case '?':
                  case '@':
                  case 'D':
                  case 'F':
                  case 'L':
                  case 'M':
                  case 'R':
                  case 'S':
                  case '`':
                  case 'b':
                    goto LABEL_131;
                  default:
                    goto LABEL_115;
                }
              }
              goto LABEL_115;
            }
            if ( a2 > 0xBC )
            {
              if ( a2 - 279 > 7 )
              {
LABEL_122:
                if ( a2 == 222 )
                  return sub_3400BD0((__int64)a1, 0, a3, (unsigned int)v113, v114, 0, a7, 0);
                goto LABEL_127;
              }
            }
            else if ( a2 <= 0xB9 && a2 - 172 > 0xB )
            {
LABEL_127:
              if ( *(_DWORD *)(a9 + 24) != 51 )
                goto LABEL_22;
              goto LABEL_14;
            }
          }
          else if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64))v64)(
                       v63,
                       a2,
                       v24,
                       v25,
                       v103) )
          {
LABEL_115:
            if ( a2 <= 0x3E )
            {
              if ( a2 > 0x3A )
                return sub_3400BD0((__int64)a1, 0, a3, (unsigned int)v113, v114, 0, a7, 0);
              if ( a2 == 57 )
                return (unsigned __int8 *)sub_3288990((__int64)a1, (unsigned int)v113, v114);
              if ( *(_DWORD *)(a9 + 24) != 51 )
                goto LABEL_22;
LABEL_119:
              if ( a2 <= 0x37 )
                goto LABEL_22;
              goto LABEL_120;
            }
            if ( a2 <= 0x55 )
            {
              if ( a2 > 0x53 )
                return sub_3400BD0((__int64)a1, 0, a3, (unsigned int)v113, v114, 0, a7, 0);
              if ( *(_DWORD *)(a9 + 24) != 51 )
                goto LABEL_22;
LABEL_120:
              switch ( a2 )
              {
                case '8':
                case '9':
                case ';':
                case '<':
                case '=':
                case '>':
                  return (unsigned __int8 *)sub_3288990((__int64)a1, (unsigned int)v113, v114);
                case ':':
                case 'T':
                case 'U':
                  return sub_3400BD0((__int64)a1, 0, a3, (unsigned int)v113, v114, 0, a7, 0);
                case 'R':
                case 'S':
                  return sub_34015B0((__int64)a1, a3, (unsigned int)v113, v114, 0, 0, a7);
                default:
                  goto LABEL_22;
              }
            }
            goto LABEL_122;
          }
LABEL_131:
          a7 = _mm_loadu_si128((const __m128i *)&a8);
          *(_QWORD *)&a8 = a9;
          DWORD2(a8) = DWORD2(a9);
          *(_QWORD *)&a9 = a7.m128i_i64[0];
          DWORD2(a9) = a7.m128i_i32[2];
          goto LABEL_12;
        }
        v67 = *(_QWORD *)(v44 + 40);
        v68 = *(_QWORD *)(v67 + 80);
        v69 = *(_DWORD *)(v68 + 24);
        if ( v69 != 35 && v69 != 11 || !v16 )
          goto LABEL_9;
        v70 = *(_QWORD *)(v68 + 96);
        v71 = *(_QWORD **)(v70 + 24);
        if ( *(_DWORD *)(v70 + 32) > 0x40u )
          v71 = (_QWORD *)*v71;
        v72 = *(_QWORD *)(v16 + 96);
        v73 = *(_QWORD **)(v72 + 24);
        if ( *(_DWORD *)(v72 + 32) > 0x40u )
          v73 = (_QWORD *)*v73;
        if ( v73 == v71 )
        {
          v74 = *(_QWORD *)(*(_QWORD *)(v67 + 40) + 48LL) + 16LL * *(unsigned int *)(v67 + 48);
          if ( (_WORD)v113 == *(_WORD *)v74 && ((_WORD)v113 || v114 == *(_QWORD *)(v74 + 8)) )
          {
            return *(unsigned __int8 **)(v67 + 40);
          }
          else
          {
            v99 = sub_3280140((__int64)&v113);
            v100 = *(_QWORD *)(v67 + 40);
            v101 = *(_QWORD *)(v67 + 48);
            if ( v99 )
              return (unsigned __int8 *)sub_3406EE0(a1, v100, v101, a3, (unsigned int)v113, v114);
            else
              return sub_33FB160((__int64)a1, v100, v101, a3, (unsigned int)v113, v114, a7);
          }
        }
        else
        {
          return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 158, a3, v113, v114, v20, *(_OWORD *)v67, a9);
        }
      }
    case 0x9Fu:
      a7 = _mm_loadu_si128((const __m128i *)&a9);
      v118 = _mm_loadu_si128((const __m128i *)&a8);
      v119[0] = a7;
      result = sub_33FC250(a3, (unsigned int)v113, v114, v118.m128i_i8, 2, a1, a7);
      if ( !result )
        goto LABEL_9;
      return result;
    case 0xA1u:
      v40 = *(_QWORD *)(a8 + 48) + 16LL * DWORD2(a8);
      if ( (_WORD)v113 == *(_WORD *)v40 && ((_WORD)v113 || v114 == *(_QWORD *)(v40 + 8)) )
        return (unsigned __int8 *)a8;
      v41 = *(_DWORD *)(a8 + 24);
      if ( v41 == 51 )
        return (unsigned __int8 *)sub_3288990((__int64)a1, (unsigned int)v113, v114);
      if ( v41 != 159 )
      {
        if ( v41 == 160 )
        {
          v42 = *(_QWORD *)(a8 + 40);
          if ( (_QWORD)a9 == *(_QWORD *)(v42 + 80) && DWORD2(a9) == *(_DWORD *)(v42 + 88) )
          {
            v43 = *(_QWORD *)(*(_QWORD *)(v42 + 40) + 48LL) + 16LL * *(unsigned int *)(v42 + 48);
            if ( (_WORD)v113 == *(_WORD *)v43 && ((_WORD)v113 || v114 == *(_QWORD *)(v43 + 8)) )
              return *(unsigned __int8 **)(v42 + 40);
          }
        }
        goto LABEL_9;
      }
      v77 = *(_QWORD *)(**(_QWORD **)(a8 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a8 + 40) + 8LL);
      if ( (_WORD)v113 != *(_WORD *)v77 || !(_WORD)v113 && v114 != *(_QWORD *)(v77 + 8) )
        goto LABEL_9;
      v118.m128i_i64[0] = sub_3281590((__int64)&v113);
      v78 = *(_QWORD *)(v16 + 96);
      v79 = *(_DWORD *)(v78 + 32) <= 0x40u;
      v80 = *(_QWORD *)(v78 + 24);
      if ( !v79 )
        v80 = *(_QWORD *)v80;
      return *(unsigned __int8 **)(*(_QWORD *)(a8 + 40) + 40LL * (unsigned int)(v80 / v118.m128i_u32[0]));
    case 0xB2u:
    case 0xB3u:
      if ( sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) != 2 )
        goto LABEL_9;
      return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 188, a3, v113, v114, v21, a8, a9);
    case 0xB4u:
    case 0xB7u:
      if ( sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) == 2 )
        return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 187, a3, v113, v114, v32, a8, a9);
      goto LABEL_9;
    case 0xB5u:
    case 0xB6u:
      if ( sub_3281100((unsigned __int16 *)&v113, *((__int64 *)&v18 + 1)) != 2 )
        goto LABEL_9;
      return (unsigned __int8 *)sub_3406EB0((_DWORD)a1, 186, a3, v113, v114, v39, a8, a9);
    case 0xBAu:
      if ( !v19 )
        goto LABEL_9;
      v56 = *(_QWORD *)(v19 + 96);
      if ( sub_9867B0(v56 + 24) )
        return (unsigned __int8 *)a9;
      v57 = *(_DWORD *)(v56 + 32);
      if ( !v57 )
        return (unsigned __int8 *)a8;
      if ( v57 <= 0x40 )
      {
        if ( *(_QWORD *)(v56 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v57) )
          goto LABEL_9;
      }
      else if ( v57 != (unsigned int)sub_C445E0(v56 + 24) )
      {
        goto LABEL_9;
      }
      return (unsigned __int8 *)a8;
    case 0xBEu:
      if ( !v16 || *(_DWORD *)(a8 + 24) != 373 || (a6 & 2) == 0 )
        goto LABEL_26;
      v55 = *(_QWORD *)(v16 + 96) + 24LL;
      sub_9865C0((__int64)&v118, *(_QWORD *)(**(_QWORD **)(a8 + 40) + 96LL) + 24LL);
      sub_C47AC0((__int64)&v118, v55);
      v106 = sub_3401900((__int64)a1, a3, v113, v114, (__int64)&v118, 1, a7);
      sub_969240(v118.m128i_i64);
      return v106;
    case 0xBFu:
    case 0xC0u:
LABEL_26:
      v105 = v19;
      result = sub_3401190(a1, a8, DWORD2(a8), a9, *((__int64 *)&a9 + 1), v19, a7);
      v20 = v105;
      if ( !result )
        goto LABEL_27;
      return result;
    case 0xC1u:
    case 0xC2u:
LABEL_27:
      if ( (_WORD)v113 == 2 || v20 && sub_9867B0(*(_QWORD *)(v20 + 96) + 24LL) )
        return (unsigned __int8 *)a8;
      goto LABEL_9;
    case 0xDEu:
      v54 = *(_WORD *)(a9 + 96);
      if ( (_WORD)v113 != v54 )
        goto LABEL_9;
      if ( *(_QWORD *)(a9 + 104) == v114 )
        return (unsigned __int8 *)a8;
      goto LABEL_78;
    case 0xE6u:
      v53 = *(_QWORD *)(a8 + 48) + 16LL * DWORD2(a8);
      v54 = *(_WORD *)v53;
      if ( (_WORD)v113 != *(_WORD *)v53 )
        goto LABEL_9;
      if ( v114 == *(_QWORD *)(v53 + 8) )
        return (unsigned __int8 *)a8;
LABEL_78:
      if ( !v54 )
        goto LABEL_9;
      return (unsigned __int8 *)a8;
    default:
      goto LABEL_9;
  }
}
