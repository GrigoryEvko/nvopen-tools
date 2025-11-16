// Function: sub_1769090
// Address: 0x1769090
//
__int64 __fastcall sub_1769090(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // r12
  __int64 v14; // r13
  int v15; // eax
  __int64 v16; // r13
  unsigned int v17; // ebx
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 v25; // r15
  __m128 v26; // xmm0
  __int64 v27; // rbx
  __m128i v28; // xmm1
  char v29; // al
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rcx
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // rbx
  __int64 v36; // r13
  _QWORD *v37; // rax
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v41; // rdi
  _QWORD *v42; // rdi
  __int32 v43; // edx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int16 v47; // ax
  __int16 v48; // r12
  __int64 v49; // rbx
  __int64 v50; // r14
  _QWORD **v51; // rax
  _QWORD *v52; // r13
  __int64 *v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int16 v61; // ax
  __int64 v62; // rax
  char v63; // al
  __int64 *v64; // rsi
  __int16 *v65; // r15
  __int64 v66; // rax
  __int64 v67; // r15
  __int64 v68; // r15
  unsigned __int8 v69; // al
  char v70; // al
  __int64 v71; // r15
  __int64 v72; // r14
  __int16 v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // r13
  _QWORD *v78; // rax
  __int64 v79; // r12
  _QWORD *v80; // rax
  _QWORD *v81; // r14
  _QWORD *v82; // r13
  _QWORD *v83; // rax
  __int64 **v84; // rax
  __int64 v85; // rbx
  _QWORD *v86; // rax
  __int64 v87; // rax
  __int64 v88; // r12
  _QWORD *v89; // rax
  __int64 v90; // rax
  __int64 v91; // r14
  char v92; // al
  __int64 v93; // r14
  __int64 v94; // r15
  __int64 v95; // r12
  _QWORD *v96; // rax
  __int64 v97; // r12
  _QWORD *v98; // rax
  __int64 v99; // r12
  _QWORD *v100; // rax
  unsigned int i; // r14d
  __int64 v102; // rax
  char v103; // al
  void *v104; // rax
  __int64 v105; // r14
  __int64 v106; // [rsp+10h] [rbp-D0h]
  __int64 v107; // [rsp+20h] [rbp-C0h]
  __int16 *v108; // [rsp+20h] [rbp-C0h]
  __int64 v109; // [rsp+20h] [rbp-C0h]
  int v110; // [rsp+20h] [rbp-C0h]
  char v111; // [rsp+2Bh] [rbp-B5h]
  unsigned int v112; // [rsp+2Ch] [rbp-B4h]
  __int64 v113; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v114; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v115[3]; // [rsp+48h] [rbp-98h] BYREF
  __int64 v116; // [rsp+60h] [rbp-80h] BYREF
  __int64 *v117; // [rsp+68h] [rbp-78h] BYREF
  __int64 v118; // [rsp+70h] [rbp-70h]
  char v119; // [rsp+7Ah] [rbp-66h]
  __m128 v120; // [rsp+80h] [rbp-60h] BYREF
  __m128i v121; // [rsp+90h] [rbp-50h]
  __int64 v122; // [rsp+A0h] [rbp-40h]

  v13 = a2;
  v14 = *(_QWORD *)(a2 - 48);
  v15 = *(unsigned __int8 *)(v14 + 16);
  if ( (unsigned __int8)v15 <= 0x17u )
  {
    if ( (_BYTE)v15 == 17 )
    {
      v17 = 3;
    }
    else
    {
      v17 = 2;
      if ( (unsigned __int8)v15 <= 0x10u )
        v17 = (_BYTE)v15 != 9;
    }
  }
  else if ( (unsigned int)(v15 - 60) <= 0xC
         || sub_15FB6B0(*(_QWORD *)(a2 - 48), a2, a3, a4)
         || (a2 = 0, sub_15FB6D0(v14, 0, a3, a4))
         || (v17 = 5, sub_15FB730(v14, 0, a3, a4)) )
  {
    v16 = *(_QWORD *)(v13 - 24);
    v17 = 4;
    v18 = *(unsigned __int8 *)(v16 + 16);
    if ( (unsigned __int8)v18 <= 0x17u )
      goto LABEL_4;
LABEL_13:
    if ( (unsigned int)(v18 - 60) <= 0xC
      || sub_15FB6B0(v16, a2, a3, a4)
      || sub_15FB6D0(v16, 0, v19, v20)
      || sub_15FB730(v16, 0, v21, v22) )
    {
      v23 = 4;
    }
    else
    {
      v23 = 5;
    }
    goto LABEL_23;
  }
  v16 = *(_QWORD *)(v13 - 24);
  v18 = *(unsigned __int8 *)(v16 + 16);
  if ( (unsigned __int8)v18 > 0x17u )
    goto LABEL_13;
LABEL_4:
  if ( (_BYTE)v18 == 17 )
  {
    v23 = 3;
  }
  else
  {
    if ( (unsigned __int8)v18 <= 0x10u )
    {
      v111 = 0;
      if ( (_BYTE)v18 == 9 )
        goto LABEL_25;
      if ( !v17 )
        goto LABEL_8;
      goto LABEL_24;
    }
    v23 = 2;
  }
LABEL_23:
  if ( v17 < v23 )
  {
LABEL_8:
    *(_WORD *)(v13 + 18) = sub_15FF5D0(*(_WORD *)(v13 + 18) & 0x7FFF) | *(_WORD *)(v13 + 18) & 0x8000;
    sub_16484A0((__int64 *)(v13 - 48), (__int64 *)(v13 - 24));
    v111 = 1;
    v16 = *(_QWORD *)(v13 - 24);
    goto LABEL_25;
  }
LABEL_24:
  v111 = 0;
  v16 = *(_QWORD *)(v13 - 24);
LABEL_25:
  v24 = *(unsigned __int16 *)(v13 + 18);
  v122 = v13;
  v25 = v13;
  v26 = (__m128)_mm_loadu_si128((const __m128i *)(a1 + 2672));
  v27 = *(_QWORD *)(v13 - 48);
  v28 = _mm_loadu_si128((const __m128i *)(a1 + 2688));
  BYTE1(v24) &= ~0x80u;
  v120 = v26;
  v121 = v28;
  v112 = v24;
  v29 = sub_15F24E0(v13);
  v30 = v27;
  v107 = (__int64)sub_13D91C0(v112, v27, v16, v29, &v120);
  if ( v107 )
  {
    v35 = *(_QWORD *)(v13 + 8);
    if ( v35 )
    {
      v36 = *(_QWORD *)a1;
      do
      {
        v37 = sub_1648700(v35);
        sub_170B990(v36, (__int64)v37);
        v35 = *(_QWORD *)(v35 + 8);
      }
      while ( v35 );
      if ( v13 == v107 )
        v107 = sub_1599EF0(*(__int64 ***)v13);
      sub_164D160(v13, v107, v26, *(double *)v28.m128i_i64, a7, a8, v38, v39, a11, a12);
      return v25;
    }
    return 0;
  }
  if ( v27 == v16 )
  {
    switch ( v112 )
    {
      case 1u:
      case 3u:
      case 5u:
      case 7u:
        v61 = *(_WORD *)(v13 + 18) & 0x8000 | 7;
        break;
      case 8u:
      case 0xAu:
      case 0xCu:
      case 0xEu:
        v61 = *(_WORD *)(v13 + 18) & 0x8000 | 8;
        break;
      default:
        goto LABEL_34;
    }
    *(_WORD *)(v13 + 18) = v61;
    v62 = sub_15A06D0(*(__int64 ***)v16, v27, v112, v32);
    sub_1593B40((_QWORD *)(v13 - 24), v62);
  }
  else
  {
LABEL_34:
    if ( v112 - 7 <= 1 )
    {
      if ( !(unsigned __int8)sub_17574F0((_BYTE *)v27, v27, v31, v32) && sub_14AB850((__int64 *)v27, v27, v55, v56) )
      {
        v57 = sub_15A06D0(*(__int64 ***)v27, v27, v55, v56);
        sub_1593B40((_QWORD *)(v13 - 48), v57);
        return v25;
      }
      if ( !(unsigned __int8)sub_17574F0((_BYTE *)v16, v27, v55, v56) && sub_14AB850((__int64 *)v16, v27, v58, v32) )
      {
        v60 = sub_15A06D0(*(__int64 ***)v27, v27, v59, v32);
        sub_1593B40((_QWORD *)(v13 - 24), v60);
        return v25;
      }
    }
    v41 = *(_QWORD *)(v13 + 8);
    if ( v41 )
    {
      if ( !*(_QWORD *)(v41 + 8) )
      {
        v42 = sub_1648700(v41);
        if ( *((_BYTE *)v42 + 16) == 79 )
        {
          v30 = (__int64)&v114;
          v120.m128_u64[0] = sub_14B2890((__int64)v42, &v114, &v116, 0, 0);
          v120.m128_i32[2] = v43;
          if ( v120.m128_i32[0] )
            return 0;
        }
      }
    }
    v44 = *(unsigned __int8 *)(v16 + 16);
    if ( (unsigned __int8)v44 <= 0x10u && *(_BYTE *)(v27 + 16) > 0x17u )
    {
      switch ( *(_BYTE *)(v27 + 16) )
      {
        case '&':
          v68 = *(_QWORD *)(v27 - 48);
          v69 = *(_BYTE *)(v68 + 16);
          if ( v69 == 14 )
          {
            if ( *(void **)(v68 + 32) == sub_16982C0() )
            {
              v94 = *(_QWORD *)(v68 + 40);
              if ( (*(_BYTE *)(v94 + 26) & 7) != 3 )
                break;
              v71 = v94 + 8;
            }
            else
            {
              v70 = *(_BYTE *)(v68 + 50);
              v71 = v68 + 32;
              if ( (v70 & 7) != 3 )
                break;
            }
            if ( (*(_BYTE *)(v71 + 18) & 8) == 0 )
              break;
          }
          else
          {
            v44 = *(_QWORD *)v68;
            if ( *(_BYTE *)(*(_QWORD *)v68 + 8LL) != 16 || v69 > 0x10u )
              break;
            v90 = sub_15A1020(*(_BYTE **)(v27 - 48), v30, v44, v32);
            v91 = v90;
            if ( v90 && *(_BYTE *)(v90 + 16) == 14 )
            {
              if ( *(void **)(v90 + 32) == sub_16982C0() )
              {
                v105 = *(_QWORD *)(v91 + 40);
                if ( (*(_BYTE *)(v105 + 26) & 7) != 3 )
                  break;
                v93 = v105 + 8;
              }
              else
              {
                v92 = *(_BYTE *)(v91 + 50);
                v93 = v91 + 32;
                if ( (v92 & 7) != 3 )
                  break;
              }
              if ( (*(_BYTE *)(v93 + 18) & 8) == 0 )
                break;
            }
            else
            {
              v110 = *(_QWORD *)(*(_QWORD *)v68 + 32LL);
              if ( v110 )
              {
                for ( i = 0; i != v110; ++i )
                {
                  v30 = i;
                  v102 = sub_15A0A60(v68, i);
                  v44 = v102;
                  if ( !v102 )
                    goto LABEL_44;
                  v103 = *(_BYTE *)(v102 + 16);
                  v106 = v44;
                  if ( v103 != 9 )
                  {
                    if ( v103 != 14 )
                      goto LABEL_44;
                    v104 = sub_16982C0();
                    v44 = v106;
                    if ( *(void **)(v106 + 32) == v104 )
                    {
                      v44 = *(_QWORD *)(v106 + 40);
                      if ( (*(_BYTE *)(v44 + 26) & 7) != 3 )
                        goto LABEL_44;
                      v44 += 8;
                    }
                    else
                    {
                      if ( (*(_BYTE *)(v106 + 50) & 7) != 3 )
                        goto LABEL_44;
                      v44 = v106 + 32;
                    }
                    if ( (*(_BYTE *)(v44 + 18) & 8) == 0 )
                      goto LABEL_44;
                  }
                }
              }
            }
          }
          v72 = *(_QWORD *)(v27 - 24);
          if ( !v72 )
            break;
          v73 = sub_15FF5D0(*(_WORD *)(v13 + 18) & 0x7FFF);
          v76 = sub_15A2BF0((__int64 *)v16, v30, v74, v75, *(double *)v26.m128_u64, *(double *)v28.m128i_i64, a7);
          v121.m128i_i16[0] = 257;
          v77 = v76;
          v78 = sub_1648A60(56, 2u);
          v25 = (__int64)v78;
          if ( v78 )
            sub_1758370((__int64)v78, v73, v72, v77, (__int64)&v120);
          return v25;
        case '6':
          v66 = sub_13CF970(v27);
          v67 = *(_QWORD *)v66;
          if ( *(_BYTE *)(*(_QWORD *)v66 + 16LL) == 56 )
          {
            v44 = *(_QWORD *)(v67 - 24LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v44 + 16) == 3 && (*(_BYTE *)(v44 + 80) & 1) != 0 )
            {
              v109 = *(_QWORD *)(v67 - 24LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF));
              if ( !sub_15E4F60(v109) )
                __asm { jmp     rax }
            }
          }
          break;
        case 'A':
        case 'B':
          if ( (_BYTE)v44 != 14 )
            break;
          v25 = sub_175F2F0(
                  (__int64 *)a1,
                  v13,
                  v27,
                  (_QWORD *)v16,
                  v26,
                  *(double *)v28.m128i_i64,
                  a7,
                  a8,
                  v33,
                  v34,
                  a11,
                  a12);
          if ( !v25 )
            break;
          return v25;
        case 'D':
          if ( (_BYTE)v44 != 14 )
            break;
          v63 = *(_BYTE *)(**(_QWORD **)(v27 - 24) + 8LL);
          switch ( v63 )
          {
            case 1:
              v108 = (__int16 *)sub_1698260();
              break;
            case 2:
              v108 = (__int16 *)sub_1698270();
              break;
            case 3:
              v108 = (__int16 *)sub_1698280();
              break;
            case 5:
              v108 = (__int16 *)sub_1698290();
              break;
            case 4:
              v108 = (__int16 *)sub_16982A0();
              break;
            case 6:
              v108 = (__int16 *)sub_16982C0();
              break;
            default:
              goto LABEL_44;
          }
          v64 = (__int64 *)(v16 + 32);
          v65 = (__int16 *)sub_16982C0();
          if ( *(__int16 **)(v16 + 32) == v65 )
            sub_169C6E0(v115, (__int64)v64);
          else
            sub_16986C0(v115, v64);
          sub_16A3360((__int64)&v114, v108, 0, (bool *)&v113);
          if ( (__int16 *)v115[0] == v65 )
            sub_169C6E0(&v117, (__int64)v115);
          else
            sub_16986C0(&v117, v115);
          if ( v65 == (__int16 *)v117 )
          {
            if ( (*(_BYTE *)(v118 + 26) & 8) != 0 )
              sub_169C8D0((__int64)&v117, *(double *)v26.m128_u64, *(double *)v28.m128i_i64, a7);
          }
          else if ( (v119 & 8) != 0 )
          {
            sub_1699490((__int64)&v117);
          }
          if ( (_BYTE)v113 )
            goto LABEL_82;
          if ( v108 == v65 )
            sub_169C580(&v120.m128_u64[1], (__int64)v65);
          else
            sub_1698390((__int64)&v120.m128_i64[1], (__int64)v108);
          if ( v65 == (__int16 *)v120.m128_u64[1] )
            sub_16A1F30((__int64)&v120.m128_i64[1], 0, *(double *)v26.m128_u64, *(double *)v28.m128i_i64, a7);
          else
            sub_169B400((__int64)&v120.m128_i64[1], 0);
          if ( (unsigned int)sub_14A9E40((__int64)&v116, (__int64)&v120) )
            goto LABEL_120;
          v84 = &v117;
          if ( v65 == (__int16 *)v117 )
            v84 = (__int64 **)(v118 + 8);
          if ( (*((_BYTE *)v84 + 18) & 7) == 3 )
          {
LABEL_120:
            sub_127D120(&v120.m128_u64[1]);
            v85 = *(_QWORD *)(v27 - 24);
            v86 = (_QWORD *)sub_16498A0(v16);
            v87 = sub_159CCF0(v86, (__int64)&v114);
            v121.m128i_i16[0] = 257;
            v88 = v87;
            v89 = sub_1648A60(56, 2u);
            v25 = (__int64)v89;
            if ( v89 )
              sub_1758370((__int64)v89, v112, v85, v88, (__int64)&v120);
            sub_127D120(&v117);
            sub_127D120(v115);
            return v25;
          }
          sub_127D120(&v120.m128_u64[1]);
LABEL_82:
          sub_127D120(&v117);
          sub_127D120(v115);
          break;
        case 'M':
          if ( *(_QWORD *)(v13 + 40) != *(_QWORD *)(v27 + 40) )
            break;
          v25 = sub_17127D0((__int64 *)a1, v13, v27, v26, *(double *)v28.m128i_i64, a7, a8, v33, v34, a11, a12);
          if ( !v25 )
            break;
          return v25;
        case 'N':
          if ( !sub_1593BB0(v16, v30, v44, v32) || (unsigned int)sub_14AB140(v27 | 4, *(__int64 **)(a1 + 2648)) != 96 )
            break;
          switch ( v112 )
          {
            case 1u:
            case 6u:
            case 9u:
            case 0xEu:
              v79 = *(_QWORD *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
              v121.m128i_i16[0] = 257;
              v80 = sub_1648A60(56, 2u);
              v25 = (__int64)v80;
              if ( v80 )
                sub_1758370((__int64)v80, v112, v79, v16, (__int64)&v120);
              break;
            case 2u:
              v95 = *(_QWORD *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
              v121.m128i_i16[0] = 257;
              v96 = sub_1648A60(56, 2u);
              v25 = (__int64)v96;
              if ( v96 )
                sub_1758370((__int64)v96, 6, v95, v16, (__int64)&v120);
              break;
            case 3u:
              v99 = *(_QWORD *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
              v121.m128i_i16[0] = 257;
              v100 = sub_1648A60(56, 2u);
              v25 = (__int64)v100;
              if ( v100 )
                sub_1758370((__int64)v100, 7, v99, v16, (__int64)&v120);
              break;
            case 5u:
              v97 = *(_QWORD *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
              v121.m128i_i16[0] = 257;
              v98 = sub_1648A60(56, 2u);
              v25 = (__int64)v98;
              if ( v98 )
                sub_1758370((__int64)v98, 1, v97, v16, (__int64)&v120);
              break;
            default:
              goto LABEL_44;
          }
          return v25;
        default:
          break;
      }
    }
LABEL_44:
    v117 = &v113;
    if ( (unsigned __int8)sub_171FB50((__int64)&v116, v27, v44, v32)
      && (v120.m128_u64[1] = (unsigned __int64)&v114, (unsigned __int8)sub_171FB50((__int64)&v120, v16, v45, v46)) )
    {
      v47 = sub_15FF5D0(*(_WORD *)(v13 + 18) & 0x7FFF);
      v121.m128i_i16[0] = 257;
      v48 = v47;
      v25 = (__int64)sub_1648A60(56, 2u);
      if ( v25 )
      {
        v49 = v113;
        v50 = v114;
        v51 = *(_QWORD ***)v113;
        if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) == 16 )
        {
          v52 = v51[4];
          v53 = (__int64 *)sub_1643320(*v51);
          v54 = (__int64)sub_16463B0(v53, (unsigned int)v52);
        }
        else
        {
          v54 = sub_1643320(*v51);
        }
        sub_15FEC10(v25, v54, 52, v48, v49, v50, (__int64)&v120, 0);
      }
    }
    else
    {
      if ( *(_BYTE *)(v27 + 16) != 68
        || *(_BYTE *)(v16 + 16) != 68
        || (v81 = *(_QWORD **)(v27 - 24), v82 = *(_QWORD **)(v16 - 24), *v82 != *v81) )
      {
        v25 = v13;
        if ( v111 )
          return v25;
        return 0;
      }
      v121.m128i_i16[0] = 257;
      v83 = sub_1648A60(56, 2u);
      v25 = (__int64)v83;
      if ( v83 )
        sub_1758370((__int64)v83, v112, (__int64)v81, (__int64)v82, (__int64)&v120);
    }
  }
  return v25;
}
