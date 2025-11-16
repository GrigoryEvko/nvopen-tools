// Function: sub_3786920
// Address: 0x3786920
//
__m128i *__fastcall sub_3786920(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r11
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r15
  __m128i v8; // xmm0
  __int64 v9; // r9
  __int64 v10; // r13
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int16 v13; // dx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __m128i *result; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // r11
  __int64 v21; // r9
  unsigned __int16 *v22; // rax
  int v23; // edx
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // r13
  __int16 v27; // ax
  __int64 v28; // rdx
  bool v29; // al
  __int64 v30; // rsi
  _QWORD *v31; // r13
  __int64 v32; // rbx
  unsigned int v33; // r14d
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  unsigned __int16 v37; // r13
  __int64 v38; // rdx
  __m128i v39; // rax
  __int16 v40; // ax
  __int64 v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // r13
  __m128i v44; // rax
  unsigned int v45; // eax
  int v46; // r9d
  __int16 v47; // r13
  __int64 v48; // r8
  __int64 v49; // rdi
  unsigned int v50; // edx
  __int64 v51; // r9
  unsigned __int8 *v52; // rax
  unsigned int v53; // edx
  unsigned int v54; // eax
  __int64 v55; // rax
  __int8 v56; // cl
  __int64 v57; // rax
  int v58; // edx
  __m128i v59; // rax
  unsigned __int8 v60; // al
  __int64 v61; // r13
  __m128i v62; // rax
  unsigned __int64 v63; // rdx
  _QWORD *v64; // rdi
  __int128 v65; // rax
  unsigned __int8 *v66; // rax
  unsigned int v67; // edx
  __int64 v68; // r14
  __int64 v69; // rdx
  char v70; // cl
  unsigned __int64 v71; // rax
  __int16 v72; // bx
  bool v73; // al
  __int64 v74; // rdx
  int v75; // eax
  int v76; // ecx
  __int64 v77; // rsi
  unsigned __int16 v78; // ax
  __int64 v79; // rdx
  bool v80; // al
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // rdx
  int v84; // esi
  __int128 v85; // [rsp-30h] [rbp-180h]
  unsigned __int64 v86; // [rsp+10h] [rbp-140h]
  unsigned __int64 v87; // [rsp+10h] [rbp-140h]
  __int128 v88; // [rsp+10h] [rbp-140h]
  __int64 v89; // [rsp+20h] [rbp-130h]
  __int64 v90; // [rsp+20h] [rbp-130h]
  unsigned int v91; // [rsp+28h] [rbp-128h]
  unsigned __int64 v92; // [rsp+28h] [rbp-128h]
  unsigned __int8 v93; // [rsp+28h] [rbp-128h]
  __int64 v94; // [rsp+28h] [rbp-128h]
  __int64 v95; // [rsp+28h] [rbp-128h]
  unsigned __int64 v96; // [rsp+30h] [rbp-120h]
  __m128i *v97; // [rsp+30h] [rbp-120h]
  _QWORD *v98; // [rsp+30h] [rbp-120h]
  __int64 v99; // [rsp+30h] [rbp-120h]
  unsigned __int64 v100; // [rsp+38h] [rbp-118h]
  unsigned __int64 v101; // [rsp+38h] [rbp-118h]
  __m128i *v102; // [rsp+40h] [rbp-110h]
  unsigned __int8 *v103; // [rsp+50h] [rbp-100h]
  unsigned int v104; // [rsp+80h] [rbp-D0h] BYREF
  unsigned __int64 v105; // [rsp+88h] [rbp-C8h]
  __int64 v106; // [rsp+90h] [rbp-C0h] BYREF
  int v107; // [rsp+98h] [rbp-B8h]
  __m128i v108; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int64 v109; // [rsp+B0h] [rbp-A0h]
  __int64 v110; // [rsp+B8h] [rbp-98h]
  __int128 v111; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v112; // [rsp+D0h] [rbp-80h]
  __int128 v113; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v114; // [rsp+F0h] [rbp-60h]
  __m128i v115; // [rsp+100h] [rbp-50h] BYREF
  __int64 v116; // [rsp+110h] [rbp-40h]
  __int64 v117; // [rsp+118h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_QWORD *)v4;
  v7 = *(_QWORD *)(v4 + 8);
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v9 = *(_QWORD *)(v4 + 40);
  v10 = 16LL * *(unsigned int *)(v4 + 8);
  v11 = *(_DWORD *)(v4 + 48);
  v12 = v10 + *(_QWORD *)(*(_QWORD *)v4 + 48LL);
  v91 = v11;
  v13 = *(_WORD *)v12;
  v105 = *(_QWORD *)(v12 + 8);
  v14 = *(_DWORD *)(v9 + 24);
  LOWORD(v104) = v13;
  if ( v14 != 35 && v14 != 11 )
    goto LABEL_3;
  v19 = *(_QWORD *)(v9 + 96);
  if ( *(_DWORD *)(v19 + 32) <= 0x40u )
    v96 = *(_QWORD *)(v19 + 24);
  else
    v96 = **(_QWORD **)(v19 + 24);
  v89 = v9;
  v86 = v5;
  *(_QWORD *)&v111 = 0;
  DWORD2(v111) = 0;
  *(_QWORD *)&v113 = 0;
  DWORD2(v113) = 0;
  sub_375E8D0((__int64)a1, v6, v7, (__int64)&v111, (__int64)&v113);
  v20 = v86;
  v21 = v89;
  v22 = (unsigned __int16 *)(*(_QWORD *)(v111 + 48) + 16LL * DWORD2(v111));
  v23 = *v22;
  v24 = *((_QWORD *)v22 + 1);
  v115.m128i_i16[0] = v23;
  v115.m128i_i64[1] = v24;
  if ( (_WORD)v23 )
  {
    v25 = word_4456340[v23 - 1];
    if ( v25 <= v96 )
      goto LABEL_10;
    return (__m128i *)sub_33EC010(
                        (_QWORD *)a1[1],
                        (__int64 *)a2,
                        v111,
                        *((unsigned __int64 *)&v111 + 1),
                        v8.m128i_i64[0],
                        v8.m128i_i64[1]);
  }
  v54 = sub_3007240((__int64)&v115);
  v21 = v89;
  v20 = v86;
  v25 = v54;
  if ( v54 > v96 )
    return (__m128i *)sub_33EC010(
                        (_QWORD *)a1[1],
                        (__int64 *)a2,
                        v111,
                        *((unsigned __int64 *)&v111 + 1),
                        v8.m128i_i64[0],
                        v8.m128i_i64[1]);
LABEL_10:
  v26 = *(_QWORD *)(v20 + 48) + v10;
  v27 = *(_WORD *)v26;
  v28 = *(_QWORD *)(v26 + 8);
  v115.m128i_i16[0] = v27;
  v115.m128i_i64[1] = v28;
  if ( v27 )
  {
    v29 = (unsigned __int16)(v27 - 176) <= 0x34u;
  }
  else
  {
    v90 = v21;
    v87 = v25;
    v29 = sub_3007100((__int64)&v115);
    v21 = v90;
    v25 = v87;
  }
  if ( v29 )
  {
LABEL_3:
    if ( (unsigned __int8)sub_3761870(a1, a2, **(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL), 1) )
      return 0;
    v36 = *(_QWORD *)(a2 + 80);
    v106 = v36;
    if ( v36 )
      sub_B96E90((__int64)&v106, v36, 1);
    v107 = *(_DWORD *)(a2 + 72);
    if ( (_WORD)v104 )
    {
      v37 = word_4456580[(unsigned __int16)v104 - 1];
      v38 = 0;
    }
    else
    {
      v37 = sub_3009970((__int64)&v104, v36, v15, v16, v17);
    }
    v108.m128i_i16[0] = v37;
    v108.m128i_i64[1] = v38;
    v39.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v108);
    v115 = v39;
    if ( v39.m128i_i64[0] )
    {
      v59.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v108);
      v115 = v59;
      if ( (v59.m128i_i8[0] & 7) == 0 )
      {
        v60 = sub_33CD850(a1[1], v104, v105, 0);
        v61 = a1[1];
        v93 = v60;
        v62.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v104);
        v115 = v62;
        LOBYTE(v110) = v62.m128i_i8[8];
        v109 = (unsigned __int64)(v62.m128i_i64[0] + 7) >> 3;
        v98 = sub_33EDE90(v61, v109, v110, v93);
        v100 = v63;
        sub_2EAC300((__int64)&v111, *(_QWORD *)(a1[1] + 40), *((_DWORD *)v98 + 24), 0);
        v64 = (_QWORD *)a1[1];
        v115 = 0u;
        v116 = 0;
        v117 = 0;
        *(_QWORD *)&v65 = sub_33F4560(
                            v64,
                            (unsigned __int64)(v64 + 36),
                            0,
                            (__int64)&v106,
                            v6,
                            v7,
                            (unsigned __int64)v98,
                            v100,
                            v111,
                            v112,
                            v93,
                            0,
                            (__int64)&v115);
        v88 = v65;
        v66 = sub_3466750(*a1, (_QWORD *)a1[1], (__int64)v98, v100, v104, v105, v8, *(_OWORD *)&v8);
        v115 = 0u;
        v99 = (__int64)v66;
        v68 = a1[1];
        v116 = 0;
        v101 = v67 | v100 & 0xFFFFFFFF00000000LL;
        v117 = 0;
        *(_QWORD *)&v113 = sub_2D5B750((unsigned __int16 *)&v108);
        *((_QWORD *)&v113 + 1) = v69;
        v70 = -1;
        v71 = -(__int64)(((unsigned __int64)v113 >> 3) | (1LL << v93)) & (((unsigned __int64)v113 >> 3) | (1LL << v93));
        if ( v71 )
        {
          _BitScanReverse64(&v71, v71);
          v70 = 63 - (v71 ^ 0x3F);
        }
        LOBYTE(v72) = v70;
        sub_2EAC3A0((__int64)&v113, *(__int64 **)(v68 + 40));
        HIBYTE(v72) = 1;
        result = sub_33F1DB0(
                   (__int64 *)v68,
                   1,
                   (__int64)&v106,
                   **(unsigned __int16 **)(a2 + 48),
                   *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                   v72,
                   v88,
                   v99,
                   v101,
                   v113,
                   v114,
                   v108.m128i_i64[0],
                   v108.m128i_i64[1],
                   0,
                   (__int64)&v115);
LABEL_31:
        if ( v106 )
        {
          v102 = result;
          sub_B91220((__int64)&v106, v106);
          return v102;
        }
        return result;
      }
    }
    if ( v37 )
    {
      if ( (unsigned __int16)(v37 - 17) <= 0xD3u )
      {
        v115.m128i_i16[0] = v37;
        v40 = sub_30369B0((unsigned __int16 *)&v115);
        v41 = 0;
      }
      else
      {
        if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
          BUG();
        v55 = 16LL * (v37 - 1);
        v56 = byte_444C4A0[v55 + 8];
        v57 = *(_QWORD *)&byte_444C4A0[v55];
        v115.m128i_i8[8] = v56;
        v115.m128i_i64[0] = v57;
        v58 = sub_CA1930(&v115);
        v40 = 2;
        if ( v58 != 1 )
        {
          v40 = 3;
          if ( v58 != 2 )
          {
            v40 = 4;
            if ( v58 != 4 )
            {
              v40 = 5;
              if ( v58 != 8 )
              {
                v40 = 6;
                if ( v58 != 16 )
                {
                  v40 = 7;
                  if ( v58 != 32 )
                  {
                    v40 = 8;
                    if ( v58 != 64 )
                      v40 = 9 * (v58 == 128);
                  }
                }
              }
            }
          }
        }
        v41 = 0;
      }
    }
    else if ( sub_30070B0((__int64)&v108) )
    {
      v40 = sub_300A990((unsigned __int16 *)&v108, v36);
    }
    else
    {
      v40 = sub_30072B0((__int64)&v108);
    }
    LOWORD(v113) = v40;
    v42 = a1[1];
    *((_QWORD *)&v113 + 1) = v41;
    v43 = *(_QWORD **)(v42 + 64);
    v44.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v113);
    v115 = v44;
    v45 = sub_CA1930(&v115);
    if ( v45 > 8 )
    {
      _BitScanReverse(&v45, v45 - 1);
      v75 = v45 ^ 0x1F;
      v76 = 32 - v75;
      if ( v75 == 28 )
      {
        v47 = 6;
        v108.m128i_i64[1] = 0;
        v108.m128i_i16[0] = 6;
        v115 = _mm_loadu_si128(&v108);
      }
      else if ( v76 == 5 )
      {
        v108.m128i_i16[0] = 7;
        v47 = 7;
        v108.m128i_i64[1] = 0;
        v115 = _mm_loadu_si128(&v108);
      }
      else
      {
        if ( v76 != 6 )
        {
          if ( v76 == 7 )
          {
            v47 = 9;
            v48 = 0;
            v108.m128i_i64[1] = 0;
            v108.m128i_i16[0] = 9;
            v115 = _mm_loadu_si128(&v108);
          }
          else
          {
            v77 = (unsigned int)(1 << (32 - v75));
            v78 = sub_3007020(v43, v77);
            v108.m128i_i16[0] = v78;
            v47 = v78;
            v48 = v79;
            v108.m128i_i64[1] = v79;
            v115 = _mm_loadu_si128(&v108);
            if ( v78 )
            {
              if ( (unsigned __int16)(v78 - 17) <= 0xD3u )
              {
                v48 = 0;
                v47 = word_4456580[v78 - 1];
              }
            }
            else
            {
              v95 = v79;
              v80 = sub_30070B0((__int64)&v115);
              v48 = v95;
              if ( v80 )
              {
                v47 = sub_3009970((__int64)&v115, v77, v81, v82, v95);
                v48 = v83;
              }
            }
          }
          goto LABEL_28;
        }
        v47 = 8;
        v108.m128i_i64[1] = 0;
        v108.m128i_i16[0] = 8;
        v115 = _mm_loadu_si128(&v108);
      }
    }
    else
    {
      v47 = 5;
      v108.m128i_i64[1] = 0;
      v108.m128i_i16[0] = 5;
      v115 = _mm_loadu_si128(&v108);
    }
    v48 = 0;
LABEL_28:
    v115.m128i_i16[0] = v47;
    v115.m128i_i64[1] = v48;
    if ( (_WORD)v104 )
    {
      if ( (unsigned __int16)(v104 - 17) <= 0xD3u )
      {
        v84 = word_4456340[(unsigned __int16)v104 - 1];
        if ( (unsigned __int16)(v104 - 176) > 0x34u )
          v47 = sub_2D43050(v47, v84);
        else
          v47 = sub_2D43AD0(v47, v84);
        v48 = 0;
      }
    }
    else
    {
      v94 = v48;
      v73 = sub_30070B0((__int64)&v104);
      v48 = v94;
      if ( v73 )
      {
        v47 = sub_3009490((unsigned __int16 *)&v104, v115.m128i_u32[0], v115.m128i_i64[1]);
        v48 = v74;
      }
    }
    v49 = a1[1];
    LOWORD(v104) = v47;
    v105 = v48;
    v103 = sub_33FAF80(v49, 215, (__int64)&v106, v104, v48, v46, v8);
    *((_QWORD *)&v85 + 1) = v50 | v7 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v85 = v103;
    v52 = sub_3406EB0(
            (_QWORD *)a1[1],
            0x9Eu,
            (__int64)&v106,
            v108.m128i_u32[0],
            v108.m128i_i64[1],
            v51,
            v85,
            *(_OWORD *)&v8);
    result = (__m128i *)sub_33FAFB0(
                          a1[1],
                          (__int64)v52,
                          v53,
                          (__int64)&v106,
                          **(unsigned __int16 **)(a2 + 48),
                          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                          v8);
    goto LABEL_31;
  }
  v30 = *(_QWORD *)(a2 + 80);
  v31 = (_QWORD *)a1[1];
  v32 = *(_QWORD *)(*(_QWORD *)(v21 + 48) + 16LL * v91 + 8);
  v33 = *(unsigned __int16 *)(*(_QWORD *)(v21 + 48) + 16LL * v91);
  v115.m128i_i64[0] = v30;
  if ( v30 )
  {
    v92 = v25;
    sub_B96E90((__int64)&v115, v30, 1);
    v25 = v92;
  }
  v115.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  v34 = sub_3400BD0((__int64)v31, v96 - v25, (__int64)&v115, v33, v32, 0, v8, 0);
  result = (__m128i *)sub_33EC010(v31, (__int64 *)a2, v113, *((unsigned __int64 *)&v113 + 1), (__int64)v34, v35);
  if ( v115.m128i_i64[0] )
  {
    v97 = result;
    sub_B91220((__int64)&v115, v115.m128i_i64[0]);
    return v97;
  }
  return result;
}
