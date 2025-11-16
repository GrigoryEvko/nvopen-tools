// Function: sub_3322160
// Address: 0x3322160
//
unsigned __int64 __fastcall sub_3322160(
        __int64 *a1,
        size_t a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  int v8; // r13d
  __int64 v9; // r15
  __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rax
  unsigned __int64 result; // rax
  unsigned __int16 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  bool (__fastcall *v17)(__int64, __int64, unsigned __int16); // r8
  __int64 (*v18)(); // rax
  __int64 *v19; // rax
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned __int64 v30; // r15
  __int64 v31; // rdi
  __int128 v32; // rax
  int v33; // r9d
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rdx
  bool v38; // al
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // r8
  char (__fastcall *v43)(__int64, unsigned int); // rdx
  unsigned __int16 *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  bool (__fastcall *v47)(__int64, __int64, unsigned __int16); // r9
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // esi
  int v52; // edi
  int v53; // eax
  int v54; // eax
  int v55; // eax
  int v56; // ecx
  __int64 v57; // rdi
  __int64 v58; // rdx
  int v59; // edx
  unsigned __int16 *v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  bool (__fastcall *v63)(__int64, __int64, unsigned __int16); // r8
  __int64 (*v64)(); // rax
  char v65; // r8
  unsigned __int16 *v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  bool (__fastcall *v69)(__int64, __int64, unsigned __int16); // r8
  __int64 (*v70)(); // rax
  __int64 v71; // r15
  int v72; // r9d
  bool v73; // al
  bool v74; // al
  __int64 (*v75)(); // rax
  char v76; // al
  __int64 *v77; // rax
  __int64 v78; // r15
  unsigned int v79; // edx
  unsigned __int64 v80; // r9
  unsigned int v81; // eax
  unsigned __int64 v82; // r15
  __int64 v83; // rsi
  __int64 v84; // rdi
  __int128 v85; // rax
  int v86; // r9d
  __int64 v87; // r13
  __int64 v88; // rsi
  __int64 v89; // rdi
  __int64 v90; // rsi
  unsigned __int8 v91; // al
  int v92; // r11d
  __int128 v93; // rax
  __int64 v94; // r14
  int v95; // r9d
  __int64 v96; // r13
  __int64 v97; // rdx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // r8
  __int64 v101; // r9
  unsigned int v102; // edx
  __int64 v103; // rax
  unsigned __int64 v104; // rt2
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  __int128 v108; // [rsp-20h] [rbp-130h]
  __int128 v109; // [rsp-10h] [rbp-120h]
  __int64 v110; // [rsp+0h] [rbp-110h]
  __int64 v111; // [rsp+8h] [rbp-108h]
  __int128 v112; // [rsp+10h] [rbp-100h]
  int v113; // [rsp+28h] [rbp-E8h]
  int v114; // [rsp+2Ch] [rbp-E4h]
  char v115; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v116; // [rsp+38h] [rbp-D8h]
  __int64 v117; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v118; // [rsp+40h] [rbp-D0h]
  __int64 v119; // [rsp+48h] [rbp-C8h]
  __int128 *v120; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v121; // [rsp+48h] [rbp-C8h]
  char v122; // [rsp+8Eh] [rbp-82h] BYREF
  char v123; // [rsp+8Fh] [rbp-81h] BYREF
  __m128i v124; // [rsp+90h] [rbp-80h] BYREF
  __m128i v125; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v126; // [rsp+B0h] [rbp-60h] BYREF
  _QWORD *v127; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v128; // [rsp+C8h] [rbp-48h]
  __int64 v129; // [rsp+D0h] [rbp-40h]
  int v130; // [rsp+D8h] [rbp-38h]

  if ( !*((_BYTE *)a1 + 36) )
  {
    result = sub_331F6A0(a1, a2, a3, a4, a5, a6);
    if ( result )
      return result;
  }
  v8 = *(_DWORD *)(a2 + 24);
  v9 = a1[1];
  v10 = (unsigned int)v8;
  if ( (unsigned int)v8 > 0x1F3 || (v59 = *(unsigned __int8 *)(v9 + (v8 >> 3) + 525170), _bittest(&v59, v8 & 7)) )
  {
    v11 = *((_DWORD *)a1 + 6);
    v12 = *a1;
    v127 = a1;
    BYTE4(v128) = 0;
    LODWORD(v128) = v11;
    v129 = v12;
    result = (*(__int64 (__fastcall **)(__int64, size_t, _QWORD **))(*(_QWORD *)v9 + 2144LL))(v9, a2, &v127);
    if ( result )
      return result;
    v8 = *(_DWORD *)(a2 + 24);
    v9 = a1[1];
    v10 = (unsigned int)v8;
  }
  if ( v8 <= 215 )
  {
    if ( v8 <= 212 )
    {
      if ( v8 > 192 )
        goto LABEL_39;
      if ( v8 <= 189 )
      {
        if ( v8 <= 58 )
        {
          if ( v8 > 55 )
            goto LABEL_12;
        }
        else if ( (unsigned int)(v8 - 186) <= 2 )
        {
LABEL_12:
          if ( !*((_BYTE *)a1 + 33) )
            goto LABEL_75;
          v14 = *(unsigned __int16 **)(a2 + 48);
          v15 = *v14;
          v16 = *((_QWORD *)v14 + 1);
          v124.m128i_i16[0] = v15;
          v124.m128i_i64[1] = v16;
          if ( (_WORD)v15 )
          {
            if ( (unsigned __int16)(v15 - 17) <= 0xD3u
              || (unsigned __int16)(v15 - 2) > 7u && (unsigned __int16)(v15 - 176) > 0x1Fu )
            {
              goto LABEL_75;
            }
            v17 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
            if ( v17 == sub_302E170 )
            {
              if ( !*(_QWORD *)(v9 + 8 * v15 + 112) )
                goto LABEL_19;
LABEL_75:
              v42 = v9;
              v10 = (unsigned int)v8;
              goto LABEL_40;
            }
          }
          else
          {
            if ( sub_30070B0((__int64)&v124) || !sub_3007070((__int64)&v124) )
              goto LABEL_75;
            v17 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
            if ( v17 == sub_302E170 )
            {
LABEL_19:
              v125 = _mm_loadu_si128(&v124);
              v18 = *(__int64 (**)())(*(_QWORD *)v9 + 2208LL);
              if ( v18 != sub_302E1A0 )
              {
                v115 = ((__int64 (__fastcall *)(__int64, size_t, _QWORD, __m128i *))v18)(v9, a2, 0, &v125);
                if ( v115 )
                {
                  v19 = *(__int64 **)(a2 + 40);
                  v122 = 0;
                  v119 = *v19;
                  v114 = *((_DWORD *)v19 + 2);
                  *(_QWORD *)&v112 = sub_32EA990(a1, *v19, v19[1], v125.m128i_u32[0], v125.m128i_i64[1], &v122);
                  v20 = v112;
                  v21 = *(_QWORD *)(a2 + 40);
                  *((_QWORD *)&v112 + 1) = v22;
                  v23 = *(_QWORD *)(v21 + 40);
                  v24 = *(_QWORD *)(v21 + 48);
                  v123 = 0;
                  v117 = v23;
                  v113 = *(_DWORD *)(v21 + 48);
                  v25 = sub_32EA990(a1, v23, v24, v125.m128i_u32[0], v125.m128i_i64[1], &v123);
                  v27 = *(_QWORD *)(a2 + 80);
                  v28 = v25;
                  v29 = v26;
                  v30 = v25;
                  v126.m128i_i64[0] = v27;
                  if ( v27 )
                  {
                    v111 = v26;
                    v110 = v25;
                    sub_B96E90((__int64)&v126, v27, 1);
                    v28 = v110;
                    v29 = v111;
                  }
                  v31 = *a1;
                  *((_QWORD *)&v109 + 1) = v29;
                  *(_QWORD *)&v109 = v28;
                  v126.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                  *(_QWORD *)&v32 = sub_3406EB0(
                                      v31,
                                      v8,
                                      (unsigned int)&v126,
                                      v125.m128i_i32[0],
                                      v125.m128i_i32[2],
                                      v29,
                                      v112,
                                      v109);
                  v34 = sub_33FAF80(v31, 216, (unsigned int)&v126, v124.m128i_i32[0], v124.m128i_i32[2], v33, v32);
                  v36 = v35;
                  v37 = *(_QWORD *)(v119 + 56);
                  v38 = 0;
                  if ( v37 )
                    v38 = *(_QWORD *)(v37 + 32) == 0;
                  v122 &= !v38;
                  if ( v114 == v113 && v119 == v117 )
                  {
                    v115 = 0;
                  }
                  else
                  {
                    v39 = *(_QWORD *)(v117 + 56);
                    if ( v39 )
                      v115 = *(_QWORD *)(v39 + 32) != 0;
                  }
                  v127 = (_QWORD *)v34;
                  v128 = v36;
                  v123 &= v115;
                  sub_32EB790((__int64)a1, a2, (__int64 *)&v127, 1, 1);
                  if ( v122 )
                  {
                    if ( !v123 )
                      goto LABEL_169;
                    if ( (unsigned __int8)sub_33CFFC0(v117, v119) )
                    {
                      v103 = v119;
                      v119 = v117;
                      v104 = v30;
                      v30 = v112;
                      v20 = v104;
                      v117 = v103;
                    }
                    if ( v122 )
                    {
LABEL_169:
                      if ( *(_DWORD *)(v20 + 24) != 328 )
                      {
                        v127 = (_QWORD *)v20;
                        sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v127);
                        if ( *(int *)(v20 + 88) < 0 )
                        {
                          *(_DWORD *)(v20 + 88) = *((_DWORD *)a1 + 12);
                          v107 = *((unsigned int *)a1 + 12);
                          if ( v107 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
                          {
                            sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v107 + 1, 8u, v40, v41);
                            v107 = *((unsigned int *)a1 + 12);
                          }
                          *(_QWORD *)(a1[5] + 8 * v107) = v20;
                          ++*((_DWORD *)a1 + 12);
                        }
                      }
                      sub_32EA6D0(a1, v119, v20);
                    }
                  }
                  if ( v123 )
                  {
                    if ( *(_DWORD *)(v30 + 24) != 328 )
                    {
                      v127 = (_QWORD *)v30;
                      sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v127);
                      if ( *(int *)(v30 + 88) < 0 )
                      {
                        *(_DWORD *)(v30 + 88) = *((_DWORD *)a1 + 12);
                        v105 = *((unsigned int *)a1 + 12);
                        if ( v105 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
                        {
                          sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v105 + 1, 8u, v100, v101);
                          v105 = *((unsigned int *)a1 + 12);
                        }
                        *(_QWORD *)(a1[5] + 8 * v105) = v30;
                        ++*((_DWORD *)a1 + 12);
                      }
                    }
                    sub_32EA6D0(a1, v117, v30);
                  }
                  if ( v126.m128i_i64[0] )
                    sub_B91220((__int64)&v126, v126.m128i_i64[0]);
                  return a2;
                }
                goto LABEL_117;
              }
              goto LABEL_74;
            }
          }
          if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v17)(
                  v9,
                  (unsigned int)v8,
                  v124.m128i_u32[0],
                  v124.m128i_i64[1]) )
          {
            v9 = a1[1];
            goto LABEL_19;
          }
LABEL_117:
          v8 = *(_DWORD *)(a2 + 24);
          v9 = a1[1];
          goto LABEL_75;
        }
LABEL_39:
        v42 = v9;
        goto LABEL_40;
      }
      if ( !*((_BYTE *)a1 + 33) )
        goto LABEL_75;
      v60 = *(unsigned __int16 **)(a2 + 48);
      v61 = *v60;
      v62 = *((_QWORD *)v60 + 1);
      v125.m128i_i16[0] = v61;
      v125.m128i_i64[1] = v62;
      if ( (_WORD)v61 )
      {
        if ( (unsigned __int16)(v61 - 17) <= 0xD3u
          || (unsigned __int16)(v61 - 2) > 7u && (unsigned __int16)(v61 - 176) > 0x1Fu )
        {
          goto LABEL_75;
        }
        v63 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
        if ( v63 == sub_302E170 )
        {
          if ( *(_QWORD *)(v9 + 8 * v61 + 112) )
            goto LABEL_75;
          goto LABEL_73;
        }
      }
      else
      {
        if ( sub_30070B0((__int64)&v125) || !sub_3007070((__int64)&v125) )
          goto LABEL_75;
        v63 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
        if ( v63 == sub_302E170 )
        {
LABEL_73:
          v126 = _mm_loadu_si128(&v125);
          v64 = *(__int64 (**)())(*(_QWORD *)v9 + 2208LL);
          if ( v64 == sub_302E1A0 )
          {
LABEL_74:
            v8 = *(_DWORD *)(a2 + 24);
            goto LABEL_75;
          }
          if ( !((unsigned __int8 (__fastcall *)(__int64, size_t, _QWORD, __m128i *))v64)(v9, a2, 0, &v126) )
            goto LABEL_117;
          v77 = *(__int64 **)(a2 + 40);
          v124.m128i_i8[0] = 0;
          v78 = v77[1];
          if ( v8 == 191 )
          {
            v80 = sub_32EAC70(a1, *v77, v78, v126.m128i_u32[0], v126.m128i_i64[1]);
            v81 = v102;
          }
          else
          {
            if ( v8 == 192 )
              v80 = sub_32EA820(a1, *v77, v78, v126.m128i_u32[0], v126.m128i_i64[1]);
            else
              v80 = sub_32EA990(a1, *v77, v78, v126.m128i_u32[0], v126.m128i_i64[1], &v124);
            v81 = v79;
          }
          v82 = v81 | v78 & 0xFFFFFFFF00000000LL;
          if ( !v80 )
          {
            v8 = *(_DWORD *)(a2 + 24);
            v9 = a1[1];
            goto LABEL_75;
          }
          v83 = *(_QWORD *)(a2 + 80);
          v127 = (_QWORD *)v83;
          if ( v83 )
          {
            v118 = v80;
            sub_B96E90((__int64)&v127, v83, 1);
            v80 = v118;
          }
          v84 = *a1;
          v116 = v80;
          LODWORD(v128) = *(_DWORD *)(a2 + 72);
          *((_QWORD *)&v108 + 1) = v82;
          *(_QWORD *)&v108 = v80;
          *(_QWORD *)&v85 = sub_3406EB0(
                              v84,
                              v8,
                              (unsigned int)&v127,
                              v126.m128i_i32[0],
                              v126.m128i_i32[2],
                              v80,
                              v108,
                              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
          v87 = sub_33FAF80(v84, 216, (unsigned int)&v127, v125.m128i_i32[0], v125.m128i_i32[2], v86, v85);
          if ( v124.m128i_i8[0] )
            sub_32EA6D0(a1, **(_QWORD **)(a2 + 40), v116);
          if ( !*(_DWORD *)(a2 + 24) )
          {
            if ( v127 )
            {
              sub_B91220((__int64)&v127, (__int64)v127);
              v8 = *(_DWORD *)(a2 + 24);
              v9 = a1[1];
            }
            else
            {
              v9 = a1[1];
              v8 = 0;
            }
            goto LABEL_75;
          }
          result = v87;
          if ( v127 )
          {
            sub_B91220((__int64)&v127, (__int64)v127);
            result = v87;
          }
LABEL_95:
          if ( result )
            return result;
          v8 = *(_DWORD *)(a2 + 24);
          v9 = a1[1];
          goto LABEL_75;
        }
      }
      if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v63)(
             v9,
             (unsigned int)v8,
             v125.m128i_u32[0],
             v125.m128i_i64[1]) )
      {
        goto LABEL_117;
      }
      v9 = a1[1];
      goto LABEL_73;
    }
    if ( !*((_BYTE *)a1 + 33) )
      goto LABEL_75;
    v66 = *(unsigned __int16 **)(a2 + 48);
    v67 = *v66;
    v68 = *((_QWORD *)v66 + 1);
    v125.m128i_i16[0] = v67;
    v125.m128i_i64[1] = v68;
    if ( (_WORD)v67 )
    {
      if ( (unsigned __int16)(v67 - 17) <= 0xD3u
        || (unsigned __int16)(v67 - 2) > 7u && (unsigned __int16)(v67 - 176) > 0x1Fu )
      {
        goto LABEL_75;
      }
      v69 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
      if ( v69 == sub_302E170 )
      {
        if ( *(_QWORD *)(v9 + 8 * v67 + 112) )
          goto LABEL_75;
        goto LABEL_91;
      }
    }
    else
    {
      if ( sub_30070B0((__int64)&v125) || !sub_3007070((__int64)&v125) )
        goto LABEL_75;
      v69 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
      if ( v69 == sub_302E170 )
      {
LABEL_91:
        v126 = _mm_loadu_si128(&v125);
        v70 = *(__int64 (**)())(*(_QWORD *)v9 + 2208LL);
        if ( v70 == sub_302E1A0 )
          goto LABEL_74;
        if ( !((unsigned __int8 (__fastcall *)(__int64, size_t, _QWORD, __m128i *))v70)(v9, a2, 0, &v126) )
          goto LABEL_117;
        v71 = *a1;
        v120 = *(__int128 **)(a2 + 40);
        sub_3285E70((__int64)&v127, a2);
        result = sub_33FAF80(
                   v71,
                   *(_DWORD *)(a2 + 24),
                   (unsigned int)&v127,
                   v125.m128i_i32[0],
                   v125.m128i_i32[2],
                   v72,
                   *v120);
        if ( v127 )
        {
          v121 = result;
          sub_B91220((__int64)&v127, (__int64)v127);
          result = v121;
        }
        goto LABEL_95;
      }
    }
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v69)(
           v9,
           (unsigned int)v8,
           v125.m128i_u32[0],
           v125.m128i_i64[1]) )
    {
      goto LABEL_117;
    }
    v9 = a1[1];
    goto LABEL_91;
  }
  if ( v8 != 298 )
    goto LABEL_39;
  v42 = v9;
  if ( !*((_BYTE *)a1 + 33) || (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    goto LABEL_81;
  v44 = *(unsigned __int16 **)(a2 + 48);
  v45 = *v44;
  v46 = *((_QWORD *)v44 + 1);
  v125.m128i_i16[0] = v45;
  v125.m128i_i64[1] = v46;
  if ( (_WORD)v45 )
  {
    if ( (unsigned __int16)(v45 - 17) > 0xD3u
      && ((unsigned __int16)(v45 - 2) <= 7u || (unsigned __int16)(v45 - 176) <= 0x1Fu) )
    {
      v47 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
      if ( v47 == sub_302E170 )
      {
        if ( *(_QWORD *)(v9 + 8 * v45 + 112) )
        {
          v43 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v9 + 1360LL);
          if ( v43 == sub_2FE3400 )
            goto LABEL_56;
LABEL_82:
          v65 = v43(v42, v10);
          result = 0;
          if ( !v65 )
            return result;
LABEL_57:
          v48 = *(_QWORD *)(a2 + 40);
          v49 = *(_QWORD *)v48;
          v50 = *(_QWORD *)(v48 + 40);
          v51 = *(_DWORD *)(v48 + 8);
          v52 = *(_DWORD *)(v48 + 48);
          if ( *(_QWORD *)v48 != v50 || v51 != v52 )
          {
            v53 = *(_DWORD *)(v49 + 24);
            if ( v53 == 11 || v53 == 35 || (v54 = *(_DWORD *)(v50 + 24), v54 != 35) && v54 != 11 )
            {
              v55 = *(_DWORD *)(a2 + 28);
              v127 = (_QWORD *)v50;
              LODWORD(v128) = v52;
              v56 = *(_DWORD *)(a2 + 68);
              v57 = *a1;
              v129 = v49;
              v58 = *(_QWORD *)(a2 + 48);
              v130 = v51;
              return sub_33D00B0(v57, *(_DWORD *)(a2 + 24), v58, v56, (unsigned int)&v127, 2, v55);
            }
          }
          return 0;
        }
        goto LABEL_115;
      }
      goto LABEL_118;
    }
LABEL_81:
    v10 = 298;
    v43 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v9 + 1360LL);
    result = 0;
    if ( v43 == sub_2FE3400 )
      return result;
    goto LABEL_82;
  }
  v73 = sub_30070B0((__int64)&v125);
  v42 = v9;
  if ( v73 )
    goto LABEL_81;
  v74 = sub_3007070((__int64)&v125);
  v42 = v9;
  if ( !v74 )
    goto LABEL_81;
  v47 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v9 + 2192LL);
  if ( v47 == sub_302E170 )
    goto LABEL_115;
LABEL_118:
  v76 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v47)(
          v9,
          298,
          v125.m128i_u32[0],
          v125.m128i_i64[1],
          v42);
  v42 = a1[1];
  if ( !v76 )
  {
LABEL_115:
    v126 = _mm_loadu_si128(&v125);
    v75 = *(__int64 (**)())(*(_QWORD *)v42 + 2208LL);
    if ( v75 != sub_302E1A0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, size_t, _QWORD, __m128i *))v75)(v42, a2, 0, &v126) )
      {
        v88 = *(_QWORD *)(a2 + 80);
        v127 = (_QWORD *)v88;
        if ( v88 )
          sub_B96E90((__int64)&v127, v88, 1);
        v89 = *(_QWORD *)(a2 + 104);
        v90 = *(unsigned __int16 *)(a2 + 96);
        LODWORD(v128) = *(_DWORD *)(a2 + 72);
        v91 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
        if ( *(_DWORD *)(a2 + 24) != 298 || (v92 = 1, v91) )
          v92 = v91;
        *(_QWORD *)&v93 = sub_33F1B30(
                            *a1,
                            v92,
                            (unsigned int)&v127,
                            v126.m128i_i32[0],
                            v126.m128i_i32[2],
                            *(_QWORD *)(a2 + 112),
                            **(_QWORD **)(a2 + 40),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                            v90,
                            v89);
        v94 = v93;
        v96 = sub_33FAF80(*a1, 216, (unsigned int)&v127, v125.m128i_i32[0], v125.m128i_i32[2], v95, v93);
        sub_34161C0(*a1, a2, 0, v96, v97);
        sub_34161C0(*a1, a2, 1, v94, 1);
        if ( *(_DWORD *)(v96 + 24) != 328 )
        {
          v124.m128i_i64[0] = v96;
          sub_32B3B20((__int64)(a1 + 71), v124.m128i_i64);
          if ( *(int *)(v96 + 88) < 0 )
          {
            *(_DWORD *)(v96 + 88) = *((_DWORD *)a1 + 12);
            v106 = *((unsigned int *)a1 + 12);
            if ( v106 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
            {
              sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v106 + 1, 8u, v98, v99);
              v106 = *((unsigned int *)a1 + 12);
            }
            *(_QWORD *)(a1[5] + 8 * v106) = v96;
            ++*((_DWORD *)a1 + 12);
          }
        }
        sub_32CF870((__int64)a1, a2);
        if ( v127 )
          sub_B91220((__int64)&v127, (__int64)v127);
        return a2;
      }
      v42 = a1[1];
    }
  }
  v10 = *(unsigned int *)(a2 + 24);
LABEL_40:
  v43 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v42 + 1360LL);
  if ( v43 != sub_2FE3400 )
    goto LABEL_82;
  if ( (int)v10 <= 98 )
  {
    if ( (int)v10 > 55 )
    {
      switch ( (int)v10 )
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
          goto LABEL_57;
        default:
          return 0;
      }
    }
    return 0;
  }
  if ( (int)v10 > 188 )
  {
LABEL_56:
    result = 0;
    if ( (unsigned int)(v10 - 279) > 7 )
      return result;
    goto LABEL_57;
  }
  if ( (int)v10 > 185 )
    goto LABEL_57;
  result = 0;
  if ( (unsigned int)(v10 - 172) <= 0xB )
    goto LABEL_57;
  return result;
}
