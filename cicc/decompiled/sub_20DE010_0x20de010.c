// Function: sub_20DE010
// Address: 0x20de010
//
__int64 __fastcall sub_20DE010(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  _QWORD *v4; // rdx
  __int64 v5; // r14
  const __m128i *v6; // rax
  const __m128i *v7; // rcx
  __int64 v8; // rdx
  int v9; // esi
  __int64 v10; // r8
  int v11; // esi
  unsigned int v12; // edi
  __int64 *v13; // rdx
  __int64 v14; // r10
  __int64 v15; // rdi
  __int64 *v16; // r15
  __int64 v17; // r13
  _QWORD *v18; // rbx
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // rdx
  _QWORD *v24; // rdx
  __int64 *v25; // rax
  char v26; // dl
  __int64 v27; // rax
  int v28; // ecx
  __int64 v29; // rdx
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 (*v36)(); // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  _QWORD *v40; // rdx
  __int64 v41; // rdx
  unsigned __int64 v42; // rdi
  __int32 v43; // eax
  __m128i *v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  unsigned __int64 v47; // rsi
  __int64 v48; // rax
  __int64 *v50; // rsi
  __int64 *v51; // rcx
  __int64 *v52; // r10
  __int64 *v53; // r9
  __int64 v54; // r12
  unsigned __int64 v55; // r15
  __int64 v56; // r8
  __int64 *v57; // rdi
  unsigned int v58; // r11d
  __int64 *v59; // rdx
  __int64 *v60; // rsi
  char v61; // al
  __int64 v62; // rdi
  int v63; // edx
  _QWORD *v64; // rdx
  const __m128i *v65; // rcx
  __m128i *v66; // rax
  _QWORD *v67; // r14
  _QWORD *v68; // r12
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  _QWORD *v72; // rdx
  _QWORD *v73; // rdx
  unsigned __int64 v74; // rdi
  __int32 v75; // eax
  __m128i *v76; // rsi
  unsigned __int64 v77; // rdx
  unsigned __int8 v78; // al
  __int64 v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rdi
  __int64 (*v82)(); // rax
  char v83; // cl
  __int64 *v84; // r10
  __int64 *v85; // r9
  unsigned __int64 v86; // r13
  __int64 v87; // r15
  __int64 v88; // r8
  __int64 *v89; // rdi
  unsigned int v90; // r11d
  __int64 *v91; // rax
  __int64 *v92; // rsi
  int v93; // eax
  int v94; // r9d
  int v95; // r8d
  __int64 v96; // [rsp+18h] [rbp-248h]
  unsigned __int8 v97; // [rsp+27h] [rbp-239h]
  __int64 v98; // [rsp+28h] [rbp-238h]
  _QWORD *v99; // [rsp+30h] [rbp-230h]
  __int64 *v100; // [rsp+38h] [rbp-228h]
  __int64 v102; // [rsp+40h] [rbp-220h] BYREF
  __int64 v103; // [rsp+48h] [rbp-218h] BYREF
  __m128i v104; // [rsp+50h] [rbp-210h] BYREF
  __int64 v105; // [rsp+60h] [rbp-200h] BYREF
  __int64 *v106; // [rsp+68h] [rbp-1F8h]
  __int64 *v107; // [rsp+70h] [rbp-1F0h]
  __int64 v108; // [rsp+78h] [rbp-1E8h]
  int v109; // [rsp+80h] [rbp-1E0h]
  _BYTE v110[72]; // [rsp+88h] [rbp-1D8h] BYREF
  _BYTE *v111; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v112; // [rsp+D8h] [rbp-188h]
  _BYTE v113[160]; // [rsp+E0h] [rbp-180h] BYREF
  __m128i v114; // [rsp+180h] [rbp-E0h] BYREF
  _BYTE v115[208]; // [rsp+190h] [rbp-D0h] BYREF

  v97 = *(_BYTE *)(a1 + 137);
  if ( !v97 )
    return v97;
  v2 = a1;
  v3 = a2;
  v99 = (_QWORD *)(a2 + 320);
  if ( *(_BYTE *)(a1 + 136) )
  {
LABEL_3:
    v4 = *(_QWORD **)(v3 + 328);
LABEL_4:
    v97 = 0;
    goto LABEL_5;
  }
  v65 = *(const __m128i **)a1;
  v66 = *(__m128i **)a1;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
    *(_QWORD *)(a1 + 8) = v65;
  v67 = *(_QWORD **)(a2 + 328);
  v4 = v67;
  if ( v67 == v99 )
  {
    if ( dword_4FCF4A0 )
      goto LABEL_4;
    goto LABEL_175;
  }
  do
  {
    while ( 1 )
    {
      v71 = v66 - v65;
      if ( dword_4FCF4A0 == v71 )
      {
        v3 = a2;
        goto LABEL_159;
      }
      v72 = *(_QWORD **)(v2 + 40);
      v69 = *(_QWORD **)(v2 + 32);
      if ( v72 == v69 )
      {
        v73 = &v69[*(unsigned int *)(v2 + 52)];
        if ( v69 == v73 )
        {
          v68 = *(_QWORD **)(v2 + 32);
        }
        else
        {
          do
          {
            if ( (_QWORD *)*v69 == v67 )
              break;
            ++v69;
          }
          while ( v73 != v69 );
          v68 = v73;
        }
LABEL_132:
        while ( v73 != v69 )
        {
          if ( *v69 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_121;
          ++v69;
        }
        if ( v69 != v68 )
          goto LABEL_122;
      }
      else
      {
        v68 = &v72[*(unsigned int *)(v2 + 48)];
        v69 = sub_16CC9F0(v2 + 24, (__int64)v67);
        if ( (_QWORD *)*v69 == v67 )
        {
          v79 = *(_QWORD *)(v2 + 40);
          if ( v79 == *(_QWORD *)(v2 + 32) )
            v80 = *(unsigned int *)(v2 + 52);
          else
            v80 = *(unsigned int *)(v2 + 48);
          v73 = (_QWORD *)(v79 + 8 * v80);
          goto LABEL_132;
        }
        v70 = *(_QWORD *)(v2 + 40);
        if ( v70 == *(_QWORD *)(v2 + 32) )
        {
          v73 = (_QWORD *)(v70 + 8LL * *(unsigned int *)(v2 + 52));
          v69 = v73;
          goto LABEL_132;
        }
        v69 = (_QWORD *)(v70 + 8LL * *(unsigned int *)(v2 + 48));
LABEL_121:
        if ( v69 != v68 )
          goto LABEL_122;
      }
      if ( v67[11] == v67[12] )
        break;
LABEL_122:
      v67 = (_QWORD *)v67[1];
      v66 = *(__m128i **)(v2 + 8);
      v65 = *(const __m128i **)v2;
      if ( v67 == v99 )
        goto LABEL_141;
    }
    v74 = sub_1DD6160((__int64)v67);
    if ( (_QWORD *)v74 == v67 + 3 )
      v75 = 0;
    else
      v75 = sub_20D60E0(v74);
    v114.m128i_i32[0] = v75;
    v76 = *(__m128i **)(v2 + 8);
    v114.m128i_i64[1] = (__int64)v67;
    if ( v76 == *(__m128i **)(v2 + 16) )
    {
      sub_20DCBD0((const __m128i **)v2, v76, &v114);
      goto LABEL_122;
    }
    if ( v76 )
    {
      *v76 = _mm_loadu_si128(&v114);
      v76 = *(__m128i **)(v2 + 8);
    }
    v66 = v76 + 1;
    v65 = *(const __m128i **)v2;
    *(_QWORD *)(v2 + 8) = v76 + 1;
    v67 = (_QWORD *)v67[1];
  }
  while ( v67 != v99 );
LABEL_141:
  v3 = a2;
  v77 = (char *)v66 - (char *)v65;
  v71 = v66 - v65;
  if ( dword_4FCF4A0 != v71 )
    goto LABEL_142;
LABEL_159:
  if ( (_DWORD)v71 )
  {
    v84 = *(__int64 **)(v2 + 40);
    v85 = *(__int64 **)(v2 + 32);
    v86 = 0;
    v87 = 16LL * (unsigned int)v71;
    do
    {
      v88 = v65[v86 / 0x10].m128i_i64[1];
      if ( v85 != v84 )
        goto LABEL_161;
      v89 = &v85[*(unsigned int *)(v2 + 52)];
      v90 = *(_DWORD *)(v2 + 52);
      if ( v85 != v89 )
      {
        v91 = v85;
        v92 = 0;
        while ( v88 != *v91 )
        {
          if ( *v91 == -2 )
            v92 = v91;
          if ( v89 == ++v91 )
          {
            if ( !v92 )
              goto LABEL_187;
            *v92 = v88;
            v84 = *(__int64 **)(v2 + 40);
            --*(_DWORD *)(v2 + 56);
            v85 = *(__int64 **)(v2 + 32);
            ++*(_QWORD *)(v2 + 24);
            v65 = *(const __m128i **)v2;
            goto LABEL_162;
          }
        }
        goto LABEL_162;
      }
LABEL_187:
      if ( v90 < *(_DWORD *)(v2 + 48) )
      {
        *(_DWORD *)(v2 + 52) = v90 + 1;
        *v89 = v88;
        v85 = *(__int64 **)(v2 + 32);
        ++*(_QWORD *)(v2 + 24);
        v84 = *(__int64 **)(v2 + 40);
        v65 = *(const __m128i **)v2;
      }
      else
      {
LABEL_161:
        sub_16CCBA0(v2 + 24, v65[v86 / 0x10].m128i_i64[1]);
        v84 = *(__int64 **)(v2 + 40);
        v85 = *(__int64 **)(v2 + 32);
        v65 = *(const __m128i **)v2;
      }
LABEL_162:
      v86 += 16LL;
    }
    while ( v86 != v87 );
  }
LABEL_175:
  v77 = *(_QWORD *)(v2 + 8) - (_QWORD)v65;
LABEL_142:
  if ( v77 <= 0x10 )
    goto LABEL_3;
  v78 = sub_20DDD10((__int64 *)v2, 0, 0, *(_DWORD *)(v2 + 140));
  v4 = *(_QWORD **)(v3 + 328);
  v97 = v78;
LABEL_5:
  v5 = v4[1];
  if ( (_QWORD *)v5 == v99 )
    return v97;
  do
  {
    if ( (unsigned int)((__int64)(*(_QWORD *)(v5 + 72) - *(_QWORD *)(v5 + 64)) >> 3) <= 1 )
      goto LABEL_80;
    v105 = 0;
    v106 = (__int64 *)v110;
    v107 = (__int64 *)v110;
    v108 = 8;
    v109 = 0;
    v98 = *(_QWORD *)v5;
    v6 = *(const __m128i **)v2;
    v7 = *(const __m128i **)v2;
    if ( *(_QWORD *)v2 != *(_QWORD *)(v2 + 8) )
      *(_QWORD *)(v2 + 8) = v6;
    if ( *(_BYTE *)(v2 + 136) )
    {
      v8 = *(_QWORD *)(v2 + 176);
      if ( v8 )
      {
        v9 = *(_DWORD *)(v8 + 256);
        if ( v9 )
        {
          v10 = *(_QWORD *)(v8 + 240);
          v11 = v9 - 1;
          v12 = v11 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v13 = (__int64 *)(v10 + 16LL * v12);
          v14 = *v13;
          if ( v5 == *v13 )
          {
LABEL_13:
            v15 = v13[1];
            v96 = v15;
            if ( v15 && **(_QWORD **)(v15 + 32) == v5 )
              goto LABEL_80;
            goto LABEL_15;
          }
          v63 = 1;
          while ( v14 != -8 )
          {
            v94 = v63 + 1;
            v12 = v11 & (v63 + v12);
            v13 = (__int64 *)(v10 + 16LL * v12);
            v14 = *v13;
            if ( v5 == *v13 )
              goto LABEL_13;
            v63 = v94;
          }
        }
        v96 = 0;
      }
    }
LABEL_15:
    v100 = *(__int64 **)(v5 + 72);
    if ( v100 == *(__int64 **)(v5 + 64) )
      goto LABEL_75;
    v16 = *(__int64 **)(v5 + 64);
    v17 = v2;
    do
    {
      v21 = *(_QWORD *)(v17 + 8);
      v22 = *v16;
      v23 = (v21 - (__int64)v6) >> 4;
      if ( dword_4FCF4A0 == v23 )
      {
        v2 = v17;
        goto LABEL_92;
      }
      v24 = *(_QWORD **)(v17 + 40);
      v19 = *(_QWORD **)(v17 + 32);
      if ( v24 == v19 )
      {
        v18 = &v19[*(unsigned int *)(v17 + 52)];
        if ( v19 == v18 )
        {
          v64 = *(_QWORD **)(v17 + 32);
        }
        else
        {
          do
          {
            if ( v22 == *v19 )
              break;
            ++v19;
          }
          while ( v18 != v19 );
          v64 = v18;
        }
LABEL_31:
        while ( v64 != v19 )
        {
          if ( *v19 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_20;
          ++v19;
        }
        if ( v18 != v19 )
          goto LABEL_21;
      }
      else
      {
        v18 = &v24[*(unsigned int *)(v17 + 48)];
        v19 = sub_16CC9F0(v17 + 24, *v16);
        if ( v22 == *v19 )
        {
          v45 = *(_QWORD *)(v17 + 40);
          if ( v45 == *(_QWORD *)(v17 + 32) )
            v46 = *(unsigned int *)(v17 + 52);
          else
            v46 = *(unsigned int *)(v17 + 48);
          v64 = (_QWORD *)(v45 + 8 * v46);
          goto LABEL_31;
        }
        v20 = *(_QWORD *)(v17 + 40);
        if ( v20 == *(_QWORD *)(v17 + 32) )
        {
          v19 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(v17 + 52));
          v64 = v19;
          goto LABEL_31;
        }
        v19 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(v17 + 48));
LABEL_20:
        if ( v18 != v19 )
          goto LABEL_21;
      }
      if ( v22 == v5 )
        goto LABEL_21;
      v25 = v106;
      if ( v107 != v106 )
        goto LABEL_35;
      v50 = &v106[HIDWORD(v108)];
      if ( v106 != v50 )
      {
        v51 = 0;
        while ( v22 != *v25 )
        {
          if ( *v25 == -2 )
            v51 = v25;
          if ( v50 == ++v25 )
          {
            if ( !v51 )
              goto LABEL_109;
            *v51 = v22;
            --v109;
            ++v105;
            goto LABEL_36;
          }
        }
        goto LABEL_21;
      }
LABEL_109:
      if ( HIDWORD(v108) < (unsigned int)v108 )
      {
        ++HIDWORD(v108);
        *v50 = v22;
        ++v105;
      }
      else
      {
LABEL_35:
        sub_16CCBA0((__int64)&v105, v22);
        if ( !v26 )
          goto LABEL_21;
      }
LABEL_36:
      if ( !(unsigned __int8)sub_1DD61A0(v22) )
      {
        if ( !*(_BYTE *)(v17 + 136) )
          goto LABEL_43;
        v27 = *(_QWORD *)(v17 + 176);
        if ( !v27 )
          goto LABEL_43;
        v28 = *(_DWORD *)(v27 + 256);
        v29 = 0;
        if ( v28 )
        {
          v30 = *(_QWORD *)(v27 + 240);
          v31 = v28 - 1;
          v32 = v31 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v33 = (__int64 *)(v30 + 16LL * v32);
          v34 = *v33;
          if ( v22 == *v33 )
          {
LABEL_41:
            v29 = v33[1];
          }
          else
          {
            v93 = 1;
            while ( v34 != -8 )
            {
              v95 = v93 + 1;
              v32 = v31 & (v93 + v32);
              v33 = (__int64 *)(v30 + 16LL * v32);
              v34 = *v33;
              if ( v22 == *v33 )
                goto LABEL_41;
              v93 = v95;
            }
            v29 = 0;
          }
        }
        if ( v96 == v29 )
        {
LABEL_43:
          v35 = *(_QWORD *)(v17 + 144);
          v102 = 0;
          v111 = v113;
          v103 = 0;
          v112 = 0x400000000LL;
          v36 = *(__int64 (**)())(*(_QWORD *)v35 + 264LL);
          if ( v36 != sub_1D820E0 )
          {
            if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))v36)(
                   v35,
                   v22,
                   &v102,
                   &v103,
                   &v111,
                   1) )
            {
LABEL_69:
              if ( v111 != v113 )
                _libc_free((unsigned __int64)v111);
              goto LABEL_21;
            }
            v114.m128i_i64[0] = (__int64)v115;
            v114.m128i_i64[1] = 0x400000000LL;
            v39 = v102;
            if ( !(_DWORD)v112 )
              goto LABEL_46;
            sub_20D61B0((__int64)&v114, (__int64)&v111, v37, v38, (int)&v111, (int)&v114);
            v39 = v102;
            if ( !(_DWORD)v112 || v102 != v5 )
              goto LABEL_46;
            v81 = *(_QWORD *)(v17 + 144);
            v82 = *(__int64 (**)())(*(_QWORD *)v81 + 624LL);
            if ( v82 == sub_1D918B0 || ((unsigned __int8 (__fastcall *)(__int64, __m128i *))v82)(v81, &v114) )
              goto LABEL_67;
            v39 = v102;
            if ( !v103 )
            {
              v83 = *(_BYTE *)(v5 + 180);
              v40 = *(_QWORD **)(v22 + 8);
              if ( v40 != v99 )
              {
                v103 = *(_QWORD *)(v22 + 8);
                if ( v83 )
                {
LABEL_48:
                  if ( v39 )
                    goto LABEL_49;
LABEL_180:
                  if ( (_QWORD *)v5 != v40 )
                  {
LABEL_67:
                    if ( (_BYTE *)v114.m128i_i64[0] != v115 )
                      _libc_free(v114.m128i_u64[0]);
                    goto LABEL_69;
                  }
LABEL_61:
                  v42 = sub_1DD6160(v22);
                  if ( v42 == v22 + 24 )
                    v43 = 0;
                  else
                    v43 = sub_20D60E0(v42);
                  v104.m128i_i32[0] = v43;
                  v44 = *(__m128i **)(v17 + 8);
                  v104.m128i_i64[1] = v22;
                  if ( v44 == *(__m128i **)(v17 + 16) )
                  {
                    sub_20DCBD0((const __m128i **)v17, v44, &v104);
                  }
                  else
                  {
                    if ( v44 )
                    {
                      *v44 = _mm_loadu_si128(&v104);
                      v44 = *(__m128i **)(v17 + 8);
                    }
                    *(_QWORD *)(v17 + 8) = v44 + 1;
                  }
                  goto LABEL_67;
                }
LABEL_52:
                if ( !v39 || (_DWORD)v112 && !v103 )
                  goto LABEL_61;
LABEL_55:
                sub_1DD6F40(v104.m128i_i64, v22);
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v17 + 144) + 280LL))(
                  *(_QWORD *)(v17 + 144),
                  v22,
                  0);
                if ( (_DWORD)v112 )
                {
                  v41 = v102;
                  if ( v102 == v5 )
                    v41 = v103;
                  (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64, _QWORD, __m128i *, _QWORD))(**(_QWORD **)(v17 + 144) + 288LL))(
                    *(_QWORD *)(v17 + 144),
                    v22,
                    v41,
                    0,
                    v114.m128i_i64[0],
                    v114.m128i_u32[2],
                    &v104,
                    0);
                }
                if ( v104.m128i_i64[0] )
                  sub_161E7C0((__int64)&v104, v104.m128i_i64[0]);
                goto LABEL_61;
              }
              if ( !v83 )
                goto LABEL_52;
              v40 = 0;
              if ( !v102 )
                goto LABEL_67;
            }
            else
            {
LABEL_46:
              if ( !*(_BYTE *)(v5 + 180) )
                goto LABEL_52;
              v40 = *(_QWORD **)(v22 + 8);
              if ( v40 != v99 )
                goto LABEL_48;
              v40 = 0;
              if ( !v39 )
                goto LABEL_67;
LABEL_49:
              if ( v103 )
              {
                if ( v103 != v5 && v5 != v39 )
                  goto LABEL_67;
                goto LABEL_52;
              }
            }
            if ( (_DWORD)v112 )
            {
              if ( v5 != v39 )
                goto LABEL_180;
              goto LABEL_61;
            }
            if ( v5 != v39 )
              goto LABEL_67;
            if ( !v102 )
              goto LABEL_61;
            goto LABEL_55;
          }
        }
      }
LABEL_21:
      v6 = *(const __m128i **)v17;
      ++v16;
      v7 = *(const __m128i **)v17;
    }
    while ( v100 != v16 );
    v2 = v17;
LABEL_75:
    v21 = *(_QWORD *)(v2 + 8);
    v47 = v21 - (_QWORD)v6;
    v23 = (v21 - (__int64)v6) >> 4;
    if ( dword_4FCF4A0 != v23 )
    {
LABEL_76:
      if ( v47 > 0x10 )
        goto LABEL_106;
      goto LABEL_77;
    }
LABEL_92:
    v47 = v21 - (_QWORD)v6;
    if ( !(_DWORD)v23 )
      goto LABEL_76;
    v52 = *(__int64 **)(v2 + 40);
    v53 = *(__int64 **)(v2 + 32);
    v54 = 16LL * (unsigned int)v23;
    v55 = 0;
LABEL_96:
    while ( 2 )
    {
      v56 = v6[v55 / 0x10].m128i_i64[1];
      if ( v53 != v52 )
        goto LABEL_94;
      v57 = &v53[*(unsigned int *)(v2 + 52)];
      v58 = *(_DWORD *)(v2 + 52);
      if ( v53 == v57 )
      {
LABEL_156:
        if ( v58 < *(_DWORD *)(v2 + 48) )
        {
          *(_DWORD *)(v2 + 52) = v58 + 1;
          *v57 = v56;
          v53 = *(__int64 **)(v2 + 32);
          ++*(_QWORD *)(v2 + 24);
          v52 = *(__int64 **)(v2 + 40);
          v6 = *(const __m128i **)v2;
          goto LABEL_95;
        }
LABEL_94:
        sub_16CCBA0(v2 + 24, v6[v55 / 0x10].m128i_i64[1]);
        v52 = *(__int64 **)(v2 + 40);
        v53 = *(__int64 **)(v2 + 32);
        v6 = *(const __m128i **)v2;
        goto LABEL_95;
      }
      v59 = v53;
      v60 = 0;
      while ( v56 != *v59 )
      {
        if ( *v59 == -2 )
          v60 = v59;
        if ( v57 == ++v59 )
        {
          if ( !v60 )
            goto LABEL_156;
          *v60 = v56;
          v6 = *(const __m128i **)v2;
          v55 += 16LL;
          --*(_DWORD *)(v2 + 56);
          v52 = *(__int64 **)(v2 + 40);
          ++*(_QWORD *)(v2 + 24);
          v53 = *(__int64 **)(v2 + 32);
          v7 = v6;
          if ( v55 != v54 )
            goto LABEL_96;
          goto LABEL_105;
        }
      }
LABEL_95:
      v55 += 16LL;
      v7 = v6;
      if ( v55 != v54 )
        continue;
      break;
    }
LABEL_105:
    v47 = *(_QWORD *)(v2 + 8) - (_QWORD)v6;
    if ( v47 > 0x10 )
    {
LABEL_106:
      v61 = sub_20DDD10((__int64 *)v2, v5, (__int64 *)(v98 & 0xFFFFFFFFFFFFFFF8LL), *(_DWORD *)(v2 + 140));
      v7 = *(const __m128i **)v2;
      v97 |= v61;
      v48 = *(_QWORD *)v5;
      if ( *(_QWORD *)(v2 + 8) - *(_QWORD *)v2 == 16 )
        goto LABEL_107;
      goto LABEL_78;
    }
LABEL_77:
    v48 = *(_QWORD *)v5;
    if ( v47 == 16 )
    {
LABEL_107:
      v62 = v7->m128i_i64[1];
      if ( v62 != (v48 & 0xFFFFFFFFFFFFFFF8LL) )
        sub_20D69B0(v62, v5, *(_QWORD *)(v2 + 144));
    }
LABEL_78:
    if ( v107 != v106 )
      _libc_free((unsigned __int64)v107);
LABEL_80:
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( (_QWORD *)v5 != v99 );
  return v97;
}
