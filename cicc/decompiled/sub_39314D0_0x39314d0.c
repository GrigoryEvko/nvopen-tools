// Function: sub_39314D0
// Address: 0x39314d0
//
_QWORD *__fastcall sub_39314D0(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rax
  unsigned __int64 v20; // rsi
  _QWORD *result; // rax
  const char *v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // r11
  __int64 v25; // r8
  __int64 v26; // r14
  unsigned __int32 v27; // eax
  __int64 v28; // r9
  unsigned __int16 v29; // cx
  char v30; // r13
  __int64 v31; // r12
  void *v32; // rax
  unsigned __int64 v33; // r8
  __int64 v34; // r8
  __int64 v35; // r13
  int v36; // edx
  int v37; // edx
  __int64 v38; // rsi
  unsigned int v39; // ecx
  __int64 *v40; // rax
  __int64 v41; // r8
  unsigned int v42; // esi
  __int64 v43; // r9
  unsigned int v44; // ecx
  __int64 v45; // rdx
  __m128i *v46; // rsi
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rsi
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int32 v52; // eax
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  int v55; // eax
  void *v56; // rax
  char v57; // al
  char v58; // al
  unsigned __int64 v59; // rax
  __int64 v60; // rax
  unsigned int v61; // esi
  __int64 v62; // r9
  unsigned int v63; // ecx
  __int64 v64; // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // rax
  unsigned __int64 v68; // rax
  void *v69; // rdx
  __int64 v70; // rax
  unsigned __int64 v71; // rax
  unsigned __int64 v72; // rax
  unsigned __int64 *v73; // rax
  __int64 v74; // rax
  int v75; // r11d
  _QWORD *v76; // rdi
  int v77; // eax
  int v78; // edx
  unsigned __int64 *v79; // rdi
  int v80; // eax
  int v81; // ecx
  __int64 v82; // r8
  unsigned int v83; // eax
  __int64 v84; // rsi
  int v85; // r10d
  _QWORD *v86; // r9
  int v87; // eax
  int v88; // eax
  __int64 v89; // rsi
  int v90; // r9d
  unsigned int v91; // r12d
  _QWORD *v92; // r8
  __int64 v93; // rcx
  int v94; // eax
  int v95; // r11d
  int v96; // eax
  int v97; // eax
  int v98; // eax
  __int64 v99; // rsi
  int v100; // r9d
  unsigned int v101; // r12d
  __int64 v102; // rcx
  unsigned __int64 v103; // rax
  unsigned __int64 v104; // rdx
  __int64 v105; // rax
  unsigned __int64 v106; // rax
  int v107; // eax
  int v108; // eax
  int v109; // ecx
  __int64 v110; // r8
  unsigned int v111; // eax
  __int64 v112; // rsi
  int v113; // r10d
  unsigned __int64 v114; // rax
  void *v115; // rdx
  __int64 v116; // rax
  int v117; // edi
  char v118; // [rsp+7h] [rbp-C9h]
  __int64 v119; // [rsp+8h] [rbp-C8h]
  __int64 v120; // [rsp+10h] [rbp-C0h]
  __int64 v121; // [rsp+10h] [rbp-C0h]
  __int64 v122; // [rsp+10h] [rbp-C0h]
  __int64 v123; // [rsp+10h] [rbp-C0h]
  __int64 v124; // [rsp+10h] [rbp-C0h]
  __int64 v125; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v127; // [rsp+18h] [rbp-B8h]
  __int64 v128; // [rsp+18h] [rbp-B8h]
  __int64 v129; // [rsp+20h] [rbp-B0h]
  int v130; // [rsp+28h] [rbp-A8h]
  unsigned __int32 v131; // [rsp+28h] [rbp-A8h]
  __int64 v132; // [rsp+28h] [rbp-A8h]
  __int64 v133; // [rsp+28h] [rbp-A8h]
  __int64 v134; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v135; // [rsp+38h] [rbp-98h]
  _QWORD v136[2]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v137[2]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v138; // [rsp+60h] [rbp-70h]
  __m128i v139; // [rsp+70h] [rbp-60h] BYREF
  __m128i v140; // [rsp+80h] [rbp-50h] BYREF
  __m128i v141; // [rsp+90h] [rbp-40h] BYREF

  v130 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 48LL))(
                       *(_QWORD *)(a2 + 8),
                       *(unsigned int *)(a5 + 12))
                   + 16);
  v135 = *(_QWORD *)(a4 + 24);
  v134 = a9;
  v14 = sub_38D01B0((__int64)a3, a4);
  v15 = *(_QWORD *)a2;
  v129 = *(unsigned int *)(a5 + 8) + v14;
  if ( !a8 )
  {
    v24 = a7;
    v25 = v130 & 1;
    if ( !a7 )
    {
LABEL_43:
      v121 = v15;
      v52 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
              *(_QWORD *)(a1 + 8),
              v15,
              &a7,
              a5,
              v25);
      v118 = 0;
      v28 = v121;
      v131 = v52;
      goto LABEL_44;
    }
    goto LABEL_12;
  }
  if ( (v130 & 1) != 0 )
  {
    v140.m128i_i8[1] = 1;
    v22 = "No relocation available to represent this relative expression";
LABEL_9:
    v23 = *(_QWORD *)(a5 + 16);
    v139.m128i_i64[0] = (__int64)v22;
    v140.m128i_i8[0] = 3;
    return sub_38BE3D0(v15, v23, (__int64)&v139);
  }
  v16 = *(_QWORD *)(a8 + 24);
  v17 = *(_QWORD *)v16;
  v18 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v18 )
  {
    v50 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_48:
    if ( v135 != *(_QWORD *)(v50 + 24) )
      goto LABEL_49;
    goto LABEL_42;
  }
  if ( (*(_BYTE *)(v16 + 9) & 0xC) != 8 )
    goto LABEL_5;
  *(_BYTE *)(v16 + 8) |= 4u;
  v132 = v15;
  v47 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v16 + 24));
  v15 = v132;
  v18 = 0;
  v48 = v47;
  v49 = v47 | *(_QWORD *)v16 & 7LL;
  *(_QWORD *)v16 = v49;
  LOBYTE(v17) = v49;
  if ( !v48 )
  {
LABEL_5:
    v19 = 0;
    if ( (v17 & 4) != 0 )
    {
      v73 = *(unsigned __int64 **)(v16 - 8);
      v18 = *v73;
      v19 = v73 + 2;
    }
    v136[0] = v19;
    v20 = *(_QWORD *)(a5 + 16);
    v137[0] = "symbol '";
    v137[1] = v136;
    v136[1] = v18;
    v138 = 1283;
    v139.m128i_i64[0] = (__int64)v137;
    v140.m128i_i16[0] = 770;
    v139.m128i_i64[1] = (__int64)"' can not be undefined in a subtraction expression";
    return sub_38BE3D0(v15, v20, (__int64)&v139);
  }
  v50 = v49 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v50 )
    goto LABEL_48;
  if ( (*(_BYTE *)(v16 + 9) & 0xC) == 8 )
  {
    *(_BYTE *)(v16 + 8) |= 4u;
    v50 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v16 + 24));
    v15 = v132;
    *(_QWORD *)v16 = v50 | *(_QWORD *)v16 & 7LL;
    if ( v50 )
      goto LABEL_48;
  }
  if ( v135 )
  {
LABEL_49:
    v140.m128i_i8[1] = 1;
    v22 = "Cannot represent a difference across sections";
    goto LABEL_9;
  }
LABEL_42:
  v133 = v15;
  v51 = sub_38D0440(a3, v16);
  v24 = a7;
  v25 = 1;
  v15 = v133;
  v134 = v129 + v134 - v51;
  if ( !a7 )
    goto LABEL_43;
LABEL_12:
  v26 = *(_QWORD *)(v24 + 24);
  v118 = 0;
  if ( v26 )
  {
    if ( (*(_BYTE *)(v26 + 9) & 0xC) == 8 )
    {
      v53 = *(_QWORD *)(v26 + 24);
      *(_BYTE *)(v26 + 8) |= 4u;
      if ( *(_DWORD *)v53 == 2 && *(_WORD *)(v53 + 16) == 27 )
      {
        v118 = 1;
        v26 = *(_QWORD *)(v53 + 24);
      }
    }
  }
  v119 = v24;
  v120 = v15;
  v27 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
          *(_QWORD *)(a1 + 8),
          v15,
          &a7,
          a5,
          v25);
  v28 = v120;
  v131 = v27;
  v29 = *(_WORD *)(v119 + 16);
  if ( v29 > 0x38u )
    goto LABEL_197;
  if ( ((1LL << v29) & 0xE0000000000424LL) != 0 )
  {
    v31 = v134;
    v65 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 12LL) & 1) == 0 )
    {
      v31 = 0;
      v65 = v134;
    }
    *a6 = v65;
    if ( !v26 )
    {
      v33 = 0;
      v30 = 1;
      goto LABEL_23;
    }
    goto LABEL_62;
  }
  if ( ((1LL << v29) & 0x100000000000000LL) == 0 )
  {
LABEL_197:
    if ( (*(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      if ( (*(_BYTE *)(v26 + 9) & 0xC) != 8 )
        goto LABEL_19;
      *(_BYTE *)(v26 + 8) |= 4u;
      v54 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
      v28 = v120;
      *(_QWORD *)v26 = v54 | *(_QWORD *)v26 & 7LL;
      if ( !v54 )
        goto LABEL_19;
    }
    v122 = v28;
    v55 = sub_38E27C0(v26);
    v28 = v122;
    if ( !v55 )
    {
      v56 = (void *)(*(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v56 )
        goto LABEL_145;
      if ( (*(_BYTE *)(v26 + 9) & 0xC) != 8 )
        goto LABEL_57;
      *(_BYTE *)(v26 + 8) |= 4u;
      v103 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
      v28 = v122;
      v104 = v103;
      v105 = v103 | *(_QWORD *)v26 & 7LL;
      *(_QWORD *)v26 = v105;
      if ( !v104 )
        goto LABEL_57;
      v56 = (void *)(v105 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v56 )
      {
LABEL_145:
        if ( off_4CF6DB8 == v56 )
          goto LABEL_57;
        v106 = *(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( (*(_BYTE *)(v26 + 9) & 0xC) != 8 )
        {
          if ( !off_4CF6DB8 )
            goto LABEL_57;
LABEL_193:
          BUG();
        }
        *(_BYTE *)(v26 + 8) |= 4u;
        v114 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
        v28 = v122;
        v115 = (void *)v114;
        v116 = v114 | *(_QWORD *)v26 & 7LL;
        *(_QWORD *)v26 = v116;
        if ( off_4CF6DB8 == v115 )
          goto LABEL_57;
        v106 = v116 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v106 )
        {
          if ( (*(_BYTE *)(v26 + 9) & 0xC) != 8 )
            goto LABEL_193;
          *(_BYTE *)(v26 + 8) |= 4u;
          v106 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
          v28 = v122;
          *(_QWORD *)v26 = v106 | *(_QWORD *)v26 & 7LL;
          if ( !v106 )
            goto LABEL_193;
        }
      }
      v107 = *(_DWORD *)(*(_QWORD *)(v106 + 24) + 172LL);
      if ( (v107 & 0x10) != 0 )
      {
        if ( v134 )
        {
LABEL_19:
          v30 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 12LL) & 1;
          if ( v30 )
          {
LABEL_20:
            v31 = v134;
            *a6 = 0;
LABEL_21:
            v32 = (void *)(*(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v32 )
              goto LABEL_94;
            v33 = 0;
            if ( (*(_BYTE *)(v26 + 9) & 0xC) != 8 )
              goto LABEL_23;
            *(_BYTE *)(v26 + 8) |= 4u;
            v128 = v28;
            v66 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
            v28 = v128;
            v33 = v66;
            v67 = v66 | *(_QWORD *)v26 & 7LL;
            *(_QWORD *)v26 = v67;
            if ( !v33 )
              goto LABEL_23;
            v32 = (void *)(v67 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v32 )
            {
LABEL_94:
              if ( v32 == off_4CF6DB8 )
              {
LABEL_98:
                v33 = 0;
                goto LABEL_23;
              }
              v33 = *(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL;
            }
            else
            {
              v33 = 0;
              if ( (*(_BYTE *)(v26 + 9) & 0xC) != 8 )
                goto LABEL_23;
              *(_BYTE *)(v26 + 8) |= 4u;
              v68 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
              v33 = 0;
              v69 = (void *)v68;
              v28 = v128;
              v70 = v68 | *(_QWORD *)v26 & 7LL;
              *(_QWORD *)v26 = v70;
              if ( off_4CF6DB8 == v69 )
                goto LABEL_23;
              v71 = v70 & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v71 )
              {
                if ( (*(_BYTE *)(v26 + 9) & 0xC) == 8 )
                {
                  *(_BYTE *)(v26 + 8) |= 4u;
                  v72 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24));
                  v28 = v128;
                  v33 = v72;
                  *(_QWORD *)v26 = v72 | *(_QWORD *)v26 & 7LL;
                  if ( v72 )
                    goto LABEL_91;
                }
LABEL_23:
                v127 = v33;
                result = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD, unsigned __int64))(*(_QWORD *)a1 + 80LL))(
                                     a1,
                                     v28,
                                     *(_QWORD *)(a5 + 16),
                                     v135);
                if ( !(_BYTE)result )
                  return result;
                v34 = v127;
                if ( !v30 )
                {
                  if ( v127 )
                  {
                    v34 = *(_QWORD *)(v127 + 8);
                    if ( v34 )
                      *(_BYTE *)(v34 + 9) |= 2u;
                  }
                  goto LABEL_75;
                }
                v35 = 0;
                if ( v26 )
                {
                  v36 = *(_DWORD *)(a1 + 72);
                  v35 = v26;
                  if ( v36 )
                  {
                    v37 = v36 - 1;
                    v38 = *(_QWORD *)(a1 + 56);
                    v39 = v37 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
                    v40 = (__int64 *)(v38 + 16LL * v39);
                    v41 = *v40;
                    if ( *v40 == v26 )
                    {
LABEL_28:
                      v35 = v40[1];
                      if ( !v35 )
                        v35 = v26;
                    }
                    else
                    {
                      v94 = 1;
                      while ( v41 != -8 )
                      {
                        v117 = v94 + 1;
                        v39 = v37 & (v94 + v39);
                        v40 = (__int64 *)(v38 + 16LL * v39);
                        v41 = *v40;
                        if ( *v40 == v26 )
                          goto LABEL_28;
                        v94 = v117;
                      }
                      v35 = v26;
                    }
                  }
                  if ( v118 )
                    sub_38E2760(v35);
                  else
                    *(_BYTE *)(v35 + 9) |= 2u;
                }
                v42 = *(_DWORD *)(a1 + 40);
                v139.m128i_i64[1] = v35;
                v140.m128i_i64[1] = v31;
                v139.m128i_i64[0] = v129;
                v141.m128i_i64[0] = v26;
                v140.m128i_i32[0] = v131;
                v141.m128i_i64[1] = v134;
                if ( v42 )
                {
                  v43 = *(_QWORD *)(a1 + 24);
                  v44 = (v42 - 1) & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
                  result = (_QWORD *)(v43 + 32LL * v44);
                  v45 = *result;
                  if ( v135 == *result )
                  {
LABEL_34:
                    v46 = (__m128i *)result[2];
                    if ( v46 != (__m128i *)result[3] )
                    {
                      if ( !v46 )
                      {
LABEL_37:
                        result[2] = v46 + 3;
                        return result;
                      }
LABEL_36:
                      *v46 = _mm_loadu_si128(&v139);
                      v46[1] = _mm_loadu_si128(&v140);
                      v46[2] = _mm_loadu_si128(&v141);
                      v46 = (__m128i *)result[2];
                      goto LABEL_37;
                    }
                    goto LABEL_190;
                  }
                  v75 = 1;
                  v76 = 0;
                  while ( v45 != -8 )
                  {
                    if ( v45 == -16 && !v76 )
                      v76 = result;
                    v44 = (v42 - 1) & (v75 + v44);
                    result = (_QWORD *)(v43 + 32LL * v44);
                    v45 = *result;
                    if ( v135 == *result )
                      goto LABEL_34;
                    ++v75;
                  }
                  if ( !v76 )
                    v76 = result;
                  v77 = *(_DWORD *)(a1 + 32);
                  ++*(_QWORD *)(a1 + 16);
                  v78 = v77 + 1;
                  if ( 4 * (v77 + 1) < 3 * v42 )
                  {
                    if ( v42 - *(_DWORD *)(a1 + 36) - v78 > v42 >> 3 )
                      goto LABEL_105;
                    sub_3930E00(a1 + 16, v42);
                    v87 = *(_DWORD *)(a1 + 40);
                    if ( v87 )
                    {
                      v88 = v87 - 1;
                      v89 = *(_QWORD *)(a1 + 24);
                      v90 = 1;
                      v91 = v88 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
                      v92 = 0;
                      v78 = *(_DWORD *)(a1 + 32) + 1;
                      v76 = (_QWORD *)(v89 + 32LL * v91);
                      v93 = *v76;
                      if ( v135 != *v76 )
                      {
                        while ( v93 != -8 )
                        {
                          if ( v93 == -16 && !v92 )
                            v92 = v76;
                          v91 = v88 & (v90 + v91);
                          v76 = (_QWORD *)(v89 + 32LL * v91);
                          v93 = *v76;
                          if ( v135 == *v76 )
                            goto LABEL_105;
                          ++v90;
                        }
                        goto LABEL_120;
                      }
LABEL_105:
                      *(_DWORD *)(a1 + 32) = v78;
                      if ( *v76 != -8 )
                        --*(_DWORD *)(a1 + 36);
                      v79 = v76 + 1;
                      *v79 = 0;
                      v46 = 0;
                      v79[1] = 0;
                      *(v79 - 1) = v135;
                      v79[2] = 0;
                      return (_QWORD *)sub_392F310(v79, v46, &v139);
                    }
                    goto LABEL_192;
                  }
                }
                else
                {
                  ++*(_QWORD *)(a1 + 16);
                }
                sub_3930E00(a1 + 16, 2 * v42);
                v80 = *(_DWORD *)(a1 + 40);
                if ( v80 )
                {
                  v81 = v80 - 1;
                  v82 = *(_QWORD *)(a1 + 24);
                  v83 = (v80 - 1) & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
                  v78 = *(_DWORD *)(a1 + 32) + 1;
                  v76 = (_QWORD *)(v82 + 32LL * v83);
                  v84 = *v76;
                  if ( v135 == *v76 )
                    goto LABEL_105;
                  v85 = 1;
                  v86 = 0;
                  while ( v84 != -8 )
                  {
                    if ( !v86 && v84 == -16 )
                      v86 = v76;
                    v83 = v81 & (v85 + v83);
                    v76 = (_QWORD *)(v82 + 32LL * v83);
                    v84 = *v76;
                    if ( v135 == *v76 )
                      goto LABEL_105;
                    ++v85;
                  }
                  goto LABEL_114;
                }
LABEL_192:
                ++*(_DWORD *)(a1 + 32);
                BUG();
              }
              v33 = v71;
            }
LABEL_91:
            v33 = *(_QWORD *)(v33 + 24);
            goto LABEL_23;
          }
          v74 = v134;
LABEL_93:
          v30 = 1;
          v31 = 0;
          *a6 = v74;
          goto LABEL_21;
        }
        if ( (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 12LL) & 1) == 0 )
        {
          v74 = 0;
          goto LABEL_93;
        }
      }
      if ( (v107 & 0x400) != 0 )
        goto LABEL_19;
LABEL_57:
      v123 = v28;
      v57 = sub_390AF00(a2, v26);
      v28 = v123;
      if ( v57 )
        goto LABEL_19;
      v58 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(
              *(_QWORD *)(a1 + 8),
              v26,
              v131);
      v28 = v123;
      if ( v58 )
        goto LABEL_19;
LABEL_64:
      if ( (*(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL) != 0
        || (v31 = v134, (*(_BYTE *)(v26 + 9) & 0xC) == 8)
        && (*(_BYTE *)(v26 + 8) |= 4u,
            v124 = v28,
            v59 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v26 + 24)),
            v28 = v124,
            *(_QWORD *)v26 = v59 | *(_QWORD *)v26 & 7LL,
            v59) )
      {
        v125 = v28;
        v60 = sub_38D0440(a3, v26);
        v28 = v125;
        v31 = v134 + v60;
      }
      v30 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 12LL) & 1;
      if ( v30 )
      {
        v30 = 0;
        *a6 = 0;
      }
      else
      {
        *a6 = v31;
        v31 = 0;
      }
      goto LABEL_21;
    }
    v30 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 12LL) & 1;
    if ( v30 )
      goto LABEL_20;
    v31 = 0;
    *a6 = v134;
LABEL_62:
    v30 = 1;
    goto LABEL_21;
  }
  if ( v26 )
    goto LABEL_64;
LABEL_44:
  v30 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 12LL) & 1;
  if ( !v30 )
  {
    v31 = 0;
    v26 = 0;
    *a6 = v134;
    goto LABEL_98;
  }
  *a6 = 0;
  result = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD, unsigned __int64, _QWORD))(*(_QWORD *)a1 + 80LL))(
                       a1,
                       v28,
                       *(_QWORD *)(a5 + 16),
                       v135,
                       0);
  if ( (_BYTE)result )
  {
    v31 = v134;
    v26 = 0;
    v34 = 0;
LABEL_75:
    v61 = *(_DWORD *)(a1 + 40);
    v139.m128i_i64[1] = v34;
    v140.m128i_i64[1] = v31;
    v139.m128i_i64[0] = v129;
    v141.m128i_i64[0] = v26;
    v140.m128i_i32[0] = v131;
    v141.m128i_i64[1] = v134;
    if ( v61 )
    {
      v62 = *(_QWORD *)(a1 + 24);
      v63 = (v61 - 1) & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
      result = (_QWORD *)(v62 + 32LL * v63);
      v64 = *result;
      if ( v135 == *result )
      {
LABEL_77:
        v46 = (__m128i *)result[2];
        if ( v46 != (__m128i *)result[3] )
        {
          if ( !v46 )
            goto LABEL_37;
          goto LABEL_36;
        }
LABEL_190:
        v79 = result + 1;
        return (_QWORD *)sub_392F310(v79, v46, &v139);
      }
      v95 = 1;
      v76 = 0;
      while ( v64 != -8 )
      {
        if ( v64 == -16 && !v76 )
          v76 = result;
        v63 = (v61 - 1) & (v95 + v63);
        result = (_QWORD *)(v62 + 32LL * v63);
        v64 = *result;
        if ( v135 == *result )
          goto LABEL_77;
        ++v95;
      }
      if ( !v76 )
        v76 = result;
      v96 = *(_DWORD *)(a1 + 32);
      ++*(_QWORD *)(a1 + 16);
      v78 = v96 + 1;
      if ( 4 * (v96 + 1) < 3 * v61 )
      {
        if ( v61 - *(_DWORD *)(a1 + 36) - v78 > v61 >> 3 )
          goto LABEL_105;
        sub_3930E00(a1 + 16, v61);
        v97 = *(_DWORD *)(a1 + 40);
        if ( v97 )
        {
          v98 = v97 - 1;
          v99 = *(_QWORD *)(a1 + 24);
          v100 = 1;
          v101 = v98 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
          v92 = 0;
          v78 = *(_DWORD *)(a1 + 32) + 1;
          v76 = (_QWORD *)(v99 + 32LL * v101);
          v102 = *v76;
          if ( *v76 != v135 )
          {
            while ( v102 != -8 )
            {
              if ( !v92 && v102 == -16 )
                v92 = v76;
              v101 = v98 & (v100 + v101);
              v76 = (_QWORD *)(v99 + 32LL * v101);
              v102 = *v76;
              if ( v135 == *v76 )
                goto LABEL_105;
              ++v100;
            }
LABEL_120:
            if ( v92 )
              v76 = v92;
            goto LABEL_105;
          }
          goto LABEL_105;
        }
LABEL_194:
        ++*(_DWORD *)(a1 + 32);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 16);
    }
    sub_3930E00(a1 + 16, 2 * v61);
    v108 = *(_DWORD *)(a1 + 40);
    if ( !v108 )
      goto LABEL_194;
    v109 = v108 - 1;
    v110 = *(_QWORD *)(a1 + 24);
    v111 = (v108 - 1) & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
    v78 = *(_DWORD *)(a1 + 32) + 1;
    v76 = (_QWORD *)(v110 + 32LL * v111);
    v112 = *v76;
    if ( v135 == *v76 )
      goto LABEL_105;
    v113 = 1;
    v86 = 0;
    while ( v112 != -8 )
    {
      if ( !v86 && v112 == -16 )
        v86 = v76;
      v111 = v109 & (v113 + v111);
      v76 = (_QWORD *)(v110 + 32LL * v111);
      v112 = *v76;
      if ( v135 == *v76 )
        goto LABEL_105;
      ++v113;
    }
LABEL_114:
    if ( v86 )
      v76 = v86;
    goto LABEL_105;
  }
  return result;
}
