// Function: sub_1CF4130
// Address: 0x1cf4130
//
__int64 __fastcall sub_1CF4130(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __m128i si128,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rdi
  int v16; // r12d
  __m128i *v17; // r9
  unsigned __int32 v18; // eax
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 *v22; // rax
  int v23; // r14d
  __int64 *v24; // rcx
  __int64 *v25; // rdx
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 result; // rax
  __int64 v29; // r12
  __int64 v30; // r11
  __int64 v31; // r8
  unsigned int v32; // edi
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r13
  int v36; // r15d
  unsigned int v37; // esi
  int v38; // eax
  int v39; // edi
  __int64 v40; // rsi
  unsigned int v41; // eax
  int v42; // ecx
  __int64 *v43; // rdx
  __int64 v44; // r8
  __int64 v45; // r12
  __int64 i; // r13
  _QWORD *v47; // rsi
  _QWORD *j; // rbx
  _QWORD *v49; // rdi
  int v50; // r10d
  int v51; // eax
  int v52; // eax
  int v53; // eax
  __int64 v54; // rdi
  __int64 *v55; // r8
  unsigned int v56; // r14d
  int v57; // r9d
  __int64 v58; // rsi
  __int64 v59; // r13
  __int64 v60; // r14
  double v61; // xmm4_8
  double v62; // xmm5_8
  _QWORD *v63; // rax
  __int64 v64; // rcx
  unsigned __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // r8
  unsigned int v68; // edi
  _QWORD *v69; // rax
  __int64 v70; // rcx
  int v71; // r15d
  unsigned int v72; // esi
  __int64 v73; // r13
  int v74; // eax
  int v75; // edi
  __int64 v76; // rsi
  unsigned int v77; // eax
  int v78; // ecx
  _QWORD *v79; // rdx
  __int64 v80; // r8
  int v81; // r10d
  int v82; // eax
  int v83; // eax
  int v84; // eax
  __int64 v85; // rdi
  _QWORD *v86; // r8
  unsigned int v87; // r14d
  int v88; // r9d
  __int64 v89; // rsi
  __int64 v90; // rax
  __m128 *v91; // rdx
  __int64 v92; // r12
  _BYTE *v93; // rax
  int v94; // r10d
  _QWORD *v95; // r9
  int v96; // r10d
  __int64 *v97; // r9
  __int64 v98; // [rsp+8h] [rbp-2B8h]
  __int64 v99; // [rsp+10h] [rbp-2B0h]
  __int64 v100; // [rsp+20h] [rbp-2A0h]
  __int64 v101; // [rsp+28h] [rbp-298h]
  __int64 v102; // [rsp+30h] [rbp-290h]
  __int64 v103; // [rsp+38h] [rbp-288h]
  __int64 v104; // [rsp+38h] [rbp-288h]
  __int64 v105; // [rsp+38h] [rbp-288h]
  __int64 v106; // [rsp+38h] [rbp-288h]
  __int64 v107; // [rsp+38h] [rbp-288h]
  __m128i *v108; // [rsp+38h] [rbp-288h]
  __m128i v109[2]; // [rsp+40h] [rbp-280h] BYREF
  __m128i v110; // [rsp+60h] [rbp-260h] BYREF
  __int64 v111; // [rsp+70h] [rbp-250h]
  __m128i v112; // [rsp+80h] [rbp-240h] BYREF
  __int64 v113; // [rsp+90h] [rbp-230h] BYREF
  __int64 v114; // [rsp+98h] [rbp-228h]

  if ( byte_4FC07C0 )
  {
    v90 = sub_16BA580((__int64)a1, a2, a3);
    v91 = *(__m128 **)(v90 + 24);
    v92 = v90;
    if ( *(_QWORD *)(v90 + 16) - (_QWORD)v91 <= 0x16u )
    {
      v92 = sub_16E7EE0(v90, "IR Module before CSSA:\n", 0x17u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42E2030);
      v91[1].m128_i32[0] = 1397965600;
      v91[1].m128_i16[2] = 14913;
      v91[1].m128_i8[6] = 10;
      *v91 = (__m128)si128;
      *(_QWORD *)(v90 + 24) += 23LL;
    }
    sub_155BB10(*(_QWORD *)(*a1 + 40LL), v92, 0, 0, 0, si128);
    v93 = *(_BYTE **)(v92 + 24);
    if ( *(_BYTE **)(v92 + 16) == v93 )
    {
      sub_16E7EE0(v92, "\n", 1u);
    }
    else
    {
      *v93 = 10;
      ++*(_QWORD *)(v92 + 24);
    }
  }
  v12 = a1[1];
  if ( *(_BYTE *)(v12 + 72) )
  {
    *(_DWORD *)(v12 + 76) = 0;
  }
  else
  {
    v112.m128i_i32[3] = 32;
    v112.m128i_i64[0] = (__int64)&v113;
    v13 = *(_QWORD *)(v12 + 56);
    if ( v13 )
    {
      v14 = *(_QWORD *)(v13 + 24);
      v15 = &v113;
      v113 = *(_QWORD *)(v12 + 56);
      v16 = 1;
      v112.m128i_i32[2] = 1;
      v17 = &v112;
      v114 = v14;
      *(_DWORD *)(v13 + 48) = 0;
      v18 = 1;
      do
      {
        while ( 1 )
        {
          v23 = v16++;
          v24 = &v15[2 * v18 - 2];
          v25 = (__int64 *)v24[1];
          if ( v25 != *(__int64 **)(*v24 + 32) )
            break;
          --v18;
          *(_DWORD *)(*v24 + 52) = v23;
          v112.m128i_i32[2] = v18;
          if ( !v18 )
            goto LABEL_10;
        }
        v19 = *v25;
        v24[1] = (__int64)(v25 + 1);
        v20 = v112.m128i_u32[2];
        v21 = *(_QWORD *)(v19 + 24);
        if ( v112.m128i_i32[2] >= (unsigned __int32)v112.m128i_i32[3] )
        {
          v101 = v12;
          v108 = v17;
          sub_16CD150((__int64)v17, &v113, 0, 16, v12, (int)v17);
          v15 = (__int64 *)v112.m128i_i64[0];
          v20 = v112.m128i_u32[2];
          v12 = v101;
          v17 = v108;
        }
        v22 = &v15[2 * v20];
        *v22 = v19;
        v22[1] = v21;
        v18 = ++v112.m128i_i32[2];
        *(_DWORD *)(v19 + 48) = v23;
        v15 = (__int64 *)v112.m128i_i64[0];
      }
      while ( v18 );
LABEL_10:
      *(_DWORD *)(v12 + 76) = 0;
      *(_BYTE *)(v12 + 72) = 1;
      if ( v15 != &v113 )
        _libc_free((unsigned __int64)v15);
    }
  }
  sub_1CF19E0(*a1, (__m128)si128, a5, a6, a7, a8, a9, a10, a11);
  v99 = (__int64)(a1 + 11);
  result = *a1 + 72LL;
  v98 = result;
  v102 = *(_QWORD *)(*a1 + 80LL);
  if ( v102 != result )
  {
    while ( 1 )
    {
      if ( !v102 )
        BUG();
      v29 = *(_QWORD *)(v102 + 24);
      v30 = v102 + 16;
      if ( v29 != v102 + 16 )
        break;
LABEL_28:
      result = *(_QWORD *)(v102 + 8);
      v102 = result;
      if ( v98 == result )
      {
        v45 = *(_QWORD *)(*a1 + 80LL);
        for ( i = *a1 + 72LL; i != v45; v45 = *(_QWORD *)(v45 + 8) )
        {
          v47 = (_QWORD *)(v45 - 24);
          if ( !v45 )
            v47 = 0;
          result = sub_1CF34C0((__int64)a1, v47, (__m128)si128, a5, a6, a7, v26, v27, a10, a11);
        }
        goto LABEL_33;
      }
    }
    while ( 1 )
    {
      if ( !v29 )
LABEL_135:
        BUG();
      v35 = v29 - 24;
      if ( *(_BYTE *)(v29 - 8) != 77 )
      {
        if ( *(_QWORD *)(v102 + 24) == v29 )
          goto LABEL_65;
        goto LABEL_52;
      }
      v36 = *((_DWORD *)a1 + 30);
      v37 = *((_DWORD *)a1 + 28);
      *((_DWORD *)a1 + 30) = v36 + 1;
      if ( !v37 )
        break;
      v31 = a1[12];
      v32 = (v37 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v33 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v33;
      if ( v35 == *v33 )
      {
LABEL_17:
        v29 = *(_QWORD *)(v29 + 8);
        if ( v29 == v30 )
          goto LABEL_27;
      }
      else
      {
        v50 = 1;
        v43 = 0;
        while ( v34 != -8 )
        {
          if ( v43 || v34 != -16 )
            v33 = v43;
          v32 = (v37 - 1) & (v50 + v32);
          v34 = *(_QWORD *)(v31 + 16LL * v32);
          if ( v35 == v34 )
            goto LABEL_17;
          ++v50;
          v43 = v33;
          v33 = (__int64 *)(v31 + 16LL * v32);
        }
        if ( !v43 )
          v43 = v33;
        v51 = *((_DWORD *)a1 + 26);
        ++a1[11];
        v42 = v51 + 1;
        if ( 4 * (v51 + 1) < 3 * v37 )
        {
          if ( v37 - *((_DWORD *)a1 + 27) - v42 <= v37 >> 3 )
          {
            v104 = v30;
            sub_1541C50(v99, v37);
            v52 = *((_DWORD *)a1 + 28);
            if ( !v52 )
            {
LABEL_134:
              ++*((_DWORD *)a1 + 26);
              BUG();
            }
            v53 = v52 - 1;
            v54 = a1[12];
            v55 = 0;
            v56 = v53 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
            v30 = v104;
            v57 = 1;
            v42 = *((_DWORD *)a1 + 26) + 1;
            v43 = (__int64 *)(v54 + 16LL * v56);
            v58 = *v43;
            if ( v35 != *v43 )
            {
              while ( v58 != -8 )
              {
                if ( !v55 && v58 == -16 )
                  v55 = v43;
                v56 = v53 & (v57 + v56);
                v43 = (__int64 *)(v54 + 16LL * v56);
                v58 = *v43;
                if ( v35 == *v43 )
                  goto LABEL_24;
                ++v57;
              }
              if ( v55 )
                v43 = v55;
            }
          }
          goto LABEL_24;
        }
LABEL_22:
        v103 = v30;
        sub_1541C50(v99, 2 * v37);
        v38 = *((_DWORD *)a1 + 28);
        if ( !v38 )
          goto LABEL_134;
        v39 = v38 - 1;
        v40 = a1[12];
        v30 = v103;
        v41 = (v38 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v42 = *((_DWORD *)a1 + 26) + 1;
        v43 = (__int64 *)(v40 + 16LL * v41);
        v44 = *v43;
        if ( v35 != *v43 )
        {
          v96 = 1;
          v97 = 0;
          while ( v44 != -8 )
          {
            if ( v44 == -16 && !v97 )
              v97 = v43;
            v41 = v39 & (v96 + v41);
            v43 = (__int64 *)(v40 + 16LL * v41);
            v44 = *v43;
            if ( v35 == *v43 )
              goto LABEL_24;
            ++v96;
          }
          if ( v97 )
            v43 = v97;
        }
LABEL_24:
        *((_DWORD *)a1 + 26) = v42;
        if ( *v43 != -8 )
          --*((_DWORD *)a1 + 27);
        *v43 = v35;
        *((_DWORD *)v43 + 2) = v36;
        v29 = *(_QWORD *)(v29 + 8);
        if ( v29 == v30 )
        {
LABEL_27:
          v35 = v102 - 8;
          if ( *(_QWORD *)(v102 + 24) == v29 )
            goto LABEL_28;
LABEL_52:
          v105 = v30;
          v112.m128i_i8[8] = 1;
          v112.m128i_i64[0] = v35;
          sub_1CF2B90((__int64)v109, (__int64)a1, v102 - 24, (__int64)&v112);
          v30 = v105;
          if ( *(_QWORD *)(v102 + 24) == v105 )
            goto LABEL_65;
          v100 = v35;
          v59 = *(_QWORD *)(v102 + 24);
          do
          {
            if ( !v59 )
              goto LABEL_135;
            if ( *(_BYTE *)(v59 - 8) != 77 )
              break;
            v112.m128i_i8[8] = 1;
            v112.m128i_i64[0] = v100;
            sub_1CF2F90(&v110, (__int64)a1, v59 - 24, v109, (__int64)&v112);
            v60 = v110.m128i_i64[0];
            a5 = (__m128)_mm_load_si128(&v110);
            LOBYTE(v114) = 1;
            v112 = (__m128i)a5;
            v113 = v111;
            sub_1CF27D0((__int64)a1, v110.m128i_i64[0], &v112);
            sub_164D160(v59 - 24, v60, (__m128)si128, *(double *)a5.m128_u64, a6, a7, v61, v62, a10, a11);
            v63 = (_QWORD *)(v60 + 24 * (1LL - (*(_DWORD *)(v60 + 20) & 0xFFFFFFF)));
            if ( *v63 )
            {
              v64 = v63[1];
              v65 = v63[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v65 = v64;
              if ( v64 )
                *(_QWORD *)(v64 + 16) = *(_QWORD *)(v64 + 16) & 3LL | v65;
            }
            *v63 = v59 - 24;
            v66 = *(_QWORD *)(v59 - 16);
            v63[1] = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = (unsigned __int64)(v63 + 1) | *(_QWORD *)(v66 + 16) & 3LL;
            v63[2] = (v59 - 16) | v63[2] & 3LL;
            *(_QWORD *)(v59 - 16) = v63;
            v59 = *(_QWORD *)(v59 + 8);
          }
          while ( v59 != v105 );
          v30 = v105;
LABEL_65:
          while ( 2 )
          {
            if ( v29 == v30 )
              goto LABEL_28;
            v71 = *((_DWORD *)a1 + 30);
            v72 = *((_DWORD *)a1 + 28);
            v73 = v29 - 24;
            if ( !v29 )
              v73 = 0;
            *((_DWORD *)a1 + 30) = v71 + 1;
            if ( !v72 )
            {
              ++a1[11];
              goto LABEL_70;
            }
            v67 = a1[12];
            v68 = (v72 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
            v69 = (_QWORD *)(v67 + 16LL * v68);
            v70 = *v69;
            if ( v73 != *v69 )
            {
              v81 = 1;
              v79 = 0;
              while ( v70 != -8 )
              {
                if ( v79 || v70 != -16 )
                  v69 = v79;
                v68 = (v72 - 1) & (v81 + v68);
                v70 = *(_QWORD *)(v67 + 16LL * v68);
                if ( v73 == v70 )
                  goto LABEL_64;
                ++v81;
                v79 = v69;
                v69 = (_QWORD *)(v67 + 16LL * v68);
              }
              if ( !v79 )
                v79 = v69;
              v82 = *((_DWORD *)a1 + 26);
              ++a1[11];
              v78 = v82 + 1;
              if ( 4 * (v82 + 1) >= 3 * v72 )
              {
LABEL_70:
                v106 = v30;
                sub_1541C50(v99, 2 * v72);
                v74 = *((_DWORD *)a1 + 28);
                if ( !v74 )
                  goto LABEL_134;
                v75 = v74 - 1;
                v76 = a1[12];
                v30 = v106;
                v77 = (v74 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
                v78 = *((_DWORD *)a1 + 26) + 1;
                v79 = (_QWORD *)(v76 + 16LL * v77);
                v80 = *v79;
                if ( v73 != *v79 )
                {
                  v94 = 1;
                  v95 = 0;
                  while ( v80 != -8 )
                  {
                    if ( !v95 && v80 == -16 )
                      v95 = v79;
                    v77 = v75 & (v94 + v77);
                    v79 = (_QWORD *)(v76 + 16LL * v77);
                    v80 = *v79;
                    if ( v73 == *v79 )
                      goto LABEL_72;
                    ++v94;
                  }
                  if ( v95 )
                    v79 = v95;
                }
              }
              else if ( v72 - *((_DWORD *)a1 + 27) - v78 <= v72 >> 3 )
              {
                v107 = v30;
                sub_1541C50(v99, v72);
                v83 = *((_DWORD *)a1 + 28);
                if ( !v83 )
                  goto LABEL_134;
                v84 = v83 - 1;
                v85 = a1[12];
                v86 = 0;
                v87 = v84 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
                v30 = v107;
                v88 = 1;
                v78 = *((_DWORD *)a1 + 26) + 1;
                v79 = (_QWORD *)(v85 + 16LL * v87);
                v89 = *v79;
                if ( v73 != *v79 )
                {
                  while ( v89 != -8 )
                  {
                    if ( !v86 && v89 == -16 )
                      v86 = v79;
                    v87 = v84 & (v88 + v87);
                    v79 = (_QWORD *)(v85 + 16LL * v87);
                    v89 = *v79;
                    if ( v73 == *v79 )
                      goto LABEL_72;
                    ++v88;
                  }
                  if ( v86 )
                    v79 = v86;
                }
              }
LABEL_72:
              *((_DWORD *)a1 + 26) = v78;
              if ( *v79 != -8 )
                --*((_DWORD *)a1 + 27);
              *v79 = v73;
              *((_DWORD *)v79 + 2) = v71;
            }
LABEL_64:
            v29 = *(_QWORD *)(v29 + 8);
            continue;
          }
        }
      }
    }
    ++a1[11];
    goto LABEL_22;
  }
LABEL_33:
  for ( j = (_QWORD *)a1[5]; j; j = (_QWORD *)*j )
  {
    while ( 1 )
    {
      v49 = (_QWORD *)j[1];
      if ( !v49[1] )
        break;
      j = (_QWORD *)*j;
      if ( !j )
        return result;
    }
    result = sub_15F20C0(v49);
  }
  return result;
}
