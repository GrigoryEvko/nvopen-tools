// Function: sub_28F0920
// Address: 0x28f0920
//
__int64 __fastcall sub_28F0920(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  unsigned __int8 *v4; // r13
  __int64 v5; // r9
  int v6; // edx
  __int64 v7; // rax
  unsigned __int8 *v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r8
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // r12
  int v20; // edx
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned int v23; // ebx
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // r15
  __int64 v26; // r8
  __m128i *v27; // r14
  __m128i *v28; // rax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  char v31; // dl
  unsigned int v32; // esi
  __int64 v33; // rdx
  __int64 v34; // rdi
  int v35; // r11d
  __int64 v36; // r10
  unsigned int j; // eax
  __int64 v38; // r12
  __int64 v39; // r8
  unsigned int v40; // eax
  unsigned __int64 v41; // rdx
  __m128i v42; // xmm0
  char v43; // r15
  __m128i *v44; // rbx
  _QWORD *v45; // rdx
  _QWORD *v46; // rsi
  __int64 v47; // r14
  int v48; // edi
  int v49; // edi
  __int64 v50; // rsi
  int v51; // r11d
  unsigned int k; // eax
  __int64 v53; // r8
  unsigned int v54; // eax
  __int64 v55; // rax
  __int32 v56; // eax
  __int32 v57; // ecx
  __int64 v58; // rax
  unsigned __int64 v59; // rax
  bool v60; // zf
  unsigned __int64 v61; // rax
  __int64 v62; // r14
  int v63; // edi
  int v64; // edi
  __int64 v65; // rsi
  int v66; // r11d
  unsigned int v67; // eax
  __int64 v68; // r8
  unsigned int v69; // eax
  unsigned int v70; // eax
  __int64 v72; // [rsp+8h] [rbp-418h]
  __int64 v73; // [rsp+18h] [rbp-408h]
  __int64 v74; // [rsp+28h] [rbp-3F8h]
  __m128i v75; // [rsp+40h] [rbp-3E0h] BYREF
  unsigned int v76; // [rsp+54h] [rbp-3CCh]
  __int64 v77; // [rsp+58h] [rbp-3C8h]
  unsigned __int64 v78; // [rsp+60h] [rbp-3C0h]
  char v79; // [rsp+6Bh] [rbp-3B5h]
  unsigned int v80; // [rsp+6Ch] [rbp-3B4h]
  __int64 v81; // [rsp+70h] [rbp-3B0h]
  __int64 v82; // [rsp+78h] [rbp-3A8h]
  _QWORD v83[2]; // [rsp+80h] [rbp-3A0h] BYREF
  unsigned __int64 v84; // [rsp+90h] [rbp-390h]
  _QWORD v85[2]; // [rsp+98h] [rbp-388h] BYREF
  unsigned __int64 v86; // [rsp+A8h] [rbp-378h]
  int v87; // [rsp+B0h] [rbp-370h]
  __m128i v88; // [rsp+C0h] [rbp-360h] BYREF
  unsigned __int64 v89[2]; // [rsp+D0h] [rbp-350h] BYREF
  unsigned __int64 v90; // [rsp+E0h] [rbp-340h]
  unsigned __int64 v91[2]; // [rsp+E8h] [rbp-338h] BYREF
  unsigned __int64 v92; // [rsp+F8h] [rbp-328h]
  int v93; // [rsp+100h] [rbp-320h]
  _QWORD *i; // [rsp+110h] [rbp-310h] BYREF
  __int64 v95; // [rsp+118h] [rbp-308h]
  _QWORD v96[8]; // [rsp+120h] [rbp-300h] BYREF
  _BYTE *v97; // [rsp+160h] [rbp-2C0h] BYREF
  __int64 v98; // [rsp+168h] [rbp-2B8h]
  _BYTE v99[64]; // [rsp+170h] [rbp-2B0h] BYREF
  __m128i *v100; // [rsp+1B0h] [rbp-270h] BYREF
  __int64 v101; // [rsp+1B8h] [rbp-268h]
  _BYTE v102[512]; // [rsp+1C0h] [rbp-260h] BYREF
  __int64 v103; // [rsp+3C0h] [rbp-60h] BYREF
  __int64 v104; // [rsp+3C8h] [rbp-58h] BYREF
  unsigned __int64 v105; // [rsp+3D0h] [rbp-50h]
  __int64 *v106; // [rsp+3D8h] [rbp-48h]
  __int64 *v107; // [rsp+3E0h] [rbp-40h]
  __int64 v108; // [rsp+3E8h] [rbp-38h]

  result = *a2 + 8LL * *((unsigned int *)a2 + 2);
  v72 = *a2;
  v73 = result;
  if ( result != *a2 )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(v73 - 8);
      v74 = v3 + 48;
      v77 = *(_QWORD *)(v3 + 56);
      if ( v77 != v3 + 48 )
        break;
LABEL_54:
      v73 -= 8;
      result = v73;
      if ( v72 == v73 )
        return result;
    }
    while ( 1 )
    {
      v4 = (unsigned __int8 *)(v77 - 24);
      if ( !v77 )
        v4 = 0;
      v79 = sub_B46CC0(v4);
      if ( !v79 )
        goto LABEL_53;
      v6 = *v4;
      if ( (unsigned int)(v6 - 42) > 0x11 )
        goto LABEL_53;
      v7 = *((_QWORD *)v4 + 2);
      if ( v7 )
      {
        if ( !*(_QWORD *)(v7 + 8) && (_BYTE)v6 == **(_BYTE **)(v7 + 24) )
          goto LABEL_53;
      }
      v8 = (v4[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v4 - 1) : &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
      v11 = *(_QWORD *)v8;
      v9 = *((_QWORD *)v8 + 4);
      v10 = v96;
      v96[0] = v11;
      LODWORD(v11) = 2;
      v96[1] = v9;
      v95 = 0x800000002LL;
      v97 = v99;
      v98 = 0x800000000LL;
      v12 = 0;
      for ( i = v96; ; v10 = i )
      {
        v14 = (unsigned int)v11;
        v11 = (unsigned int)(v11 - 1);
        v15 = v10[v14 - 1];
        LODWORD(v95) = v11;
        if ( *(_BYTE *)v15 > 0x1Cu && *(_BYTE *)v15 == *v4 )
        {
          v16 = *(_QWORD *)(v15 + 16);
          if ( v16 )
          {
            if ( !*(_QWORD *)(v16 + 8) )
              break;
          }
        }
        v13 = v12 + 1;
        if ( v13 > HIDWORD(v98) )
          sub_C8D5F0((__int64)&v97, v99, v13, 8u, v14, v5);
        *(_QWORD *)&v97[8 * (unsigned int)v98] = v15;
        LODWORD(v11) = v95;
        v12 = v98 + 1;
        LODWORD(v98) = v98 + 1;
LABEL_14:
        if ( !(_DWORD)v11 )
          goto LABEL_30;
LABEL_15:
        if ( v12 > 0xA )
          goto LABEL_49;
      }
      if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
        break;
      v17 = (__int64 *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
      v18 = *v17;
      if ( v15 == *v17 )
        goto LABEL_26;
      if ( v14 > HIDWORD(v95) )
        goto LABEL_144;
LABEL_24:
      v10[v11] = v18;
      v11 = (unsigned int)(v95 + 1);
      LODWORD(v95) = v95 + 1;
      if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
        v17 = *(__int64 **)(v15 - 8);
      else
        v17 = (__int64 *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
LABEL_26:
      v19 = v17[4];
      if ( v15 == v19 )
      {
        v12 = v98;
        goto LABEL_14;
      }
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v95) )
      {
        sub_C8D5F0((__int64)&i, v96, v11 + 1, 8u, v11 + 1, v5);
        v11 = (unsigned int)v95;
      }
      i[v11] = v19;
      LODWORD(v11) = v95 + 1;
      v12 = v98;
      LODWORD(v95) = v11;
      if ( (_DWORD)v11 )
        goto LABEL_15;
LABEL_30:
      v80 = v11;
      if ( v12 <= 0xA )
      {
        v20 = *v4;
        LODWORD(v104) = 0;
        v100 = (__m128i *)v102;
        v21 = (unsigned int)(v20 - 42);
        v101 = 0x2000000000LL;
        v105 = 0;
        v106 = &v104;
        v107 = &v104;
        v108 = 0;
        if ( v12 != 1 )
        {
          v78 = 0;
          v81 = a1 + 32 * v21 + 176;
          do
          {
            v22 = ++v80;
            v82 = 8 * v78;
            v23 = v80;
            v78 = v80;
            if ( v80 < v12 )
            {
              do
              {
                v24 = *(_QWORD *)&v97[8 * v22];
                v25 = *(_QWORD *)&v97[v82];
                if ( v24 < v25 )
                {
                  v25 = *(_QWORD *)&v97[8 * v22];
                  v24 = *(_QWORD *)&v97[v82];
                }
                v88.m128i_i64[0] = v25;
                v88.m128i_i64[1] = v24;
                if ( v108 )
                {
                  sub_27E1010((__int64)&v103, &v88);
                  if ( !v31 )
                    goto LABEL_43;
                  goto LABEL_57;
                }
                v26 = (unsigned int)v101;
                v27 = &v100[v26];
                if ( v100 == &v100[v26] )
                {
                  if ( (unsigned int)v101 <= 0x1FuLL )
                    goto LABEL_79;
                }
                else
                {
                  v28 = v100;
                  while ( v25 != v28->m128i_i64[0] || v24 != v28->m128i_i64[1] )
                  {
                    if ( v27 == ++v28 )
                      goto LABEL_78;
                  }
                  if ( v27 != v28 )
                    goto LABEL_43;
LABEL_78:
                  if ( (unsigned int)v101 <= 0x1FuLL )
                  {
LABEL_79:
                    v41 = (unsigned int)v101 + 1LL;
                    v42 = _mm_load_si128(&v88);
                    if ( v41 > HIDWORD(v101) )
                    {
                      v75 = v42;
                      sub_C8D5F0((__int64)&v100, v102, v41, 0x10u, v26 * 16, v5);
                      v42 = _mm_load_si128(&v75);
                      v27 = &v100[(unsigned int)v101];
                    }
                    *v27 = v42;
                    LODWORD(v101) = v101 + 1;
                    goto LABEL_57;
                  }
                  v75.m128i_i64[0] = v25;
                  v76 = v23;
                  v44 = v100;
                  do
                  {
                    v46 = sub_27E25D0(&v103, &v104, (unsigned __int64 *)v44);
                    if ( v45 )
                      sub_28E9980((__int64)&v103, (__int64)v46, v45, v44);
                    ++v44;
                  }
                  while ( v27 != v44 );
                  v25 = v75.m128i_i64[0];
                  v23 = v76;
                }
                LODWORD(v101) = 0;
                sub_27E1010((__int64)&v103, &v88);
LABEL_57:
                v84 = v25;
                v83[0] = 4;
                v83[1] = 0;
                if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
                  sub_BD73F0((__int64)v83);
                v86 = v24;
                v85[0] = 4;
                v85[1] = 0;
                if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
                  sub_BD73F0((__int64)v85);
                v88.m128i_i64[0] = v25;
                v87 = 1;
                v88.m128i_i64[1] = v24;
                v89[0] = 4;
                v90 = v84;
                v89[1] = 0;
                if ( v84 != -4096 && v84 != 0 && v84 != -8192 )
                  sub_BD6050(v89, v83[0] & 0xFFFFFFFFFFFFFFF8LL);
                v91[0] = 4;
                v91[1] = 0;
                v92 = v86;
                if ( v86 != 0 && v86 != -4096 && v86 != -8192 )
                  sub_BD6050(v91, v85[0] & 0xFFFFFFFFFFFFFFF8LL);
                v93 = v87;
                v32 = *(_DWORD *)(v81 + 24);
                if ( !v32 )
                {
                  ++*(_QWORD *)v81;
                  goto LABEL_105;
                }
                v33 = v88.m128i_i64[0];
                v34 = *(_QWORD *)(v81 + 8);
                v35 = 1;
                v5 = v32 - 1;
                v36 = 0;
                for ( j = v5
                        & (((0xBF58476D1CE4E5B9LL
                           * (((unsigned __int32)v88.m128i_i32[2] >> 9) ^ ((unsigned __int32)v88.m128i_i32[2] >> 4)
                            | ((unsigned __int64)(((unsigned __int32)v88.m128i_i32[0] >> 9)
                                                ^ ((unsigned __int32)v88.m128i_i32[0] >> 4)) << 32))) >> 31)
                         ^ (484763065
                          * (((unsigned __int32)v88.m128i_i32[2] >> 9) ^ ((unsigned __int32)v88.m128i_i32[2] >> 4))));
                      ;
                      j = v5 & v40 )
                {
                  v38 = v34 + 72LL * j;
                  v39 = *(_QWORD *)v38;
                  if ( *(_OWORD *)v38 == *(_OWORD *)&v88 )
                  {
                    v43 = 0;
                    goto LABEL_83;
                  }
                  if ( v39 == -4096 )
                    break;
                  if ( v39 == -8192 && *(_QWORD *)(v38 + 8) == -8192 && !v36 )
                    v36 = v34 + 72LL * j;
LABEL_77:
                  v40 = v35 + j;
                  ++v35;
                }
                if ( *(_QWORD *)(v38 + 8) != -4096 )
                  goto LABEL_77;
                v55 = v81;
                if ( v36 )
                  v38 = v36;
                ++*(_QWORD *)v81;
                v56 = *(_DWORD *)(v55 + 16);
                v57 = v56 + 1;
                v75.m128i_i32[0] = v56;
                if ( 4 * (v56 + 1) < 3 * v32 )
                {
                  if ( v32 - *(_DWORD *)(v81 + 20) - v57 > v32 >> 3 )
                    goto LABEL_122;
                  v62 = v81;
                  sub_28F0570(v81, v32);
                  v63 = *(_DWORD *)(v62 + 24);
                  if ( !v63 )
                  {
LABEL_158:
                    ++*(_DWORD *)(v81 + 16);
                    BUG();
                  }
                  v33 = v88.m128i_i64[0];
                  v64 = v63 - 1;
                  v65 = *(_QWORD *)(v62 + 8);
                  v5 = 0;
                  v66 = 1;
                  v67 = v64
                      & (((0xBF58476D1CE4E5B9LL
                         * (((unsigned __int32)v88.m128i_i32[2] >> 9) ^ ((unsigned __int32)v88.m128i_i32[2] >> 4)
                          | ((unsigned __int64)(((unsigned __int32)v88.m128i_i32[0] >> 9)
                                              ^ ((unsigned __int32)v88.m128i_i32[0] >> 4)) << 32))) >> 31)
                       ^ (484763065
                        * (((unsigned __int32)v88.m128i_i32[2] >> 9) ^ ((unsigned __int32)v88.m128i_i32[2] >> 4))));
                  while ( 2 )
                  {
                    v38 = v65 + 72LL * v67;
                    v68 = *(_QWORD *)v38;
                    if ( *(_QWORD *)v38 == v88.m128i_i64[0] )
                    {
                      if ( *(_QWORD *)(v38 + 8) == v88.m128i_i64[1] )
                        goto LABEL_146;
                      if ( v68 == -4096 )
                        goto LABEL_150;
LABEL_137:
                      if ( v68 == -8192 && *(_QWORD *)(v38 + 8) == -8192 && !v5 )
                        v5 = v65 + 72LL * v67;
                    }
                    else
                    {
                      if ( v68 != -4096 )
                        goto LABEL_137;
LABEL_150:
                      if ( *(_QWORD *)(v38 + 8) == -4096 )
                        goto LABEL_151;
                    }
                    v69 = v66 + v67;
                    ++v66;
                    v67 = v64 & v69;
                    continue;
                  }
                }
LABEL_105:
                v47 = v81;
                sub_28F0570(v81, 2 * v32);
                v48 = *(_DWORD *)(v47 + 24);
                if ( !v48 )
                  goto LABEL_158;
                v33 = v88.m128i_i64[0];
                v49 = v48 - 1;
                v50 = *(_QWORD *)(v47 + 8);
                v5 = 0;
                v51 = 1;
                for ( k = v49
                        & (((0xBF58476D1CE4E5B9LL
                           * (((unsigned __int32)v88.m128i_i32[2] >> 9) ^ ((unsigned __int32)v88.m128i_i32[2] >> 4)
                            | ((unsigned __int64)(((unsigned __int32)v88.m128i_i32[0] >> 9)
                                                ^ ((unsigned __int32)v88.m128i_i32[0] >> 4)) << 32))) >> 31)
                         ^ (484763065
                          * (((unsigned __int32)v88.m128i_i32[2] >> 9) ^ ((unsigned __int32)v88.m128i_i32[2] >> 4))));
                      ;
                      k = v49 & v70 )
                {
                  while ( 1 )
                  {
                    v38 = v50 + 72LL * k;
                    v53 = *(_QWORD *)v38;
                    if ( *(_OWORD *)v38 == *(_OWORD *)&v88 )
                    {
LABEL_146:
                      v75.m128i_i32[0] = *(_DWORD *)(v81 + 16);
                      v57 = v75.m128i_i32[0] + 1;
                      goto LABEL_122;
                    }
                    if ( v53 == -4096 )
                      break;
                    if ( v53 == -8192 && *(_QWORD *)(v38 + 8) == -8192 && !v5 )
                      v5 = v50 + 72LL * k;
                    v54 = v51 + k;
                    ++v51;
                    k = v49 & v54;
                  }
                  if ( *(_QWORD *)(v38 + 8) == -4096 )
                    break;
                  v70 = v51 + k;
                  ++v51;
                }
LABEL_151:
                if ( v5 )
                  v38 = v5;
                v75.m128i_i32[0] = *(_DWORD *)(v81 + 16);
                v57 = v75.m128i_i32[0] + 1;
LABEL_122:
                *(_DWORD *)(v81 + 16) = v57;
                if ( *(_QWORD *)v38 != -4096 || *(_QWORD *)(v38 + 8) != -4096 )
                  --*(_DWORD *)(v81 + 20);
                *(_QWORD *)v38 = v33;
                v58 = v88.m128i_i64[1];
                *(_QWORD *)(v38 + 16) = 4;
                *(_QWORD *)(v38 + 8) = v58;
                *(_QWORD *)(v38 + 24) = 0;
                v59 = v90;
                v60 = v90 == 0;
                *(_QWORD *)(v38 + 32) = v90;
                if ( v59 != -4096 && !v60 && v59 != -8192 )
                  sub_BD6050((unsigned __int64 *)(v38 + 16), v89[0] & 0xFFFFFFFFFFFFFFF8LL);
                *(_QWORD *)(v38 + 40) = 4;
                *(_QWORD *)(v38 + 48) = 0;
                v61 = v92;
                v60 = v92 == -4096;
                *(_QWORD *)(v38 + 56) = v92;
                if ( v61 != 0 && !v60 && v61 != -8192 )
                  sub_BD6050((unsigned __int64 *)(v38 + 40), v91[0] & 0xFFFFFFFFFFFFFFF8LL);
                v43 = v79;
                *(_DWORD *)(v38 + 64) = v93;
LABEL_83:
                if ( v92 != 0 && v92 != -4096 && v92 != -8192 )
                  sub_BD60C0(v91);
                if ( v90 != 0 && v90 != -4096 && v90 != -8192 )
                  sub_BD60C0(v89);
                if ( v86 != 0 && v86 != -4096 && v86 != -8192 )
                  sub_BD60C0(v85);
                if ( v84 != 0 && v84 != -4096 && v84 != -8192 )
                  sub_BD60C0(v83);
                if ( !v43 )
                  ++*(_DWORD *)(v38 + 64);
LABEL_43:
                v12 = v98;
                v22 = v23 + 1;
                v23 = v22;
              }
              while ( (unsigned int)v22 < (unsigned int)v98 );
            }
          }
          while ( (unsigned __int64)v12 - 1 > v78 );
          v29 = v105;
          while ( v29 )
          {
            sub_28EA330(*(_QWORD *)(v29 + 24));
            v30 = v29;
            v29 = *(_QWORD *)(v29 + 16);
            j_j___libc_free_0(v30);
          }
          if ( v100 != (__m128i *)v102 )
            _libc_free((unsigned __int64)v100);
        }
      }
LABEL_49:
      if ( v97 != v99 )
        _libc_free((unsigned __int64)v97);
      if ( i != v96 )
        _libc_free((unsigned __int64)i);
LABEL_53:
      v77 = *(_QWORD *)(v77 + 8);
      if ( v74 == v77 )
        goto LABEL_54;
    }
    v17 = *(__int64 **)(v15 - 8);
    v18 = *v17;
    if ( v15 == *v17 )
      goto LABEL_26;
    if ( v14 <= HIDWORD(v95) )
      goto LABEL_24;
LABEL_144:
    sub_C8D5F0((__int64)&i, v96, v14, 8u, v14, v5);
    v10 = i;
    v11 = (unsigned int)v95;
    goto LABEL_24;
  }
  return result;
}
