// Function: sub_1B2E010
// Address: 0x1b2e010
//
void __fastcall sub_1B2E010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r13
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 *v10; // rdx
  signed __int64 v11; // r14
  __int64 *v12; // rax
  __int64 *v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // rax
  unsigned __int64 v16; // rbx
  char *v17; // r14
  unsigned __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r8
  unsigned int v27; // edi
  __int64 *v28; // rdx
  __int64 v29; // r11
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __m128i *v33; // rax
  __m128i v34; // xmm1
  __int64 v35; // rcx
  int v36; // esi
  int v37; // edx
  __int64 v38; // rdi
  __int64 v39; // r8
  int v40; // edx
  int v41; // r11d
  unsigned __int64 v42; // r10
  unsigned __int64 v43; // r10
  unsigned int j; // eax
  _QWORD *v45; // r10
  unsigned int v46; // eax
  __int64 *v47; // r15
  __int64 v48; // rbx
  __int64 *i; // r14
  __int64 v50; // rdx
  __int64 *v51; // r12
  __m128i *v52; // r14
  __int64 v53; // rdx
  const __m128i *v54; // r15
  const __m128i *v55; // r14
  char *v56; // rbx
  __int64 v57; // r15
  int v58; // r8d
  int v59; // r9d
  __int64 v60; // rax
  __m128i *v61; // rax
  __int64 v62; // rax
  __int64 v63; // r15
  __int64 v64; // rdx
  __int64 *v65; // rax
  __int64 v66; // rsi
  unsigned __int64 v67; // rcx
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r8
  int v72; // r9d
  unsigned int v73; // edx
  __int64 *v74; // rsi
  __int64 v75; // r10
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __m128i *v79; // rax
  __m128i v80; // xmm6
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rdi
  unsigned int v84; // esi
  __int64 *v85; // rax
  __int64 v86; // r9
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __m128i *v90; // rax
  __m128i v91; // xmm7
  __int64 v92; // rax
  int v93; // eax
  int v94; // r11d
  int v95; // esi
  int v96; // r11d
  __int64 v97; // [rsp+18h] [rbp-568h]
  __int64 *src; // [rsp+28h] [rbp-558h]
  unsigned __int64 *v99; // [rsp+38h] [rbp-548h]
  __int64 *v100; // [rsp+48h] [rbp-538h]
  __int64 *v101; // [rsp+48h] [rbp-538h]
  int v102; // [rsp+5Ch] [rbp-524h] BYREF
  __int64 *v103; // [rsp+60h] [rbp-520h] BYREF
  __int64 v104; // [rsp+68h] [rbp-518h]
  _BYTE v105[64]; // [rsp+70h] [rbp-510h] BYREF
  __m128i v106; // [rsp+B0h] [rbp-4D0h] BYREF
  __m128i v107; // [rsp+C0h] [rbp-4C0h] BYREF
  __m128i v108[23]; // [rsp+D0h] [rbp-4B0h] BYREF
  void *v109; // [rsp+240h] [rbp-340h] BYREF
  __int64 v110; // [rsp+248h] [rbp-338h]
  _BYTE v111[816]; // [rsp+250h] [rbp-330h] BYREF

  v6 = a1;
  v7 = *(__int64 **)(a2 + 16);
  if ( v7 == *(__int64 **)(a2 + 8) )
    v8 = *(unsigned int *)(a2 + 28);
  else
    v8 = *(unsigned int *)(a2 + 24);
  v9 = &v7[v8];
  while ( 1 )
  {
    if ( v9 == v7 )
      goto LABEL_7;
    if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v7;
  }
  v103 = (__int64 *)v105;
  v104 = 0x800000000LL;
  if ( v9 == v7 )
  {
LABEL_7:
    src = (__int64 *)v105;
    goto LABEL_8;
  }
  v10 = v7;
  v11 = 0;
  while ( 1 )
  {
    v12 = v10 + 1;
    if ( v9 == v10 + 1 )
      break;
    while ( 1 )
    {
      v10 = v12;
      if ( (unsigned __int64)*v12 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v9 == ++v12 )
        goto LABEL_16;
    }
    ++v11;
    if ( v9 == v12 )
      goto LABEL_17;
  }
LABEL_16:
  ++v11;
LABEL_17:
  v13 = (__int64 *)v105;
  if ( v11 > 8 )
  {
    sub_16CD150((__int64)&v103, v105, v11, 8, a5, a6);
    v13 = &v103[(unsigned int)v104];
  }
  v14 = *v7;
  do
  {
    v15 = v7 + 1;
    *v13++ = v14;
    if ( v9 == v7 + 1 )
      break;
    while ( 1 )
    {
      v14 = *v15;
      v7 = v15;
      if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v9 == ++v15 )
        goto LABEL_23;
    }
  }
  while ( v9 != v15 );
LABEL_23:
  src = v103;
  LODWORD(v104) = v11 + v104;
  v16 = (unsigned int)v104;
  v17 = (char *)&v103[v16];
  if ( &v103[v16] != v103 )
  {
    _BitScanReverse64(&v18, (__int64)(v16 * 8) >> 3);
    sub_1B2A730(v103, &v103[v16], 2LL * (int)(63 - (v18 ^ 0x3F)), a1);
    v97 = a1 + 40;
    if ( v16 > 16 )
    {
      v47 = src + 16;
      sub_1B2A9B0((char *)src, (char *)src + 128, a1);
      if ( v17 != (char *)(src + 16) )
      {
        v101 = (__int64 *)v17;
        do
        {
          v48 = *v47;
          for ( i = v47; ; i[1] = *i )
          {
            v50 = *(i - 1);
            v51 = i--;
            if ( !sub_1B29A30(a1 + 40, v48, v50) )
              break;
          }
          *v51 = v48;
          ++v47;
        }
        while ( v101 != v47 );
        v6 = a1;
      }
    }
    else
    {
      sub_1B2A9B0((char *)src, v17, a1);
    }
    src = &v103[(unsigned int)v104];
    if ( v103 != src )
    {
      v100 = v103;
      while ( 1 )
      {
        v19 = *v100;
        v102 = 0;
        v109 = v111;
        v99 = (unsigned __int64 *)v19;
        v110 = 0x1000000000LL;
        v20 = sub_1B2B830(v6, v19);
        v21 = *(__int64 **)v20;
        v22 = *(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 8);
        if ( *(_QWORD *)v20 != v22 )
          break;
LABEL_60:
        sub_1B2B520(v6, (__int64)v99, (__int64)&v109);
        v52 = (__m128i *)v109;
        v53 = 48LL * (unsigned int)v110;
        v54 = (const __m128i *)((char *)v109 + v53);
        sub_1B2C540(v106.m128i_i64, (__m128i *)v109, 0xAAAAAAAAAAAAAAABLL * (v53 >> 4));
        if ( v107.m128i_i64[0] )
          sub_1B2DEF0(v52, v54, (__m128i *)v107.m128i_i64[0], v106.m128i_i64[1], v97);
        else
          sub_1B2CB70(v52, v54, v97);
        j_j___libc_free_0(v107.m128i_i64[0], 48 * v106.m128i_i64[1]);
        v55 = (const __m128i *)v109;
        v106.m128i_i64[0] = (__int64)&v107;
        v106.m128i_i64[1] = 0x800000000LL;
        v56 = (char *)v109 + 48 * (unsigned int)v110;
        if ( v56 == v109 )
          goto LABEL_82;
        do
        {
          while ( 1 )
          {
            v57 = v55[2].m128i_i64[0];
            if ( v55[1].m128i_i64[0] | v57 )
            {
              sub_1B2B3A0(v6, (__int64)&v106, (__int64)v55);
              sub_1B2B4D0(v6, (__int64)&v106, (__int64)v55);
              v60 = v106.m128i_u32[2];
              if ( v106.m128i_i32[2] >= (unsigned __int32)v106.m128i_i32[3] )
              {
                sub_16CD150((__int64)&v106, &v107, 0, 48, v58, v59);
                v60 = v106.m128i_u32[2];
              }
              v61 = (__m128i *)(v106.m128i_i64[0] + 48 * v60);
              *v61 = _mm_loadu_si128(v55);
              v61[1] = _mm_loadu_si128(v55 + 1);
              v61[2] = _mm_loadu_si128(v55 + 2);
              v62 = (unsigned int)++v106.m128i_i32[2];
            }
            else
            {
              if ( !(unsigned __int8)sub_1B2B3A0(v6, (__int64)&v106, (__int64)v55) )
                sub_1B2B4D0(v6, (__int64)&v106, (__int64)v55);
              v62 = v106.m128i_u32[2];
            }
            if ( !(_DWORD)v62 || v55[1].m128i_i64[0] | v57 )
              goto LABEL_64;
            v63 = v106.m128i_i64[0] + 48 * v62 - 48;
            v64 = *(_QWORD *)(v63 + 16);
            if ( !v64 )
            {
              v92 = sub_1B2D320(v6, &v102, v106.m128i_i64, v99);
              *(_QWORD *)(v63 + 16) = v92;
              v64 = v92;
              v65 = (__int64 *)v55[1].m128i_i64[1];
              if ( !*v65 )
                goto LABEL_75;
              goto LABEL_73;
            }
            v65 = (__int64 *)v55[1].m128i_i64[1];
            if ( !*v65 )
              break;
LABEL_73:
            v66 = v65[1];
            v67 = v65[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v67 = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = *(_QWORD *)(v66 + 16) & 3LL | v67;
LABEL_75:
            *v65 = v64;
            if ( v64 )
              goto LABEL_76;
LABEL_64:
            v55 += 3;
            if ( v56 == (char *)v55 )
              goto LABEL_79;
          }
          *v65 = v64;
LABEL_76:
          v68 = *(_QWORD *)(v64 + 8);
          v65[1] = v68;
          if ( v68 )
            *(_QWORD *)(v68 + 16) = (unsigned __int64)(v65 + 1) | *(_QWORD *)(v68 + 16) & 3LL;
          v55 += 3;
          v65[2] = (v64 + 8) | v65[2] & 3;
          *(_QWORD *)(v64 + 8) = v65;
        }
        while ( v56 != (char *)v55 );
LABEL_79:
        if ( (__m128i *)v106.m128i_i64[0] != &v107 )
          _libc_free(v106.m128i_u64[0]);
        v55 = (const __m128i *)v109;
LABEL_82:
        if ( v55 != (const __m128i *)v111 )
          _libc_free((unsigned __int64)v55);
        if ( src == ++v100 )
        {
          src = v103;
          goto LABEL_8;
        }
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v106.m128i_i64[0] = 0;
          v106.m128i_i32[2] = 1;
          v107 = 0u;
          v108[0].m128i_i64[0] = 0;
          v108[0].m128i_i8[8] = 0;
          v35 = *v21;
          v36 = *(_DWORD *)(*v21 + 24);
          if ( v36 != 1 )
          {
            if ( (v36 & 0xFFFFFFFD) == 0 )
            {
              v37 = *(_DWORD *)(v6 + 3256);
              v38 = *(_QWORD *)(v35 + 56);
              if ( v37 )
              {
                v39 = *(_QWORD *)(v35 + 48);
                v40 = v37 - 1;
                v41 = 1;
                v42 = (((((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)
                       | ((unsigned __int64)(((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)) << 32))
                      - 1
                      - ((unsigned __int64)(((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)) << 32)) >> 22)
                    ^ ((((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)
                      | ((unsigned __int64)(((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)) << 32))
                     - 1
                     - ((unsigned __int64)(((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)) << 32));
                v43 = ((9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13)))) >> 15)
                    ^ (9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13))));
                for ( j = v40 & (((v43 - 1 - (v43 << 27)) >> 31) ^ (v43 - 1 - ((_DWORD)v43 << 27))); ; j = v40 & v46 )
                {
                  v45 = (_QWORD *)(*(_QWORD *)(v6 + 3240) + 16LL * j);
                  if ( v39 == *v45 && v38 == v45[1] )
                    break;
                  if ( *v45 == -8 && v45[1] == -8 )
                    goto LABEL_90;
                  v46 = v41 + j;
                  ++v41;
                }
                v106.m128i_i32[2] = 2;
                v81 = *(_QWORD *)(v6 + 24);
                v82 = *(unsigned int *)(v81 + 48);
                v83 = *(_QWORD *)(v81 + 32);
                if ( (_DWORD)v82 )
                {
                  v84 = (v82 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                  v85 = (__int64 *)(v83 + 16LL * v84);
                  v86 = *v85;
                  if ( v39 == *v85 )
                  {
LABEL_102:
                    if ( v85 != (__int64 *)(v83 + 16 * v82) )
                    {
                      v87 = v85[1];
                      if ( v87 )
                      {
                        v88 = *(_QWORD *)(v87 + 48);
                        v108[0].m128i_i64[0] = v35;
                        v108[0].m128i_i8[8] = 1;
                        v106.m128i_i64[0] = v88;
                        v89 = (unsigned int)v110;
                        if ( (unsigned int)v110 >= HIDWORD(v110) )
                        {
                          sub_16CD150((__int64)&v109, v111, 0, 48, v39, v86);
                          v89 = (unsigned int)v110;
                        }
                        v90 = (__m128i *)((char *)v109 + 48 * v89);
                        *v90 = _mm_loadu_si128(&v106);
                        v91 = _mm_loadu_si128(&v107);
                        LODWORD(v110) = v110 + 1;
                        v90[1] = v91;
                        v90[2] = _mm_loadu_si128(v108);
                      }
                    }
                  }
                  else
                  {
                    v93 = 1;
                    while ( v86 != -8 )
                    {
                      v94 = v93 + 1;
                      v84 = (v82 - 1) & (v93 + v84);
                      v85 = (__int64 *)(v83 + 16LL * v84);
                      v86 = *v85;
                      if ( v39 == *v85 )
                        goto LABEL_102;
                      v93 = v94;
                    }
                  }
                }
              }
              else
              {
LABEL_90:
                v106.m128i_i32[2] = 0;
                v69 = *(_QWORD *)(v6 + 24);
                v70 = *(unsigned int *)(v69 + 48);
                if ( (_DWORD)v70 )
                {
                  v71 = *(_QWORD *)(v69 + 32);
                  v72 = v70 - 1;
                  v73 = (v70 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                  v74 = (__int64 *)(v71 + 16LL * v73);
                  v75 = *v74;
                  if ( v38 == *v74 )
                  {
LABEL_92:
                    if ( v74 != (__int64 *)(v71 + 16 * v70) )
                    {
                      v76 = v74[1];
                      if ( v76 )
                      {
                        v77 = *(_QWORD *)(v76 + 48);
                        v108[0].m128i_i64[0] = v35;
                        v106.m128i_i64[0] = v77;
                        v78 = (unsigned int)v110;
                        if ( (unsigned int)v110 >= HIDWORD(v110) )
                        {
                          sub_16CD150((__int64)&v109, v111, 0, 48, v71, v72);
                          v78 = (unsigned int)v110;
                        }
                        v79 = (__m128i *)((char *)v109 + 48 * v78);
                        *v79 = _mm_loadu_si128(&v106);
                        v80 = _mm_loadu_si128(&v107);
                        LODWORD(v110) = v110 + 1;
                        v79[1] = v80;
                        v79[2] = _mm_loadu_si128(v108);
                      }
                    }
                  }
                  else
                  {
                    v95 = 1;
                    while ( v75 != -8 )
                    {
                      v96 = v95 + 1;
                      v73 = v72 & (v95 + v73);
                      v74 = (__int64 *)(v71 + 16LL * v73);
                      v75 = *v74;
                      if ( v38 == *v74 )
                        goto LABEL_92;
                      v95 = v96;
                    }
                  }
                }
              }
            }
            goto LABEL_37;
          }
          v23 = *(_QWORD *)(v6 + 24);
          v24 = *(unsigned int *)(v23 + 48);
          if ( (_DWORD)v24 )
            break;
LABEL_37:
          if ( (__int64 *)v22 == ++v21 )
            goto LABEL_60;
        }
        v25 = *(_QWORD *)(v23 + 32);
        v26 = *(_QWORD *)(*(_QWORD *)(v35 + 48) + 40LL);
        v27 = (v24 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( v26 == *v28 )
        {
LABEL_32:
          if ( v28 != (__int64 *)(v25 + 16 * v24) )
          {
            v30 = v28[1];
            if ( v30 )
            {
              v31 = *(_QWORD *)(v30 + 48);
              v108[0].m128i_i64[0] = *v21;
              v106.m128i_i64[0] = v31;
              v32 = (unsigned int)v110;
              if ( (unsigned int)v110 >= HIDWORD(v110) )
              {
                sub_16CD150((__int64)&v109, v111, 0, 48, v26, v25);
                v32 = (unsigned int)v110;
              }
              v33 = (__m128i *)((char *)v109 + 48 * v32);
              *v33 = _mm_loadu_si128(&v106);
              v34 = _mm_loadu_si128(&v107);
              LODWORD(v110) = v110 + 1;
              v33[1] = v34;
              v33[2] = _mm_loadu_si128(v108);
            }
          }
          goto LABEL_37;
        }
        while ( v29 != -8 )
        {
          v27 = (v24 - 1) & (v36 + v27);
          v28 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v28;
          if ( v26 == *v28 )
            goto LABEL_32;
          ++v36;
        }
        if ( (__int64 *)v22 == ++v21 )
          goto LABEL_60;
      }
    }
  }
LABEL_8:
  if ( src != (__int64 *)v105 )
    _libc_free((unsigned __int64)src);
}
