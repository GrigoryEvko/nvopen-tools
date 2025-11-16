// Function: sub_2CB24E0
// Address: 0x2cb24e0
//
__int64 __fastcall sub_2CB24E0(
        unsigned __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        unsigned int *a7)
{
  unsigned int v7; // r14d
  unsigned __int8 v8; // al
  __int64 v9; // rbx
  __int64 v11; // r13
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // r10
  __int64 v18; // r11
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rdx
  const __m128i *v22; // r10
  const __m128i *v23; // rbx
  __m128i v24; // xmm0
  unsigned __int64 v25; // r11
  const __m128i *v26; // r15
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  __m128i *v29; // rdx
  unsigned __int64 v30; // r13
  __int64 v31; // r15
  __int32 v32; // eax
  __int64 v33; // r8
  __int64 v34; // r9
  __int32 v35; // ecx
  __int64 v36; // rsi
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  const __m128i *v41; // rdx
  __m128i *v42; // rax
  __int64 v43; // r12
  __int64 v45; // rdx
  unsigned int v46; // eax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rdx
  const __m128i *v53; // rbx
  __m128i *v54; // rax
  const void *v55; // rsi
  __int8 *v56; // r15
  unsigned int v57; // eax
  __int64 v58; // r8
  __int64 v59; // rdx
  unsigned __int64 v60; // rax
  __int64 v61; // r9
  __int64 v62; // rcx
  const __m128i *v63; // rax
  __m128i *v64; // rdx
  unsigned int v65; // eax
  __int64 v66; // r8
  __int64 v67; // r9
  unsigned __int64 v68; // rcx
  __int64 v69; // rax
  unsigned __int64 v70; // rdx
  __int64 v71; // rcx
  const __m128i *v72; // rdx
  __m128i *v73; // rax
  const void *v74; // rsi
  __int64 v75; // rdi
  __int64 v76; // rbx
  const void *v77; // rsi
  __int64 v78; // rbx
  const void *v79; // rsi
  unsigned __int64 v80; // r13
  const void *v81; // rsi
  const __m128i *v82; // [rsp+8h] [rbp-248h]
  unsigned __int64 v83; // [rsp+10h] [rbp-240h]
  __int64 v84; // [rsp+18h] [rbp-238h]
  __int64 v85; // [rsp+18h] [rbp-238h]
  __int64 v86; // [rsp+20h] [rbp-230h]
  __int64 v87; // [rsp+20h] [rbp-230h]
  unsigned int v88; // [rsp+28h] [rbp-228h]
  unsigned int v89; // [rsp+2Ch] [rbp-224h]
  unsigned __int8 v90; // [rsp+30h] [rbp-220h]
  const void *v91; // [rsp+30h] [rbp-220h]
  unsigned __int64 v93; // [rsp+38h] [rbp-218h]
  unsigned __int64 v94; // [rsp+38h] [rbp-218h]
  unsigned __int8 v95; // [rsp+38h] [rbp-218h]
  char v97; // [rsp+50h] [rbp-200h]
  _QWORD *v98; // [rsp+50h] [rbp-200h]
  __int64 v99; // [rsp+50h] [rbp-200h]
  __int64 v100; // [rsp+50h] [rbp-200h]
  int v101; // [rsp+58h] [rbp-1F8h]
  __m128i v103; // [rsp+60h] [rbp-1F0h] BYREF
  __int64 v104; // [rsp+70h] [rbp-1E0h]
  const __m128i *v105; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v106; // [rsp+88h] [rbp-1C8h]
  _BYTE v107[192]; // [rsp+90h] [rbp-1C0h] BYREF
  _QWORD *v108; // [rsp+150h] [rbp-100h] BYREF
  __int64 v109; // [rsp+158h] [rbp-F8h]
  _QWORD v110[30]; // [rsp+160h] [rbp-F0h] BYREF

  v7 = 0;
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 12 )
    return v7;
  v8 = *(_BYTE *)a2;
  v9 = a2;
  if ( *(_BYTE *)a2 <= 0x1Cu || (unsigned int)v8 - 42 > 0x11 )
    return v7;
  v101 = v8 - 29;
  if ( v8 != 46 )
  {
    if ( ((v8 - 42) & 0xFD) != 0 )
      return v7;
    v11 = a5;
    v12 = *a7;
    if ( *a7 )
    {
      v45 = *(_QWORD *)(a2 + 16);
      if ( !v45 || *(_QWORD *)(v45 + 8) )
        return v7;
    }
    v7 = 0;
    if ( v12 >= (unsigned int)qword_50131C8 )
      return v7;
    *a7 = v12 + 1;
    v86 = *(_QWORD *)(a2 - 64);
    v105 = (const __m128i *)v107;
    v106 = 0x800000000LL;
    v13 = sub_2CB24E0(a1, v86, a2, (unsigned int)&v105, a5, (_DWORD)a6, (__int64)a7);
    v90 = v13;
    v7 = v13;
    v84 = *(_QWORD *)(a2 - 32);
    v89 = *(_DWORD *)(v11 + 8);
    v108 = v110;
    v109 = 0x800000000LL;
    v14 = sub_2CB24E0(a1, v84, a2, (unsigned int)&v108, v11, (_DWORD)a6, (__int64)a7);
    v15 = v14;
    LOBYTE(v7) = v14 | v7;
    if ( !(_BYTE)v7 )
      goto LABEL_27;
    v16 = (__int64)a6;
    v17 = a1;
    v18 = v86;
    v88 = *(_DWORD *)(v11 + 8);
    v19 = a6[2];
    if ( v19 + 1 > (unsigned __int64)a6[3] )
    {
      v74 = a6 + 4;
      v75 = (__int64)a6;
      v83 = a1;
      v95 = v15;
      v100 = v16;
      sub_C8D5F0(v75, v74, v19 + 1, 8u, v15, v16);
      v16 = v100;
      v17 = v83;
      v18 = v86;
      v15 = v95;
      v19 = *(unsigned int *)(v100 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v16 + 8 * v19) = v9;
    ++*(_DWORD *)(v16 + 8);
    if ( v90 )
    {
      v20 = (__int64)&v105->m128i_i64[3 * (unsigned int)v106];
      if ( (const __m128i *)v20 != v105 )
      {
        v97 = v15;
        v21 = *(unsigned int *)(a4 + 8);
        v93 = v17;
        v22 = (const __m128i *)((char *)v105 + 24 * (unsigned int)v106);
        v87 = v9;
        v23 = v105;
        do
        {
          v24 = _mm_loadu_si128(v23);
          v25 = v21 + 1;
          v26 = &v103;
          v27 = *(unsigned int *)(a4 + 12);
          v104 = v23[1].m128i_i64[0];
          v28 = *(_QWORD *)a4;
          v103 = v24;
          if ( v21 + 1 > v27 )
          {
            v82 = v22;
            v55 = (const void *)(a4 + 16);
            if ( v28 > (unsigned __int64)&v103 || (unsigned __int64)&v103 >= v28 + 24 * v21 )
            {
              v26 = &v103;
              sub_C8D5F0(a4, v55, v25, 0x18u, v15, v20);
              v28 = *(_QWORD *)a4;
              v21 = *(unsigned int *)(a4 + 8);
              v22 = v82;
            }
            else
            {
              v56 = &v103.m128i_i8[-v28];
              sub_C8D5F0(a4, v55, v25, 0x18u, v15, v20);
              v28 = *(_QWORD *)a4;
              v21 = *(unsigned int *)(a4 + 8);
              v22 = v82;
              v26 = (const __m128i *)&v56[*(_QWORD *)a4];
            }
          }
          v23 = (const __m128i *)((char *)v23 + 24);
          v29 = (__m128i *)(v28 + 24 * v21);
          *v29 = _mm_loadu_si128(v26);
          v29[1].m128i_i64[0] = v26[1].m128i_i64[0];
          v21 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
          *(_DWORD *)(a4 + 8) = v21;
        }
        while ( v22 != v23 );
        LOBYTE(v15) = v97;
        v17 = v93;
        v9 = v87;
      }
      if ( (_BYTE)v15 )
        goto LABEL_18;
      v65 = sub_2CAFE10(v17, v84);
      if ( v65 )
      {
        v103.m128i_i64[1] = __PAIR64__(v101, v65);
        v68 = *(unsigned int *)(v11 + 12);
        v104 = v9;
        v69 = *(unsigned int *)(v11 + 8);
        v103.m128i_i64[0] = v84;
        v70 = v69 + 1;
        if ( v69 + 1 > v68 )
        {
          v78 = *(_QWORD *)v11;
          v79 = (const void *)(v11 + 16);
          if ( *(_QWORD *)v11 > (unsigned __int64)&v103 || (unsigned __int64)&v103 >= v78 + 24 * v69 )
          {
            sub_C8D5F0(v11, v79, v70, 0x18u, v66, v67);
            v69 = *(unsigned int *)(v11 + 8);
            v72 = &v103;
            v71 = *(_QWORD *)v11;
          }
          else
          {
            sub_C8D5F0(v11, v79, v70, 0x18u, v66, v67);
            v71 = *(_QWORD *)v11;
            v69 = *(unsigned int *)(v11 + 8);
            v72 = (__m128i *)((char *)&v103 + *(_QWORD *)v11 - v78);
          }
        }
        else
        {
          v71 = *(_QWORD *)v11;
          v72 = &v103;
        }
        v7 = v90;
        v73 = (__m128i *)(v71 + 24 * v69);
        *v73 = _mm_loadu_si128(v72);
        v73[1].m128i_i64[0] = v72[1].m128i_i64[0];
        ++*(_DWORD *)(v11 + 8);
LABEL_27:
        if ( v108 != v110 )
          _libc_free((unsigned __int64)v108);
        if ( v105 != (const __m128i *)v107 )
          _libc_free((unsigned __int64)v105);
        return v7;
      }
    }
    else
    {
      v99 = v18;
      v57 = sub_2CAFE10(v17, v18);
      if ( v57 )
      {
        v59 = *(unsigned int *)(v11 + 8);
        v103.m128i_i64[1] = v57 | 0xD00000000LL;
        v60 = *(unsigned int *)(v11 + 12);
        v103.m128i_i64[0] = v99;
        v61 = v59 + 1;
        v104 = v9;
        if ( v59 + 1 > v60 )
        {
          v76 = *(_QWORD *)v11;
          v77 = (const void *)(v11 + 16);
          if ( *(_QWORD *)v11 > (unsigned __int64)&v103 || (unsigned __int64)&v103 >= v76 + 24 * v59 )
          {
            sub_C8D5F0(v11, v77, v59 + 1, 0x18u, v58, v61);
            v59 = *(unsigned int *)(v11 + 8);
            v63 = &v103;
            v62 = *(_QWORD *)v11;
          }
          else
          {
            sub_C8D5F0(v11, v77, v59 + 1, 0x18u, v58, v61);
            v62 = *(_QWORD *)v11;
            v59 = *(unsigned int *)(v11 + 8);
            v63 = (__m128i *)((char *)&v103 + *(_QWORD *)v11 - v76);
          }
        }
        else
        {
          v62 = *(_QWORD *)v11;
          v63 = &v103;
        }
        v64 = (__m128i *)(v62 + 24 * v59);
        *v64 = _mm_loadu_si128(v63);
        v64[1].m128i_i64[0] = v63[1].m128i_i64[0];
        ++*(_DWORD *)(v11 + 8);
LABEL_18:
        v91 = (const void *)(a4 + 16);
        v98 = &v108[3 * (unsigned int)v109];
        if ( v98 != v108 )
        {
          v85 = v11;
          v30 = (unsigned __int64)v108;
          do
          {
            v31 = *(_QWORD *)(v30 + 16);
            v32 = sub_2CB24C0(v101, *(_DWORD *)(v30 + 12));
            v35 = *(_DWORD *)(v30 + 8);
            v36 = *(_QWORD *)v30;
            v104 = v31;
            v103.m128i_i32[3] = v32;
            v37 = *(unsigned int *)(a4 + 8);
            v103.m128i_i32[2] = v35;
            v38 = *(unsigned int *)(a4 + 12);
            v39 = v37 + 1;
            v103.m128i_i64[0] = v36;
            if ( v37 + 1 > v38 )
            {
              if ( *(_QWORD *)a4 > (unsigned __int64)&v103
                || (v94 = *(_QWORD *)a4, (unsigned __int64)&v103 >= *(_QWORD *)a4 + 24 * v37) )
              {
                sub_C8D5F0(a4, v91, v39, 0x18u, v33, v34);
                v40 = *(_QWORD *)a4;
                v37 = *(unsigned int *)(a4 + 8);
                v41 = &v103;
              }
              else
              {
                sub_C8D5F0(a4, v91, v39, 0x18u, v33, v34);
                v40 = *(_QWORD *)a4;
                v37 = *(unsigned int *)(a4 + 8);
                v41 = (__m128i *)((char *)&v103 + *(_QWORD *)a4 - v94);
              }
            }
            else
            {
              v40 = *(_QWORD *)a4;
              v41 = &v103;
            }
            v30 += 24LL;
            v42 = (__m128i *)(v40 + 24 * v37);
            *v42 = _mm_loadu_si128(v41);
            v42[1].m128i_i64[0] = v41[1].m128i_i64[0];
            ++*(_DWORD *)(a4 + 8);
          }
          while ( v98 != (_QWORD *)v30 );
          v7 = (unsigned __int8)v7;
          v11 = v85;
        }
        if ( v88 > v89 )
        {
          v43 = 24LL * v89;
          do
          {
            *(_DWORD *)(*(_QWORD *)v11 + v43 + 12) = sub_2CB24C0(v101, *(_DWORD *)(*(_QWORD *)v11 + v43 + 12));
            v43 += 24;
          }
          while ( v43 != 24 * (v89 + (unsigned __int64)(v88 - 1 - v89) + 1) );
        }
        goto LABEL_27;
      }
    }
    v7 = 0;
    goto LABEL_27;
  }
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL) + 8LL) == 12
    && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL) + 8LL) == 12 )
  {
    v46 = sub_2CAFE10(a1, a2);
    if ( v46 )
    {
      v109 = v46 | 0xD00000000LL;
      v49 = *(unsigned int *)(a4 + 8);
      v50 = *(unsigned int *)(a4 + 12);
      v110[0] = a3;
      v51 = v49 + 1;
      v108 = (_QWORD *)a2;
      if ( v49 + 1 > v50 )
      {
        v80 = *(_QWORD *)a4;
        v53 = (const __m128i *)&v108;
        v81 = (const void *)(a4 + 16);
        if ( *(_QWORD *)a4 > (unsigned __int64)&v108 || (unsigned __int64)&v108 >= v80 + 24 * v49 )
        {
          sub_C8D5F0(a4, v81, v51, 0x18u, v47, v48);
          v49 = *(unsigned int *)(a4 + 8);
          v52 = *(_QWORD *)a4;
        }
        else
        {
          sub_C8D5F0(a4, v81, v51, 0x18u, v47, v48);
          v52 = *(_QWORD *)a4;
          v49 = *(unsigned int *)(a4 + 8);
          v53 = (const __m128i *)((char *)&v108 + *(_QWORD *)a4 - v80);
        }
      }
      else
      {
        v52 = *(_QWORD *)a4;
        v53 = (const __m128i *)&v108;
      }
      v7 = 1;
      v54 = (__m128i *)(v52 + 24 * v49);
      *v54 = _mm_loadu_si128(v53);
      v54[1].m128i_i64[0] = v53[1].m128i_i64[0];
      ++*(_DWORD *)(a4 + 8);
    }
  }
  return v7;
}
