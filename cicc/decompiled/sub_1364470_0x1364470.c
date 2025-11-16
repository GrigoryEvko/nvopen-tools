// Function: sub_1364470
// Address: 0x1364470
//
__int64 __fastcall sub_1364470(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 *a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  unsigned __int64 v12; // rbx
  __int64 v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rdi
  unsigned int v18; // r15d
  __int64 *v20; // rdx
  __int64 *v21; // r9
  __int64 v22; // r8
  __int64 v23; // rdx
  _QWORD *v24; // rdi
  _QWORD *v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rax
  __m128i v32; // xmm0
  __m128i v33; // xmm1
  char v34; // al
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 *v37; // r13
  __int64 v38; // r12
  unsigned __int64 v39; // rsi
  char v40; // r8
  __int64 v41; // rax
  unsigned int v42; // edi
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  char v47; // al
  __int64 *v48; // rdx
  __int64 v49; // rax
  int v50; // eax
  __m128i *v51; // r10
  __int64 v52; // rax
  __int64 *v53; // r9
  unsigned __int64 v54; // rsi
  __m128i *v55; // rdi
  __int64 *v56; // r12
  __int64 *v57; // rbx
  __int64 v58; // r13
  unsigned __int8 v59; // al
  char v60; // dl
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  int v63; // r13d
  int v64; // ebx
  unsigned __int64 v65; // r15
  char v66; // al
  _QWORD *v67; // rax
  __m128i *v68; // rsi
  __int64 *v69; // rcx
  __m128i *v70; // rax
  __int64 v71; // rax
  unsigned __int64 v72; // rsi
  __int64 *v73; // r15
  unsigned __int64 v74; // rax
  unsigned __int8 v75; // cl
  __int64 v76; // rdx
  __int64 *v77; // rdx
  _QWORD *v78; // rcx
  __int64 *v79; // rax
  __int64 v80; // rdx
  char v81; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v82; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v83; // [rsp+10h] [rbp-E0h]
  __int64 *v84; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v85; // [rsp+18h] [rbp-D8h]
  int v86; // [rsp+18h] [rbp-D8h]
  __int64 v87; // [rsp+18h] [rbp-D8h]
  __int64 v88; // [rsp+20h] [rbp-D0h]
  __int64 *v89; // [rsp+20h] [rbp-D0h]
  __int64 v90; // [rsp+20h] [rbp-D0h]
  __int64 v91; // [rsp+28h] [rbp-C8h]
  char v92; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v93; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v95; // [rsp+38h] [rbp-B8h]
  __int64 *v96; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+48h] [rbp-A8h]
  _BYTE v98[32]; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v99; // [rsp+70h] [rbp-80h] BYREF
  __m128i v100; // [rsp+80h] [rbp-70h] BYREF
  __int64 v101; // [rsp+90h] [rbp-60h]
  __m128i v102; // [rsp+98h] [rbp-58h] BYREF
  __m128i v103; // [rsp+A8h] [rbp-48h] BYREF
  __int64 v104; // [rsp+B8h] [rbp-38h]

  v12 = a5;
  v13 = *(_QWORD *)(a2 + 40);
  v14 = *(_QWORD **)(a1 + 792);
  if ( *(_QWORD **)(a1 + 800) != v14 )
    goto LABEL_2;
  v23 = *(unsigned int *)(a1 + 812);
  v24 = &v14[v23];
  if ( v14 == v24 )
  {
LABEL_124:
    if ( (unsigned int)v23 >= *(_DWORD *)(a1 + 808) )
    {
LABEL_2:
      sub_16CCBA0(a1 + 784, v13);
      goto LABEL_3;
    }
    *(_DWORD *)(a1 + 812) = v23 + 1;
    *v24 = v13;
    ++*(_QWORD *)(a1 + 784);
  }
  else
  {
    v25 = 0;
    while ( v13 != *v14 )
    {
      if ( *v14 == -2 )
        v25 = v14;
      if ( v24 == ++v14 )
      {
        if ( !v25 )
          goto LABEL_124;
        *v25 = v13;
        --*(_DWORD *)(a1 + 816);
        ++*(_QWORD *)(a1 + 784);
        if ( *(_BYTE *)(a5 + 16) != 77 )
          goto LABEL_4;
        goto LABEL_24;
      }
    }
  }
LABEL_3:
  if ( *(_BYTE *)(v12 + 16) == 77 )
  {
LABEL_24:
    if ( *(_QWORD *)(v12 + 40) == *(_QWORD *)(a2 + 40) )
    {
      v26 = *a4;
      v99.m128i_i64[0] = a2;
      v27 = a4[1];
      v102.m128i_i64[0] = v12;
      v28 = *a7;
      v29 = a7[1];
      v100.m128i_i64[0] = v26;
      v30 = a7[2];
      v31 = a4[2];
      v99.m128i_i64[1] = a3;
      v100.m128i_i64[1] = v27;
      v101 = v31;
      v102.m128i_i64[1] = a6;
      v103.m128i_i64[0] = v28;
      v103.m128i_i64[1] = v29;
      v104 = v30;
      if ( a2 > v12 )
      {
        v32 = _mm_loadu_si128(&v102);
        v33 = _mm_loadu_si128(&v103);
        v101 = v30;
        v102.m128i_i64[0] = a2;
        v102.m128i_i64[1] = a3;
        v103.m128i_i64[0] = v26;
        v103.m128i_i64[1] = v27;
        v104 = v31;
        v99 = v32;
        v100 = v33;
      }
      v91 = a1 + 64;
      if ( (unsigned __int8)sub_1361B70(a1 + 64, v99.m128i_i64, &v96) )
      {
        v81 = *((_BYTE *)v96 + 80);
      }
      else
      {
        v79 = sub_1362710(v91, v99.m128i_i64, v96);
        v81 = 0;
        *(__m128i *)v79 = _mm_loadu_si128(&v99);
        *((__m128i *)v79 + 1) = _mm_loadu_si128(&v100);
        v79[4] = v101;
        *(__m128i *)(v79 + 5) = _mm_loadu_si128(&v102);
        *(__m128i *)(v79 + 7) = _mm_loadu_si128(&v103);
        v80 = v104;
        *((_BYTE *)v79 + 80) = 0;
        v79[9] = v80;
      }
      v34 = sub_1361B70(v91, v99.m128i_i64, &v96);
      v35 = v96;
      if ( !v34 )
      {
        v35 = sub_1362710(v91, v99.m128i_i64, v96);
        *(__m128i *)v35 = _mm_loadu_si128(&v99);
        *((__m128i *)v35 + 1) = _mm_loadu_si128(&v100);
        v35[4] = v101;
        *(__m128i *)(v35 + 5) = _mm_loadu_si128(&v102);
        *(__m128i *)(v35 + 7) = _mm_loadu_si128(&v103);
        v36 = v104;
        *((_BYTE *)v35 + 80) = 0;
        v35[9] = v36;
      }
      *((_BYTE *)v35 + 80) = 0;
      v88 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
      {
        v93 = a6;
        v37 = a4;
        v38 = 0;
        do
        {
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v39 = *(_QWORD *)(a2 - 8);
          else
            v39 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v40 = *(_BYTE *)(v12 + 23) & 0x40;
          v41 = 0x17FFFFFFE8LL;
          v42 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
          if ( v42 )
          {
            v43 = 24LL * *(unsigned int *)(v12 + 56) + 8;
            v44 = 0;
            do
            {
              v45 = v12 - 24LL * v42;
              if ( v40 )
                v45 = *(_QWORD *)(v12 - 8);
              if ( *(_QWORD *)(v38 + v39 + 24LL * *(unsigned int *)(a2 + 56) + 8) == *(_QWORD *)(v45 + v43) )
              {
                v41 = 24 * v44;
                goto LABEL_42;
              }
              ++v44;
              v43 += 8;
            }
            while ( v42 != (_DWORD)v44 );
            v41 = 0x17FFFFFFE8LL;
          }
LABEL_42:
          if ( v40 )
            v46 = *(_QWORD *)(v12 - 8);
          else
            v46 = v12 - 24LL * v42;
          if ( (unsigned __int8)sub_1362890(
                                  (__int64 *)a1,
                                  *(_QWORD *)(v39 + 3 * v38),
                                  a3,
                                  *(_QWORD *)(v46 + v41),
                                  v93,
                                  0,
                                  *(_OWORD *)v37,
                                  v37[2],
                                  *(_OWORD *)a7,
                                  a7[2],
                                  0) )
          {
            v47 = sub_1361B70(v91, v99.m128i_i64, &v96);
            v48 = v96;
            if ( !v47 )
            {
              v48 = sub_1362710(v91, v99.m128i_i64, v96);
              *(__m128i *)v48 = _mm_loadu_si128(&v99);
              *((__m128i *)v48 + 1) = _mm_loadu_si128(&v100);
              v48[4] = v101;
              *(__m128i *)(v48 + 5) = _mm_loadu_si128(&v102);
              *(__m128i *)(v48 + 7) = _mm_loadu_si128(&v103);
              v49 = v104;
              *((_BYTE *)v48 + 80) = 0;
              v48[9] = v49;
            }
            v18 = 1;
            *((_BYTE *)v48 + 80) = v81;
            return v18;
          }
          v38 += 8;
        }
        while ( v88 != v38 );
      }
      return 0;
    }
  }
LABEL_4:
  v15 = *(_QWORD *)(a1 + 56);
  v96 = (__int64 *)v98;
  v97 = 0x400000000LL;
  if ( !v15 )
  {
    v50 = *(_DWORD *)(a2 + 20);
    v51 = &v102;
    v99.m128i_i64[0] = 0;
    v99.m128i_i64[1] = (__int64)&v102;
    v100.m128i_i64[0] = (__int64)&v102;
    v100.m128i_i64[1] = 4;
    LODWORD(v101) = 0;
    v52 = 3LL * (v50 & 0xFFFFFFF);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v53 = *(__int64 **)(a2 - 8);
      v54 = (unsigned __int64)&v53[v52];
    }
    else
    {
      v54 = a2;
      v53 = (__int64 *)(a2 - v52 * 8);
    }
    if ( (__int64 *)v54 == v53 )
      return 1;
    v92 = 0;
    v55 = &v102;
    v83 = a6;
    v89 = a4;
    v56 = (__int64 *)v54;
    v85 = v12;
    v57 = v53;
    while ( 1 )
    {
      v58 = *v57;
      v59 = *(_BYTE *)(*v57 + 16);
      if ( v59 <= 0x17u )
      {
        if ( v59 != 5 || !byte_4F97C60 || *(_WORD *)(v58 + 18) != 32 )
          goto LABEL_56;
      }
      else
      {
        if ( v59 == 77 )
        {
          if ( v55 != v51 )
            _libc_free((unsigned __int64)v55);
          goto LABEL_6;
        }
        if ( v59 != 56 || !byte_4F97C60 )
          goto LABEL_56;
      }
      if ( (*(_BYTE *)(v58 + 23) & 0x40) != 0 )
        v67 = *(_QWORD **)(v58 - 8);
      else
        v67 = (_QWORD *)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF));
      if ( a2 == *v67 && (*(_DWORD *)(v58 + 20) & 0xFFFFFFF) == 2 && *(_BYTE *)(v67[3] + 16LL) == 13 )
      {
        v92 = 1;
        goto LABEL_58;
      }
LABEL_56:
      if ( v55 == v51 )
      {
        v68 = (__m128i *)((char *)v55 + 8 * v100.m128i_u32[3]);
        if ( v55 != v68 )
        {
          v69 = 0;
          v70 = v55;
          do
          {
            if ( v58 == v70->m128i_i64[0] )
              goto LABEL_58;
            if ( v70->m128i_i64[0] == -2 )
              v69 = (__int64 *)v70;
            v70 = (__m128i *)((char *)v70 + 8);
          }
          while ( v68 != v70 );
          if ( v69 )
          {
            *v69 = v58;
            LODWORD(v101) = v101 - 1;
            ++v99.m128i_i64[0];
LABEL_92:
            v71 = (unsigned int)v97;
            if ( (unsigned int)v97 >= HIDWORD(v97) )
            {
              sub_16CD150(&v96, v98, 0, 8);
              v71 = (unsigned int)v97;
            }
            v96[v71] = v58;
            v55 = (__m128i *)v100.m128i_i64[0];
            LODWORD(v97) = v97 + 1;
            v51 = (__m128i *)v99.m128i_i64[1];
            goto LABEL_58;
          }
        }
        if ( v100.m128i_i32[3] < (unsigned __int32)v100.m128i_i32[2] )
        {
          ++v100.m128i_i32[3];
          v68->m128i_i64[0] = v58;
          ++v99.m128i_i64[0];
          goto LABEL_92;
        }
      }
      sub_16CCBA0(&v99, *v57);
      v55 = (__m128i *)v100.m128i_i64[0];
      v51 = (__m128i *)v99.m128i_i64[1];
      if ( v60 )
        goto LABEL_92;
LABEL_58:
      v57 += 3;
      if ( v56 == v57 )
      {
        a4 = v89;
        v12 = v85;
        a6 = v83;
        if ( v51 != v55 )
          _libc_free((unsigned __int64)v55);
        goto LABEL_61;
      }
    }
  }
  v16 = sub_14404E0(v15, a2);
  if ( (unsigned int)(*(_DWORD *)(v16 + 28) - *(_DWORD *)(v16 + 32)) > 6 )
  {
LABEL_6:
    v17 = v96;
    v18 = 1;
    goto LABEL_7;
  }
  v20 = *(__int64 **)(v16 + 16);
  v21 = &v20[*(unsigned int *)(v16 + 28)];
  if ( v20 != *(__int64 **)(v16 + 8) )
    v21 = &v20[*(unsigned int *)(v16 + 24)];
  if ( v20 == v21 )
    goto LABEL_15;
  while ( 1 )
  {
    v22 = *v20;
    if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v21 == ++v20 )
      goto LABEL_15;
  }
  if ( v21 == v20 )
  {
LABEL_15:
    v17 = v96;
    v18 = 1;
    goto LABEL_7;
  }
  v92 = 0;
  v72 = a2;
  v73 = v20;
  v74 = v72;
  while ( 1 )
  {
    if ( byte_4F97C60 )
    {
      v75 = *(_BYTE *)(v22 + 16);
      if ( v75 <= 0x17u )
      {
        if ( v75 != 5 || *(_WORD *)(v22 + 18) != 32 )
          goto LABEL_104;
      }
      else if ( v75 != 56 )
      {
        goto LABEL_104;
      }
      if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
        v78 = *(_QWORD **)(v22 - 8);
      else
        v78 = (_QWORD *)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF));
      if ( v74 == *v78 && (*(_DWORD *)(v22 + 20) & 0xFFFFFFF) == 2 && *(_BYTE *)(v78[3] + 16LL) == 13 )
      {
        v92 = byte_4F97C60;
        goto LABEL_107;
      }
    }
LABEL_104:
    v76 = (unsigned int)v97;
    if ( (unsigned int)v97 >= HIDWORD(v97) )
    {
      v82 = v74;
      v84 = v21;
      v87 = v22;
      sub_16CD150(&v96, v98, 0, 8);
      v76 = (unsigned int)v97;
      v74 = v82;
      v21 = v84;
      v22 = v87;
    }
    v96[v76] = v22;
    LODWORD(v97) = v97 + 1;
LABEL_107:
    v77 = v73 + 1;
    if ( v73 + 1 == v21 )
      goto LABEL_61;
    v22 = *v77;
    ++v73;
    if ( (unsigned __int64)*v77 >= 0xFFFFFFFFFFFFFFFELL )
      break;
LABEL_111:
    if ( v21 == v73 )
      goto LABEL_61;
  }
  while ( v21 != ++v77 )
  {
    v22 = *v77;
    v73 = v77;
    if ( (unsigned __int64)*v77 < 0xFFFFFFFFFFFFFFFELL )
      goto LABEL_111;
  }
LABEL_61:
  v17 = v96;
  v18 = 1;
  if ( (_DWORD)v97 )
  {
    v61 = -1;
    if ( !v92 )
      v61 = a3;
    v95 = v61;
    v18 = sub_1362890((__int64 *)a1, v12, a6, *v96, v61, a8, *(_OWORD *)a7, a7[2], *(_OWORD *)a4, a4[2], 0);
    if ( (_BYTE)v18 != 1 )
    {
      v86 = v97;
      if ( (_DWORD)v97 != 1 )
      {
        v62 = a6;
        v63 = v18;
        v90 = v12;
        v64 = 1;
        v65 = v62;
        do
        {
          v66 = sub_1362890((__int64 *)a1, v90, v65, v96[v64], v95, a8, *(_OWORD *)a7, a7[2], *(_OWORD *)a4, a4[2], 0);
          if ( v66 != (_BYTE)v63 )
          {
            if ( (v66 != 2 || (_BYTE)v63 != 3) && ((_BYTE)v63 != 2 || v66 != 3) )
              goto LABEL_6;
            v63 = 2;
          }
          ++v64;
        }
        while ( v64 != v86 );
        v18 = v63;
      }
    }
    v17 = v96;
  }
LABEL_7:
  if ( v17 != (__int64 *)v98 )
    _libc_free((unsigned __int64)v17);
  return v18;
}
