// Function: sub_2871CE0
// Address: 0x2871ce0
//
unsigned __int64 __fastcall sub_2871CE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8)
{
  unsigned __int64 v8; // rdx
  unsigned __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  const __m128i *v12; // r15
  __int64 v13; // rax
  bool v14; // zf
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // r13
  __int64 v18; // r12
  __int64 *v19; // rbx
  __int64 v20; // rcx
  __int32 v21; // edx
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // rcx
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  __int32 v29; // eax
  __int64 v30; // r11
  bool v31; // r13
  __int64 *v32; // r10
  __int64 v33; // r11
  __int64 *v34; // r14
  _QWORD *v35; // rdi
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rcx
  int v39; // eax
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  __int64 *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 *v45; // rax
  unsigned int v46; // ecx
  __int64 *v47; // rdx
  __int64 v48; // rdi
  __int64 *v49; // r11
  int v50; // eax
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __m128i v53; // xmm3
  __m128i v54; // xmm4
  __m128i v55; // xmm5
  __int32 v56; // eax
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  unsigned __int64 v59; // rax
  int v60; // r12d
  size_t v61; // r13
  __int64 v62; // rdx
  const void *v63; // rsi
  __int64 *v64; // rdx
  __int64 v65; // rcx
  int v66; // edi
  __int64 *v67; // rsi
  int v68; // edi
  unsigned int v69; // ecx
  __int64 v71; // [rsp+20h] [rbp-1D0h]
  int v73; // [rsp+30h] [rbp-1C0h]
  unsigned int v74; // [rsp+30h] [rbp-1C0h]
  __int64 v76; // [rsp+48h] [rbp-1A8h]
  __int64 v77; // [rsp+50h] [rbp-1A0h]
  __int64 v79; // [rsp+58h] [rbp-198h]
  __int64 v80[6]; // [rsp+60h] [rbp-190h] BYREF
  __m128i v81; // [rsp+90h] [rbp-160h] BYREF
  __m128i v82; // [rsp+A0h] [rbp-150h] BYREF
  __m128i v83; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v84; // [rsp+C0h] [rbp-130h]
  __int32 v85; // [rsp+C8h] [rbp-128h]
  __int64 v86; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v87; // [rsp+D8h] [rbp-118h]
  __int64 v88; // [rsp+E0h] [rbp-110h]
  __int64 v89; // [rsp+E8h] [rbp-108h]
  __int64 *v90; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v91; // [rsp+F8h] [rbp-F8h]
  _BYTE v92[32]; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v93; // [rsp+120h] [rbp-D0h] BYREF
  _BYTE *v94; // [rsp+128h] [rbp-C8h]
  __int64 v95; // [rsp+130h] [rbp-C0h]
  int v96; // [rsp+138h] [rbp-B8h]
  char v97; // [rsp+13Ch] [rbp-B4h]
  _BYTE v98[176]; // [rsp+140h] [rbp-B0h] BYREF

  v71 = a6;
  v8 = *a7 + 1;
  *a7 = v8;
  result = (unsigned int)qword_5001308 >> 1;
  if ( v8 >= result )
    return result;
  v10 = *(unsigned int *)(a4 + 8);
  v11 = a1;
  v86 = 0;
  v12 = (const __m128i *)a5;
  v87 = 0;
  v13 = *(_QWORD *)(a1 + 1320) + 2184 * v10;
  v14 = *(_BYTE *)(a6 + 28) == 0;
  v88 = 0;
  v79 = v13;
  v90 = (__int64 *)v92;
  v91 = 0x400000000LL;
  v15 = *(__int64 **)(a6 + 8);
  v89 = 0;
  if ( v14 )
    v16 = *(unsigned int *)(a6 + 16);
  else
    v16 = *(unsigned int *)(a6 + 20);
  v17 = &v15[v16];
  if ( v15 != v17 )
  {
    while ( 1 )
    {
      v18 = *v15;
      v19 = v15;
      if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v17 == ++v15 )
        goto LABEL_7;
    }
    if ( v17 != v15 )
    {
      while ( 1 )
      {
        v93 = v18;
        if ( *(_BYTE *)(v79 + 2148) )
        {
          v40 = *(_QWORD **)(v79 + 2128);
          v41 = &v40[*(unsigned int *)(v79 + 2140)];
          if ( v40 == v41 )
            goto LABEL_45;
          while ( v18 != *v40 )
          {
            if ( v41 == ++v40 )
              goto LABEL_45;
          }
        }
        else if ( !sub_C8CA60(v79 + 2120, v18) )
        {
          goto LABEL_45;
        }
        if ( !(_DWORD)v88 )
        {
          v42 = &v90[(unsigned int)v91];
          if ( v42 == sub_284FC00(v90, (__int64)v42, &v93) )
            sub_2871A60((__int64)&v86, v18, v43, v44, a5, a6);
          goto LABEL_45;
        }
        if ( (_DWORD)v89 )
        {
          a6 = v87;
          a5 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
          v46 = (v89 - 1) & a5;
          v47 = (__int64 *)(v87 + 8LL * v46);
          v48 = *v47;
          if ( v18 == *v47 )
            goto LABEL_45;
          v73 = 1;
          v49 = 0;
          while ( v48 != -4096 )
          {
            if ( v48 == -8192 && !v49 )
              v49 = v47;
            v46 = (v89 - 1) & (v73 + v46);
            v47 = (__int64 *)(v87 + 8LL * v46);
            v48 = *v47;
            if ( v18 == *v47 )
              goto LABEL_45;
            ++v73;
          }
          if ( v49 )
            v47 = v49;
          ++v86;
          v50 = v88 + 1;
          if ( 4 * ((int)v88 + 1) < (unsigned int)(3 * v89) )
          {
            if ( (int)v89 - HIDWORD(v88) - v50 > (unsigned int)v89 >> 3 )
              goto LABEL_61;
            v74 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
            sub_2871610((__int64)&v86, v89);
            if ( !(_DWORD)v89 )
            {
LABEL_109:
              LODWORD(v88) = v88 + 1;
              BUG();
            }
            a5 = v74;
            v68 = 1;
            v67 = 0;
            v69 = (v89 - 1) & v74;
            v47 = (__int64 *)(v87 + 8LL * v69);
            a6 = *v47;
            v50 = v88 + 1;
            if ( v18 == *v47 )
              goto LABEL_61;
            while ( a6 != -4096 )
            {
              if ( !v67 && a6 == -8192 )
                v67 = v47;
              a5 = (unsigned int)(v68 + 1);
              v69 = (v89 - 1) & (v68 + v69);
              v47 = (__int64 *)(v87 + 8LL * v69);
              a6 = *v47;
              if ( v18 == *v47 )
                goto LABEL_61;
              ++v68;
            }
            goto LABEL_88;
          }
        }
        else
        {
          ++v86;
        }
        sub_2871610((__int64)&v86, 2 * v89);
        if ( !(_DWORD)v89 )
          goto LABEL_109;
        a5 = (unsigned int)(v89 - 1);
        LODWORD(v65) = a5 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v47 = (__int64 *)(v87 + 8LL * (unsigned int)v65);
        a6 = *v47;
        v50 = v88 + 1;
        if ( v18 == *v47 )
          goto LABEL_61;
        v66 = 1;
        v67 = 0;
        while ( a6 != -4096 )
        {
          if ( a6 == -8192 && !v67 )
            v67 = v47;
          v65 = (unsigned int)a5 & ((_DWORD)v65 + v66);
          v47 = (__int64 *)(v87 + 8 * v65);
          a6 = *v47;
          if ( v18 == *v47 )
            goto LABEL_61;
          ++v66;
        }
LABEL_88:
        if ( v67 )
          v47 = v67;
LABEL_61:
        LODWORD(v88) = v50;
        if ( *v47 != -4096 )
          --HIDWORD(v88);
        *v47 = v18;
        v51 = (unsigned int)v91;
        v52 = (unsigned int)v91 + 1LL;
        if ( v52 > HIDWORD(v91) )
        {
          sub_C8D5F0((__int64)&v90, v92, v52, 8u, a5, a6);
          v51 = (unsigned int)v91;
        }
        v90[v51] = v18;
        LODWORD(v91) = v91 + 1;
LABEL_45:
        v45 = v19 + 1;
        if ( v19 + 1 != v17 )
        {
          while ( 1 )
          {
            v18 = *v45;
            v19 = v45;
            if ( (unsigned __int64)*v45 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v17 == ++v45 )
              goto LABEL_48;
          }
          if ( v17 != v45 )
            continue;
        }
LABEL_48:
        v11 = a1;
        break;
      }
    }
  }
LABEL_7:
  v20 = *(_QWORD *)(v11 + 8);
  v93 = 0;
  v94 = v98;
  v21 = *(_DWORD *)(v11 + 72);
  v22 = *(_QWORD *)(v11 + 48);
  v95 = 16;
  v23 = *(_QWORD *)(v11 + 56);
  v82 = (__m128i)v22;
  v96 = 0;
  v97 = 1;
  v24 = *(_QWORD *)(v79 + 760);
  v81.m128i_i64[1] = v20;
  v25 = *(unsigned int *)(v79 + 768);
  v81.m128i_i64[0] = v23;
  v85 = v21;
  v83 = 0u;
  v84 = 0;
  v76 = v24 + 112 * v25;
  if ( v76 == v24 )
    goto LABEL_27;
  v77 = v11;
  while ( v21 != 1 || *(_DWORD *)(v79 + 32) != 2 )
  {
    a6 = *(_QWORD *)(v24 + 88);
    v30 = *(unsigned int *)(v24 + 48);
    v25 = (__int64)v90;
    v31 = a6 != 0;
    v32 = &v90[(unsigned int)v91];
    a5 = v30 + (a6 != 0);
    if ( a5 > (unsigned int)v91 )
      a5 = (unsigned int)v91;
    if ( v32 != v90 )
    {
      v33 = v30;
      v34 = v90;
      do
      {
        if ( (v80[0] = *v34, v80[0] == a6) && v31
          || (v35 = *(_QWORD **)(v24 + 40), &v35[v33] != sub_284FCC0(v35, (__int64)&v35[v33], v80)) )
        {
          a5 = (unsigned int)(a5 - 1);
          if ( !(_DWORD)a5 )
            goto LABEL_10;
        }
        ++v34;
      }
      while ( v32 != v34 );
    }
    if ( !(_DWORD)a5 )
      break;
    v24 += 112;
    if ( v24 == v76 )
      goto LABEL_25;
LABEL_12:
    v21 = *(_DWORD *)(v77 + 72);
  }
LABEL_10:
  v26 = _mm_loadu_si128(v12);
  v27 = _mm_loadu_si128(v12 + 1);
  v28 = _mm_loadu_si128(v12 + 2);
  v84 = v12[3].m128i_i64[0];
  v29 = v12[3].m128i_i32[2];
  v81 = v26;
  v82 = v27;
  v85 = v29;
  v83 = v28;
  sub_C8CE00((__int64)&v93, (__int64)v98, v71, v25, a5, a6);
  sub_285BFD0((__int64)&v81, v24, (__int64)&v93, a8, v79, 0);
  if ( !sub_2851E20((__int64)&v81, a3) )
  {
LABEL_11:
    v24 += 112;
    if ( v24 == v76 )
      goto LABEL_25;
    goto LABEL_12;
  }
  v36 = *(unsigned int *)(a4 + 8);
  if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v36 + 1, 8u, a5, a6);
    v36 = *(unsigned int *)(a4 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * v36) = v24;
  v37 = *(_DWORD *)(a4 + 8) + 1;
  *(_DWORD *)(a4 + 8) = v37;
  if ( *(_DWORD *)(v77 + 1328) == v37 )
  {
    v53 = _mm_loadu_si128(&v81);
    v54 = _mm_loadu_si128(&v82);
    *(_QWORD *)(a3 + 48) = v84;
    v55 = _mm_loadu_si128(&v83);
    v56 = v85;
    *(__m128i *)a3 = v53;
    *(__m128i *)(a3 + 16) = v54;
    *(_DWORD *)(a3 + 56) = v56;
    v57 = a4;
    *(__m128i *)(a3 + 32) = v55;
    if ( a4 != a2 )
    {
      v58 = *(unsigned int *)(a4 + 8);
      v59 = *(unsigned int *)(a2 + 8);
      v60 = *(_DWORD *)(a4 + 8);
      if ( v58 <= v59 )
      {
        if ( *(_DWORD *)(a4 + 8) )
          memmove(*(void **)a2, *(const void **)a4, 8 * v58);
      }
      else
      {
        if ( v58 > *(unsigned int *)(a2 + 12) )
        {
          v61 = 0;
          *(_DWORD *)(a2 + 8) = 0;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v58, 8u, a5, a6);
          v58 = *(unsigned int *)(a4 + 8);
        }
        else
        {
          v61 = 8 * v59;
          if ( *(_DWORD *)(a2 + 8) )
          {
            memmove(*(void **)a2, *(const void **)a4, v61);
            v58 = *(unsigned int *)(a4 + 8);
          }
        }
        v62 = 8 * v58;
        v63 = (const void *)(*(_QWORD *)a4 + v61);
        if ( v63 != (const void *)(v62 + *(_QWORD *)a4) )
          memcpy((void *)(v61 + *(_QWORD *)a2), v63, v62 - v61);
      }
      *(_DWORD *)(a2 + 8) = v60;
      v57 = a4;
    }
    v39 = *(_DWORD *)(v57 + 8);
    goto LABEL_35;
  }
  sub_2871CE0(v77, a2, a3, a4, (unsigned int)&v81, (unsigned int)&v93, (__int64)a7, a8);
  if ( *a7 < (unsigned int)qword_5001308 >> 1 )
  {
    v38 = *(_QWORD *)(v24 + 88);
    v39 = *(_DWORD *)(a4 + 8);
    if ( *(unsigned int *)(v24 + 48) - ((v38 == 0) - 1LL) == 1 && v39 == 1 )
    {
      v64 = (__int64 *)(v24 + 88);
      if ( !v38 )
        v64 = *(__int64 **)(v24 + 40);
      sub_28717E0((__int64)v80, a8, v64);
      v39 = *(_DWORD *)(a4 + 8);
    }
LABEL_35:
    v25 = a4;
    *(_DWORD *)(a4 + 8) = v39 - 1;
    goto LABEL_11;
  }
LABEL_25:
  if ( !v97 )
    _libc_free((unsigned __int64)v94);
LABEL_27:
  if ( v90 != (__int64 *)v92 )
    _libc_free((unsigned __int64)v90);
  return sub_C7D6A0(v87, 8LL * (unsigned int)v89, 8);
}
