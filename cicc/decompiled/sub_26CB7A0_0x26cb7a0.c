// Function: sub_26CB7A0
// Address: 0x26cb7a0
//
void __fastcall sub_26CB7A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r9
  __int64 *v6; // rbx
  __int64 *v7; // r13
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edx
  _QWORD *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r12
  _QWORD *v14; // rcx
  int v15; // eax
  __int64 v16; // rax
  __int64 *v17; // rbx
  unsigned __int64 v18; // rcx
  __m128i *v19; // r14
  int v20; // eax
  __int64 *v21; // r14
  bool v22; // zf
  _QWORD *v23; // rax
  __int64 *v24; // rbx
  __int64 v25; // r13
  int v26; // r11d
  __int64 *v27; // r10
  unsigned int v28; // edx
  __int64 *v29; // rax
  int v30; // esi
  int v31; // edx
  int v32; // esi
  _QWORD *v33; // rdx
  int v34; // eax
  int v35; // esi
  int v36; // edx
  __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 *v39; // r14
  __int64 v40; // r15
  __int64 *v41; // rbx
  unsigned int v42; // edx
  __int64 v43; // r8
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rbx
  const __m128i *v47; // r15
  unsigned __int64 v48; // rax
  __m128i *v49; // rbx
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  __m128i *i; // rax
  bool v53; // cc
  __m128i v54; // xmm0
  unsigned __int64 v55; // rdx
  _QWORD *v56; // rax
  __int64 v57; // [rsp+8h] [rbp-158h]
  int v58; // [rsp+10h] [rbp-150h]
  __int64 v59; // [rsp+20h] [rbp-140h] BYREF
  _QWORD *v60; // [rsp+28h] [rbp-138h] BYREF
  __int64 v61; // [rsp+30h] [rbp-130h] BYREF
  __int64 *v62; // [rsp+38h] [rbp-128h]
  __int64 v63; // [rsp+40h] [rbp-120h]
  unsigned int v64; // [rsp+48h] [rbp-118h]
  __int64 *v65; // [rsp+50h] [rbp-110h] BYREF
  unsigned int v66; // [rsp+58h] [rbp-108h]
  char v67; // [rsp+60h] [rbp-100h] BYREF
  void *src; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+A8h] [rbp-B8h]
  _BYTE v70[176]; // [rsp+B0h] [rbp-B0h] BYREF

  if ( !(_DWORD)qword_4FF62C8 )
    return;
  v59 = 0;
  sub_ED2710((__int64)&v65, a1, 0, qword_4FF62C8, &v59, 1u);
  v6 = v65;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  if ( a3 )
  {
    v21 = &v65[2 * v66];
    if ( v21 != v65 )
    {
      while ( 1 )
      {
        while ( v6[1] != -1 )
        {
          v6 += 2;
          if ( v21 == v6 )
            goto LABEL_32;
        }
        v22 = (unsigned __int8)sub_26C3510((__int64)&v61, v6, &v60) == 0;
        v23 = v60;
        if ( !v22 )
          goto LABEL_31;
        v35 = v64;
        src = v60;
        ++v61;
        v36 = v63 + 1;
        v5 = (unsigned int)(4 * (v63 + 1));
        if ( (unsigned int)v5 >= 3 * v64 )
          break;
        if ( v64 - HIDWORD(v63) - v36 <= v64 >> 3 )
          goto LABEL_100;
LABEL_51:
        LODWORD(v63) = v36;
        if ( *v23 != -1 )
          --HIDWORD(v63);
        v37 = *v6;
        v23[1] = 0;
        *v23 = v37;
LABEL_31:
        v6 += 2;
        v23[1] = -1;
        if ( v21 == v6 )
          goto LABEL_32;
      }
      v35 = 2 * v64;
LABEL_100:
      sub_9D80B0((__int64)&v61, v35);
      sub_26C3510((__int64)&v61, v6, &src);
      v36 = v63 + 1;
      v23 = src;
      goto LABEL_51;
    }
LABEL_32:
    v24 = *(__int64 **)a2;
    v25 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( v25 == *(_QWORD *)a2 )
      goto LABEL_18;
    while ( 1 )
    {
      v30 = v64;
      if ( !v64 )
        break;
      v26 = 1;
      v27 = 0;
      v28 = (v64 - 1) & (((0xBF58476D1CE4E5B9LL * *v24) >> 31) ^ (484763065 * *(_DWORD *)v24));
      v29 = &v62[2 * v28];
      v5 = *v29;
      if ( *v24 != *v29 )
      {
        while ( v5 != -1 )
        {
          if ( v27 || v5 != -2 )
            v29 = v27;
          v28 = (v64 - 1) & (v26 + v28);
          v5 = v62[2 * v28];
          if ( *v24 == v5 )
            goto LABEL_35;
          ++v26;
          v27 = v29;
          v29 = &v62[2 * v28];
        }
        if ( !v27 )
          v27 = v29;
        ++v61;
        v31 = v63 + 1;
        src = v27;
        if ( 4 * ((int)v63 + 1) < 3 * v64 )
        {
          if ( v64 - HIDWORD(v63) - v31 <= v64 >> 3 )
          {
LABEL_40:
            sub_9D80B0((__int64)&v61, v30);
            sub_26C3510((__int64)&v61, v24, &src);
            v27 = (__int64 *)src;
            v31 = v63 + 1;
          }
          LODWORD(v63) = v31;
          if ( *v27 != -1 )
            --HIDWORD(v63);
          *v27 = *v24;
          v27[1] = v24[1];
          goto LABEL_36;
        }
LABEL_39:
        v30 = 2 * v64;
        goto LABEL_40;
      }
LABEL_35:
      a3 -= v24[1];
LABEL_36:
      v24 += 2;
      if ( (__int64 *)v25 == v24 )
        goto LABEL_18;
    }
    ++v61;
    src = 0;
    goto LABEL_39;
  }
  v7 = &v65[2 * v66];
  if ( v65 != v7 )
  {
    v8 = 0;
    v9 = 0;
    while ( 1 )
    {
      v13 = v6[1];
      if ( !v8 )
        break;
      v10 = (v8 - 1) & (((0xBF58476D1CE4E5B9LL * *v6) >> 31) ^ (484763065 * *(_DWORD *)v6));
      v11 = (_QWORD *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( *v6 == *v11 )
      {
LABEL_7:
        v6 += 2;
        v11[1] = v13;
        if ( v7 == v6 )
          goto LABEL_16;
        goto LABEL_8;
      }
      v58 = 1;
      v14 = 0;
      while ( v12 != -1 )
      {
        if ( v12 == -2 && !v14 )
          v14 = v11;
        v10 = (v8 - 1) & (v58 + v10);
        v11 = (_QWORD *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( *v6 == *v11 )
          goto LABEL_7;
        ++v58;
      }
      if ( !v14 )
        v14 = v11;
      ++v61;
      v15 = v63 + 1;
      src = v14;
      if ( 4 * ((int)v63 + 1) >= 3 * v8 )
        goto LABEL_11;
      if ( v8 - (v15 + HIDWORD(v63)) <= v8 >> 3 )
        goto LABEL_12;
LABEL_13:
      LODWORD(v63) = v15;
      if ( *v14 != -1 )
        --HIDWORD(v63);
      v16 = *v6;
      v6 += 2;
      v14[1] = 0;
      *v14 = v16;
      v14[1] = v13;
      if ( v7 == v6 )
        goto LABEL_16;
LABEL_8:
      v9 = (__int64)v62;
      v8 = v64;
    }
    ++v61;
    src = 0;
LABEL_11:
    v8 *= 2;
LABEL_12:
    sub_9D80B0((__int64)&v61, v8);
    sub_26C3510((__int64)&v61, v6, &src);
    v14 = src;
    v15 = v63 + 1;
    goto LABEL_13;
  }
LABEL_16:
  v17 = *(__int64 **)a2;
  if ( (unsigned __int8)sub_26C3510((__int64)&v61, *(__int64 **)a2, &v60) )
  {
    a3 = v59 - v60[1];
    v60[1] = -1;
    v59 = a3;
    goto LABEL_18;
  }
  v32 = v64;
  v33 = v60;
  ++v61;
  v34 = v63 + 1;
  src = v60;
  if ( 4 * ((int)v63 + 1) >= 3 * v64 )
  {
    v32 = 2 * v64;
    goto LABEL_102;
  }
  if ( v64 - HIDWORD(v63) - v34 <= v64 >> 3 )
  {
LABEL_102:
    sub_9D80B0((__int64)&v61, v32);
    sub_26C3510((__int64)&v61, v17, &src);
    v33 = src;
    v34 = v63 + 1;
  }
  LODWORD(v63) = v34;
  if ( *v33 != -1 )
    --HIDWORD(v63);
  *v33 = *v17;
  a3 = v59;
  v33[1] = v17[1];
LABEL_18:
  src = v70;
  v69 = 0x800000000LL;
  if ( !(_DWORD)v63 )
    goto LABEL_19;
  v38 = v62;
  v39 = &v62[2 * v64];
  if ( v62 == v39 )
    goto LABEL_19;
  while ( 1 )
  {
    v40 = *v38;
    v41 = v38;
    if ( (unsigned __int64)*v38 <= 0xFFFFFFFFFFFFFFFDLL )
      break;
    v38 += 2;
    if ( v39 == v38 )
      goto LABEL_19;
  }
  if ( v38 == v39 )
  {
LABEL_19:
    v18 = 0;
    v19 = (__m128i *)v70;
  }
  else
  {
    v42 = 0;
    v43 = v38[1];
    v44 = 0;
LABEL_60:
    v45 = (char *)src + 16 * v44;
    if ( v45 )
    {
      *v45 = v40;
      v45[1] = v43;
      v42 = v69;
    }
    for ( LODWORD(v69) = ++v42; ; LODWORD(v69) = v69 + 1 )
    {
      v41 += 2;
      if ( v41 == v39 )
        break;
      while ( (unsigned __int64)*v41 > 0xFFFFFFFFFFFFFFFDLL )
      {
        v41 += 2;
        if ( v39 == v41 )
          goto LABEL_66;
      }
      if ( v41 == v39 )
        break;
      v44 = v42;
      v40 = *v41;
      v43 = v41[1];
      if ( v42 < (unsigned __int64)HIDWORD(v69) )
        goto LABEL_60;
      v55 = v42 + 1LL;
      if ( HIDWORD(v69) < (unsigned __int64)(v44 + 1) )
      {
        v57 = v41[1];
        sub_C8D5F0((__int64)&src, v70, v55, 0x10u, v43, v5);
        v44 = (unsigned int)v69;
        v43 = v57;
      }
      v56 = (char *)src + 16 * v44;
      *v56 = v40;
      v56[1] = v43;
      v42 = v69 + 1;
    }
LABEL_66:
    v18 = v42;
    v19 = (__m128i *)src;
    v46 = 16LL * v42;
    v47 = (const __m128i *)((char *)src + v46);
    if ( src != (char *)src + v46 )
    {
      _BitScanReverse64(&v48, v46 >> 4);
      sub_26BA530((__m128i *)src, (unsigned __int64 *)((char *)src + v46), 2LL * (int)(63 - (v48 ^ 0x3F)));
      if ( (unsigned __int64)v46 <= 0x100 )
      {
        sub_26BA340(v19, v47);
      }
      else
      {
        v49 = v19 + 16;
        sub_26BA340(v19, v19 + 16);
        if ( &v19[16] != v47 )
        {
          do
          {
            v50 = v49->m128i_i64[0];
            v51 = v49->m128i_u64[1];
            for ( i = v49; ; i[1] = v54 )
            {
              v53 = v51 <= i[-1].m128i_i64[1];
              if ( v51 == i[-1].m128i_i64[1] )
                v53 = v50 <= i[-1].m128i_i64[0];
              if ( v53 )
                break;
              v54 = _mm_loadu_si128(--i);
            }
            ++v49;
            i->m128i_i64[0] = v50;
            i->m128i_i64[1] = v51;
          }
          while ( v49 != v47 );
        }
      }
      v19 = (__m128i *)src;
      v18 = (unsigned int)v69;
    }
  }
  v20 = qword_4FF62C8;
  if ( (unsigned int)qword_4FF62C8 > v18 )
    v20 = v18;
  sub_ED2230(*(__int64 ***)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL) + 40LL), a1, v19->m128i_i64, v18, a3, 0, v20);
  if ( src != v70 )
    _libc_free((unsigned __int64)src);
  sub_C7D6A0((__int64)v62, 16LL * v64, 8);
  if ( v65 != (__int64 *)&v67 )
    _libc_free((unsigned __int64)v65);
}
